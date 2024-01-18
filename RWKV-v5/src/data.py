from lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

import math
import wandb
from datasets import load_from_disk, load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from multiprocessing import cpu_count
num_cpus = cpu_count()
num_workers = cpu_count() if cpu_count() < 8 else 8

# Get the script directory
import os
SRC_DIR = os.path.dirname(os.path.realpath(__file__))

# World tokenizer
from .dataflow.trie_tokenizer import world_tokenizer_encode
import numpy as np

# We have to extract out the prepare function to be "outside the class"
# else it will not be hashed / serialized properly, and will produce the following error:
#
# ```
# Parameter 'function'=<function RWKVDataModule.prepare_data.<locals>.map_tokenizer at 0x7f7672c5e340> of the transform datasets.arrow_dataset.Dataset._map_single couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
# ```
def prepare_data_static(**kargs):

    # Check if skip_datapath_setup is enabled
    # useful for extra large datasets
    if kargs["skip_datapath_setup"] == True:
        return

    # Special handling of world_add_endoftext_token (if enabled)
    if kargs["world_add_endoftext_token"]:
        world_add_endoftext_token = True
    else:
        world_add_endoftext_token = False

    # Source data processing
    if kargs["source"] is not None:
        if kargs["tokenizer"] is None:
            raise ValueError('Tokenizer must be specified if source is specified')
        
        # =====================================================

        # Util functions
        #--------------------------------

        # Apply the data_prefix_skip_mask to the given mask
        # where relevent, and disables the training mask for the first X tokens
        data_prefix_skip_mask_val = int(kargs["data_prefix_skip_mask"])
        def apply_data_prefix_skip_mask(mask):
            mask_len = len(mask)
            if data_prefix_skip_mask_val > 0 and mask_len:
                for i in range(max(data_prefix_skip_mask_val, mask_len)):
                    mask[i] = 0
            return mask
        
        # Special handling for binidx
        #--------------------------------

        # TODO: verify this works, i have a suspicion this just creates a new "document" for each token.
        if kargs["tokenizer"] == "binidx":
            from .dataflow.binidx import MMapIndexedDataset

            # Load the MMapIndexedDataset from the source path
            mmap_dataset = MMapIndexedDataset(kargs["source"])
            mmap_dataset_len = mmap_dataset.__len__()

            # Torch dataset generator wrapper
            def gen():
                for idx in range(mmap_dataset_len):
                    # cast to supported types, note that np.int32 limit is 2,147,483,647 
                    # - so unless we have a tokenizer that exceeds this, it should be ok
                    tokens = np.array(mmap_dataset.get(idx), dtype=np.int32)
                    yield {
                        'input_ids': tokens,
                        'token_type_ids': [0] * len(tokens),
                        'attention_mask': apply_data_prefix_skip_mask([1] * len(tokens))
                    }

            # Load the huggingface dataset from the generator
            raw_src_dataset = Dataset.from_generator(gen)

            # Previous short cut save for binidx, disabled to support chunking/packing
            # ----------------------
            # Train/test split
            test_split = kargs["test_split"]
            # The minimum test size is 1, if not we will get errors in the trainer?
            if test_split <= 0 or test_split <= 0.0:
                test_split = 1
            
            # Force a split, to normlize the dataset format
            src_dataset = raw_src_dataset.train_test_split(
                test_size=test_split,shuffle=kargs["test_split_shuffle"],
                seed=42 #Fixed seed, to prevent train/test reshuffling between test runs
            )
            
            # # Save the dataset to disk
            # split_dataset.save_to_disk(kargs["data_path"])
            # # Does nothing else (done)
            # return

        else:
            # Reverting back to general purpose HF dataset / tokenizer handling
            #--------------------------------
            load_dataset_params = {
                'path': kargs["source"],
                'num_proc': num_cpus
            }

            # Handle advance params (if set)
            if kargs["source_data_dir"] is not None:
                load_dataset_params['data_dir'] = kargs["source_data_dir"]
            if kargs["source_dataset_params"] is not None:
                source_dataset_params = kargs["source_dataset_params"]
                for k, v in source_dataset_params.items():
                    load_dataset_params[k] = v

            # Load the dataset
            src_dataset = load_dataset(**load_dataset_params)

            # If for some reason the dataset is a "test" only split, and missing a "train" split, we remap it as a "train" split
            if "train" not in src_dataset.keys():
                if "test" in src_dataset.keys():
                    src_dataset["train"] = src_dataset["test"]
                    del src_dataset["test"]
                else:
                    raise ValueError('Dataset must have a "train" split')

            # If an int value is used, it is interprated as document count
            # If a floating value (<1.0) is used, it is interprated as a percentage of the dataset
            if kargs["dataset_offset"] > 0 or kargs["dataset_length"] > 0:
                # src dataset length
                train_length = len(src_dataset["train"])

                # Compute the offset position
                offset_val = kargs["dataset_offset"]

                # If offset is a float, we will use it as a percentage
                if offset_val < 0:
                    offset_val = 0
                if offset_val > 0 and offset_val < 1.0:
                    offset_val = int(train_length * offset_val) # Rounded down value

                # Compute the length position
                length_val = kargs["dataset_length"]
                if length_val < 0:
                    length_val = train_length - offset_val
                if length_val > 0 and length_val < 1.0:
                    length_val = int(train_length * length_val)
                if length_val > (train_length - offset_val):
                    length_val = (train_length - offset_val)

                # Get the subset of the dataset
                src_dataset["train"] = src_dataset["train"].select(range(offset_val, offset_val + length_val))

            # Tokenizer vars
            hf_tokenizer = None
            world_tokenizer = None

            # Load the tokenizer according to either its predefined name or its path
            # (defaults to neox)
            if kargs["tokenizer"] == "neox":
                tokenizer_file = os.path.join(SRC_DIR, "./dataflow/20B_tokenizer.json")
                hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            elif kargs["tokenizer"] == "world":
                # Setup the tokenizer
                world_tokenizer = True
            else:
                # AutoTokenizer
                tokenizerName = kargs["tokenizer"]

                # with custom args and props
                tokenizerKWArgs = {}
                tokenizerProps = {}
                if kargs["autoTokenizer"] is not None:
                    if kargs["autoTokenizer"]["kwargs"] is not None:
                        tokenizerKWArgs = kargs["autoTokenizer"]["kwargs"]
                    if kargs["autoTokenizer"]["props"] is not None:
                        tokenizerProps  = kargs["autoTokenizer"]["props"]

                # Intialize the tokenizer, with kwargs
                hf_tokenizer = AutoTokenizer.from_pretrained(tokenizerName, **tokenizerKWArgs)

                # Configure the tokenizer properties
                for k, v in tokenizerProps.items():
                    setattr(hf_tokenizer, k, v)

            # Function used to tokenize the dataset as per HF tokenizer format
            # if given the textual data, it will return the tokenized data
            def encodeTokens(x):
                if world_tokenizer is True:
                    # If x is an array of strings, we encode them seperately, and conslidate the result
                    if isinstance(x, list):
                        id_arr = []
                        type_arr = []
                        mask_arr = []
                        for i in range(len(x)):
                            enc_str = world_tokenizer_encode(x[i], world_add_endoftext_token=world_add_endoftext_token)
                            id_arr.append(enc_str)
                            type_arr.append([0] * len(enc_str))
                            mask_arr.append([1] * len(enc_str))

                        # Consolidate the result
                        return {
                            'input_ids': id_arr,
                            'token_type_ids': type_arr,
                            'attention_mask': mask_arr
                        }
                    
                    # Else we encode the string and return it following the HF tokenizer format
                    enc_str = world_tokenizer_encode(x, world_add_endoftext_token=world_add_endoftext_token)
                    return {
                        'input_ids': enc_str,
                        'token_type_ids': [0] * len(enc_str),
                        'attention_mask': [1] * len(enc_str)
                    }

                # We use the HF tokenizer as it is, and get the input_ids
                return hf_tokenizer(x)
            
            # Multi column merging default values setup
            if kargs["multi_column_keys"] is None:
                multi_column_keys = ['instruction', 'input', 'output']
                multi_column_prefix = ['Instruction:\n', 'Input:\n', 'Output:\n']
                multi_column_suffix = ['', '', '']
                multi_column_train_mask = [True, False, True]
                multi_column_separator = '\n\n'
            else:
                multi_column_keys = kargs["multi_column_keys"]
                multi_column_prefix = kargs["multi_column_prefix"]
                multi_column_suffix = kargs["multi_column_suffix"]
                multi_column_train_mask = kargs["multi_column_train_mask"]
                multi_column_separator = kargs["multi_column_separator"]
            
            # Tokenized encodings for multi column keys
            multi_column_enabled = len(multi_column_keys) > 0
            multi_column_prefix_encodings = []
            multi_column_suffix_encodings = []
            multi_column_separator_encodings = None

            # Process the multi column settings
            if multi_column_enabled:
                
                # Tokenize the multi column strings
                for i in range(len(multi_column_keys)):
                    if multi_column_prefix is not None and multi_column_prefix[i] is not None:
                        multi_column_prefix_encodings.append(encodeTokens(multi_column_prefix[i]))
                    if multi_column_suffix is not None and multi_column_suffix[i] is not None:
                        multi_column_suffix_encodings.append(encodeTokens(multi_column_suffix[i]))    
                
                # Tokenize the multi column separator
                if multi_column_separator is not None and len(multi_column_separator) > 0:
                    multi_column_separator_encodings = encodeTokens(multi_column_separator)

            conversation_prefix_encoding_map = {}
            conversation_suffix_encoding_map = {}
            conversation_end_of_conversation_token = encodeTokens(kargs["conversation_end_of_conversation"]) if kargs["conversation_end_of_conversation"] is not None else None
            conversation_enabled = False
            if 'conversation_format' in kargs and kargs["conversation_format"] is not None:
                if kargs["conversation_format"] == "iopairs":
                    # preencode all prefixes (keyed by the input key)
                    for key, prefix in kargs['conversation_input_key_prefix_map'].items():
                        conversation_prefix_encoding_map[key] = encodeTokens(prefix)
                    conversation_enabled = True
                elif kargs["conversation_format"] == "sender":
                    # preencode all prefixes (keyed by the sender value)
                    for key, relabel in kargs['conversation_sender_value_map'].items():
                        for input_key, value in kargs['conversation_input_key_map'].items():
                            if input_key not in conversation_prefix_encoding_map:
                                conversation_prefix_encoding_map[input_key] = {}
                            conversation_prefix_encoding_map[input_key][key] = encodeTokens(value.replace('{sender}', relabel))

                for key, suffix in kargs['conversation_sender_suffix'].items():
                    conversation_suffix_encoding_map[key] = encodeTokens(suffix)
                            # example conversation_prefix_encoding_map['message']['user'] = encodeTokens('\n\nUser:')

                    conversation_enabled = True

            # Maps the dataset record to the tokenized result
            # handles a wide variety of format according to the data configuration
            #
            # - custom text keys
            # - multiple key columns merged
            # - prompt/completion format
            # - text column itself
            #
            # Throws an error, if it failed to process the record
            #
            # This is called for each row record in the dataset
            def map_tokenizer(x):
                # Custom text column support
                if kargs["custom_text_key"] is not None:
                    if kargs["custom_text_key"] in x:
                        return encodeTokens(x[kargs["custom_text_key"]])
                    
                if conversation_enabled:
                    conv_key = kargs['conversation_key'] if 'conversation_key' in kargs else None
                    conversation = x[conv_key] if conv_key is not None else x

                    # Array of output values we will return
                    input_ids = []
                    token_type_ids = []
                    attention_mask = []

                    if kargs['conversation_format'] == 'iopairs':
                        # lets loop through each io pair
                        for i in range(len(conversation)):
                            # lets loop through each key in the io pair
                            for key, value in conversation[i].items():
                                # lets get the prefix for this key
                                prefix = conversation_prefix_encoding_map[key] if sender in conversation_prefix_encoding_map[key] else None

                                # Add the prefix
                                if prefix is not None:
                                    input_ids += prefix['input_ids']
                                    token_type_ids += prefix['token_type_ids']
                                    attention_mask += prefix['attention_mask']

                                # Tokenize the column
                                column_encodings = encodeTokens(value)

                                # Add the column
                                input_ids += column_encodings['input_ids']
                                token_type_ids += column_encodings['token_type_ids']

                                if key not in kargs["conversation_input_key_mask"] or kargs["conversation_input_key_mask"][key]:
                                    # If the corresponding `conversation_input_key_mask` is not set, we will assume as valid training data
                                    attention_mask += ([1] * len(column_encodings['input_ids']))
                                else: # kargs["conversation_input_key_mask"][key] is False
                                    # This means it is false, lets not pay attention to it
                                    attention_mask += ([0] * len(column_encodings['input_ids']))

                                
                                suffix = conversation_suffix_encoding_map[key] if sender in conversation_suffix_encoding_map else None

                                if suffix is not None:
                                    input_ids += suffix['input_ids']
                                    token_type_ids += suffix['token_type_ids']
                                    attention_mask += suffix['attention_mask']
                    
                    elif kargs['conversation_format'] == 'sender':
                        for i in range(len(conversation)):
                            turn = conversation[i]
                            sender = turn[kargs['conversation_sender_key']]
                                
                            for key, value in kargs['conversation_input_key_map'].items():
                                if key in turn:
                                    # lets get the prefix for this key
                                    prefix = conversation_prefix_encoding_map[key][sender] if sender in conversation_prefix_encoding_map[key] else None

                                    # Add the prefix
                                    if prefix is not None:
                                        input_ids += prefix['input_ids']
                                        token_type_ids += prefix['token_type_ids']
                                        attention_mask += prefix['attention_mask']

                                    # Tokenize the column
                                    column_encodings = encodeTokens(turn[key])

                                    # Add the column
                                    input_ids += column_encodings['input_ids']
                                    token_type_ids += column_encodings['token_type_ids']

                                    if sender not in kargs["conversation_sender_mask"] or kargs["conversation_sender_mask"][sender]:
                                        # If the corresponding `conversation_input_key_mask` is not set, we will assume as valid training data
                                        attention_mask += ([1] * len(column_encodings['input_ids']))
                                    else: # kargs["conversation_input_key_mask"][key] is False
                                        # This means it is false, lets not pay attention to it
                                        attention_mask += ([0] * len(column_encodings['input_ids']))

                                    suffix = conversation_suffix_encoding_map[sender] if sender in conversation_suffix_encoding_map else None

                                    if suffix is not None:
                                        input_ids += suffix['input_ids']
                                        token_type_ids += suffix['token_type_ids']
                                        attention_mask += suffix['attention_mask']

                    if len(input_ids) > 0  and conversation_end_of_conversation_token is not None:
                        input_ids += conversation_end_of_conversation_token['input_ids']
                        token_type_ids += conversation_end_of_conversation_token['token_type_ids']
                        attention_mask += conversation_end_of_conversation_token['attention_mask']

                    return {
                        'input_ids': input_ids,
                        'token_type_ids': token_type_ids,
                        'attention_mask': apply_data_prefix_skip_mask(attention_mask)
                    }
                        
                # Multi column merging support
                if multi_column_enabled:
                    # Lets count the number of columns we have
                    # that have data in them
                    num_columns = 0
                    for i in range(len(multi_column_keys)):
                        if multi_column_keys[i] in x and x[multi_column_keys[i]] is not None and len(x[multi_column_keys[i]]) > 0:
                            num_columns += 1
                    # If we have more than 1 column, we will have to merge them
                    if num_columns > 1:
                        # Array of output values we will return
                        input_ids = []
                        token_type_ids = []
                        attention_mask = []

                        # First item flag
                        is_first_item = True

                        # Lets loop through each column
                        for i in range(len(multi_column_keys)):
                            # And process the column if it has data
                            if multi_column_keys[i] in x and x[multi_column_keys[i]] is not None and len(x[multi_column_keys[i]]) > 0:
                                # Add the separator if this is not the first item
                                if not is_first_item and multi_column_separator_encodings is not None:
                                    input_ids += multi_column_separator_encodings['input_ids']
                                    token_type_ids += multi_column_separator_encodings['token_type_ids']
                                    attention_mask += multi_column_separator_encodings['attention_mask']
                                
                                # Add the prefix
                                if len(multi_column_prefix_encodings) > i and multi_column_prefix_encodings[i] is not None:
                                    input_ids += multi_column_prefix_encodings[i]['input_ids']
                                    token_type_ids += multi_column_prefix_encodings[i]['token_type_ids']
                                    attention_mask += multi_column_prefix_encodings[i]['attention_mask']

                                # Tokenize the column
                                column_encodings = encodeTokens(x[multi_column_keys[i]])

                                # Add the column
                                input_ids += column_encodings['input_ids']
                                token_type_ids += column_encodings['token_type_ids']

                                # Configure the attention masks accordingly
                                if i > len(multi_column_train_mask):
                                    # If the corresponding `multi_column_train_mask` is not set, we will assume as valid training data
                                    attention_mask += ([1] * len(column_encodings['input_ids']))
                                elif multi_column_train_mask[i] is False:
                                    # If the `multi_column_train_mask` is set, but configured as false, we should not pay attention to it
                                    attention_mask += ([0] * len(column_encodings['input_ids']))
                                else: # multi_column_train_mask[i] is True
                                    # This means it is true, lets pay attention once again
                                    attention_mask += ([1] * len(column_encodings['input_ids']))
                                    
                                # Add the suffix
                                if len(multi_column_suffix_encodings) > i and multi_column_suffix_encodings[i] is not None:
                                    input_ids += multi_column_suffix_encodings[i]['input_ids']
                                    token_type_ids += multi_column_suffix_encodings[i]['token_type_ids']
                                    attention_mask += multi_column_suffix_encodings[i]['attention_mask']
                                
                                # Set the first item flag to false
                                is_first_item = False
                        
                        # Return the merged columns
                        return {
                            'input_ids': input_ids,
                            'token_type_ids': token_type_ids,
                            'attention_mask': apply_data_prefix_skip_mask(attention_mask)
                        }

                # Prompt completion support
                if 'prompt' in x and 'completion' in x:
                    # Array of output values we will return
                    input_ids = None
                    token_type_ids = None
                    attention_mask = None

                    # Tokenize both prompt and completion
                    # Note that the tokenizer will process and return the input_ids in batches
                    prompt_encodings = encodeTokens(x['prompt'])
                    completion_encodings = encodeTokens(x['completion'])

                    # Join the two input_ids lists
                    input_ids = prompt_encodings['input_ids'] + completion_encodings['input_ids']
                    # Join the two token_type_ids lists
                    token_type_ids = prompt_encodings['token_type_ids'] + completion_encodings['token_type_ids']
                    # Setup the attention mask, 0 for prompt, 1 for completion, if masking is enabled
                    if kargs["disable_prompt_completion_mask"]:
                        attention_mask = ([1] * len(prompt_encodings['input_ids']) + [1] * len(completion_encodings['input_ids']))
                    else:
                        attention_mask = ([0] * len(prompt_encodings['input_ids']) + [1] * len(completion_encodings['input_ids']))

                    # Prepare and return the output object
                    return {
                        'input_ids': input_ids,
                        'token_type_ids': token_type_ids,
                        'attention_mask': apply_data_prefix_skip_mask(attention_mask),
                    }
                
                # Fallback to standard text tokenization
                if 'text' in x:
                    ret = encodeTokens(x['text'])
                    return {
                        'input_ids': ret['input_ids'],
                        'token_type_ids': ret['token_type_ids'],
                        'attention_mask': apply_data_prefix_skip_mask(ret['attention_mask']),
                    }
                
                raise ValueError('Invalid dataset format, must contain either the configured "multi column" or prompt/completion or text')

            # Map the dataset to the tokenizer, removing the old text column
            src_dataset = src_dataset.map(map_tokenizer, batched=False, num_proc=num_cpus)
            
        # =====================================================

        # Remove all features, except input_ids, token_type_ids and attention_mask
        # as the metadata/etc columns may cause problems down the line (when passed to the trainer)
        dataset_features = src_dataset["train"].features
        dataset_features_to_remove = {k: v for k, v in dataset_features.items() if k not in ["input_ids", "token_type_ids", "attention_mask"]}
        src_dataset = src_dataset.remove_columns(list(dataset_features_to_remove.keys()))
    
        # Get the newline token
        endOfDoc_tokenSet = {
            'input_ids': [[0]],
            'token_type_ids': [[0]],
            'attention_mask': [[1]],
        }


        # See if rechunking is needed, this is useful mostly for "text" based datasets
        # where we would need to split them into "digestable" context length sizes 
        # used for foundation training
        # ---

        # The rechunking function
        def rechunk_text(x):
            # Full Raw values that we will need to "rechunk"
            full_input_ids = []
            full_token_type_ids = []
            full_attention_mask = []

            # Loop through the x input, and build the raw values
            for i in range(len(x["input_ids"])):
                # Get the respective values and push them to the 
                # raw value array, effectively merging the arrays together
                # with the newline token in between
                full_input_ids += x["input_ids"][i] + endOfDoc_tokenSet["input_ids"][0]
                full_token_type_ids += x["token_type_ids"][i] + endOfDoc_tokenSet["token_type_ids"][0]
                full_attention_mask += apply_data_prefix_skip_mask( x["attention_mask"][i] ) + endOfDoc_tokenSet["attention_mask"][0]
            
            # Total length, and sample count
            # note that thte "remainder" will be discarded
            total_len = len(full_input_ids)
            total_samples = total_len // kargs["text_rechunk_size"]

            # The output arrays
            out_input_ids = []
            out_token_type_ids = []
            out_attention_mask = []

            # Generate the output arrays
            for i in range(total_samples):
                # Calculate the start and end of the sample
                start = i * kargs["text_rechunk_size"]
                end = start + kargs["text_rechunk_size"]

                # Push the sample to the output arrays
                out_input_ids.append(full_input_ids[start:end])
                out_token_type_ids.append(full_token_type_ids[start:end])
                out_attention_mask.append(apply_data_prefix_skip_mask( full_attention_mask[start:end] ))
            
            # Prepare and return the output object
            ret = {
                'input_ids': out_input_ids,
                'token_type_ids': out_token_type_ids,
                'attention_mask': out_attention_mask,
            }
            return ret
        
        # Get the kargs["processing_max_batch_size"], if not set, we will use the full dataset
        processing_max_batch_size = kargs["processing_max_batch_size"]
        if processing_max_batch_size <= 0:
            processing_max_batch_size = len(src_dataset["train"])

        # Remove empty datasets (it causes an error otherwise)
        # and perform min/max length filtering (if configured)
        def dataset_filter(x):
            row_length = len(x["input_ids"])
            if row_length <= 0:
                return False
            if kargs["min_token_size"] > 0 and row_length < kargs["min_token_size"]:
                return False
            if kargs["max_token_size"] > 0 and row_length > kargs["max_token_size"]:
                return False
            return True
        src_dataset = src_dataset.filter(dataset_filter, num_proc=num_cpus)

        # Rechunking happened
        rechunking_happened = False
        
        # Perform rechunking if needed for "text" based datasets
        text_rechunk_size = int(kargs["text_rechunk_size"])
        if text_rechunk_size > 0:
            if kargs["source"] == "text" and (kargs["text_rechunk_auto"] or kargs["text_rechunk_force"]):
                rechunking_happened = True
                src_dataset = src_dataset.map(rechunk_text, batched=True, 
                                            batch_size=min(text_rechunk_size*8, processing_max_batch_size),
                                            num_proc=num_cpus)
            
            # Perform rechunking after filtering, if source is not a "text" based 
            # dataset and text_rechunk_force is enabled
            if kargs["source"] != "text" and kargs["text_rechunk_force"]:
                rechunking_happened = True
                src_dataset = src_dataset.map(rechunk_text, batched=True, 
                                            batch_size=min(text_rechunk_size*8, processing_max_batch_size),
                                            num_proc=num_cpus)

        # Check if the dataset does not have a test split
        # and if so, perform the split
        if 'test' not in src_dataset.keys():
            test_split = kargs["test_split"]
            # The minimum test size is 1, if not we will get errors in the trainer?
            if test_split <= 0 or test_split <= 0.0:
                test_split = 1
            src_dataset = src_dataset['train'].train_test_split(
                test_size=test_split,shuffle=kargs["test_split_shuffle"],
                seed=42 #Fixed seed, to prevent train/test reshuffling between test runs
            )
        
        # Compute the sample length, as requried for the sort by length feature, and packing
        def add_length(example):
            example["sample_length"] = len(example['input_ids'])
            return example
        src_dataset['train'] = src_dataset['train'].map(add_length, batched=False, num_proc=num_cpus)

        # Does the actual sorting process (after test split!)
        if kargs["sort_by_length"]:
            if kargs["packing_enable"] and not kargs["packing_in_sequence"]:
                # Show warning if sort_by_length is enabled, with packing
                print("Warning: sort_by_length=true, packing_enable=true, with packing_in_sequence=False - sort_by_length to be ignored")
            else:
                sort_asc = kargs["sort_asc"]
                src_dataset['train'] = src_dataset['train'].sort("sample_length", reverse=not sort_asc)
        
        # Implement dataset packing, which merges the dataset row records, into "fixed sizes"
        # this is done by merging multiple dataset samples, with a 0 token in between
        # to form a single dataset sample of the desired size
        #
        # The longest dattaset sample (below the pack size) will be appended with the shortest
        # dataset samples, until the desired pack size is reached. With the process 
        # repeated for all samples
        #
        # This however will mess up the "real_ctx_len" value, as it will be the length of the
        # of the merged dataset samples, instead of the original dataset sample.
        # ---

        if kargs["packing_enable"] and rechunking_happened:
            # Show warning if packing_enable is enabled, with rechunking
            print("Warning: packing_enable=true, with text rechunking (either auto, or forced) - packing_enable will be treated as false")

        if kargs["packing_enable"] and not rechunking_happened:

            # def add_length(example):
            #     example["sample_length"] = len(example['input_ids'])
            #     return example
            # src_dataset['train'] = src_dataset['train'].map(add_length, batched=False, num_proc=num_cpus)

            # The pack size
            packing_batchsize = int(kargs["packing_batchsize"])
            packing_chunksize = int(kargs["packing_chunksize"])
            packing_min_ctx_len = int(kargs["packing_min_ctx_len"])

            if packing_min_ctx_len <= 0:
                packing_min_ctx_len = packing_chunksize

            # The pack function
            def pack_dataset_in_sequence(x):
                
                # The return resulting arrays
                id_arr = []
                type_arr = []
                mask_arr = []
                sample_len_arr = []

                # batch set chunk counting
                batchset_chunksize = [0]
                
                # The total length of the dataset
                total_len = len(x["input_ids"])

                # Preload size (we can use either packing_batchsize, or just 1)
                preload_size = 1 # packing_batchsize

                # Lets prepare the basic first chunk
                for i in range(preload_size):
                    # Port the values to the return arrays
                    id_arr.append(x["input_ids"][i])
                    type_arr.append(x["token_type_ids"][i])
                    mask_arr.append(x["attention_mask"][i])
                    sample_len_arr.append([x["sample_length"][i]])

                    # Keep the chunk count in sync
                    batchset_chunksize[0] = max( 
                        math.ceil( max(x["sample_length"][i],packing_min_ctx_len) / packing_chunksize ) * packing_chunksize, 
                        batchset_chunksize[0] 
                    )

                # Given the datasample index, try to scan and merge into existing samples (if possible)
                def merge_into_existing_samples(i):
                    # Get the sample length
                    sample_len = x["sample_length"][i]

                    # Iterate and see if we can merge the sample
                    for j in range(len(batchset_chunksize)):

                        # Get the current set chunk size
                        current_set_chunk_size = batchset_chunksize[j]
                        
                        # Iterate existing samples for the chunk
                        for k in range( j * packing_chunksize, min((j+1) * packing_chunksize, len(id_arr))):
                            # Get the existing record length
                            existing_record_len = len(id_arr[k])

                            # Check if the sample can be merged
                            if existing_record_len + 1 + sample_len < current_set_chunk_size:
                                # Merge the sample
                                id_arr[k] += endOfDoc_tokenSet["input_ids"][0] + x["input_ids"][i]
                                type_arr[k] += endOfDoc_tokenSet["token_type_ids"][0] + x["token_type_ids"][i]
                                mask_arr[k] += endOfDoc_tokenSet["attention_mask"][0] + x["attention_mask"][i]
                                sample_len_arr[k].append(sample_len)
                                
                                # Return that a merge has been done
                                return True
                            
                    # Return that no merge has been done
                    return False

                # Lets iterate the rest of the dataset, and start packing
                for i in range(preload_size, total_len):
                    # Merge if possible
                    if merge_into_existing_samples(i):
                        continue

                    # Ok merge failed, lets append and update the chunk size, of the affected batchset
                    id_arr.append(x["input_ids"][i])
                    type_arr.append(x["token_type_ids"][i])
                    mask_arr.append(x["attention_mask"][i])
                    sample_len_arr.append([x["sample_length"][i]])

                    # Update the chunk size
                    batchset_id = math.floor( len(id_arr) / packing_chunksize )
                    updated_chunksize = max( math.ceil( x["sample_length"][i] / packing_chunksize ) * packing_chunksize, packing_min_ctx_len )

                    if batchset_id >= len(batchset_chunksize):
                        batchset_chunksize.append(updated_chunksize)
                    else:
                        batchset_chunksize[batchset_id] = max(updated_chunksize, batchset_chunksize[batchset_id])

                # Prepare and return the output object
                ret = {
                    'input_ids': id_arr,
                    'token_type_ids': type_arr,
                    'attention_mask': mask_arr,
                    'sample_length': sample_len_arr
                }
                return ret

            # Shuffle the dataset if needed
            if not kargs["packing_in_sequence"]:
                src_dataset['train'] = src_dataset['train'].shuffle(seed=101)

            # Perform the dataset packing
            src_dataset['train'] = src_dataset['train'].map(pack_dataset_in_sequence, batched=True, 
                                        batch_size=min(packing_min_ctx_len*2*3*5, processing_max_batch_size),
                                        num_proc=num_cpus)
        else:
            # Remove the sample_length column, as it is no longer needed
            src_dataset['train'] = src_dataset['train'].remove_columns(["sample_length"])
        
        # If an int value is used, it is interprated as document count
        # If a floating value (<1.0) is used, it is interprated as a percentage of the dataset
        if kargs["dataset_offset"] > 0 or kargs["dataset_length"] > 0:
            # src dataset length
            train_length = len(src_dataset["train"])

            # Compute the offset position
            offset_val = kargs["dataset_offset"]

            # If offset is a float, we will use it as a percentage
            if offset_val < 0:
                offset_val = 0
            if offset_val > 0 and offset_val < 1.0:
                offset_val = int(train_length * offset_val) # Rounded down value

            # Compute the length position
            length_val = kargs["dataset_length"]
            if length_val < 0:
                length_val = train_length - offset_val
            if length_val > 0 and length_val < 1.0:
                length_val = int(train_length * length_val)
            if length_val > (train_length - offset_val):
                length_val = (train_length - offset_val)

            # Get the subset of the dataset
            src_dataset["train"] = src_dataset["train"].select(range(offset_val, offset_val + length_val))

        # Save the dataset to disk
        src_dataset.save_to_disk(kargs["data_path"])

# Dataloader collator for merging multiple dataset records together
# we use token 0 for padding, with a learning mask value of 0
def dataloader_collator_fn(records):
    # Get the maximum number of records 
    # (aka the batch size)
    records_len = len(records)
    
    # Compute the total length of the records
    input_ids_len = 0
    token_type_ids_len = 0
    attention_mask_len = 0

    # Loop through the records and compute the max length
    for i in range(records_len):
        input_ids_len = max(input_ids_len, len(records[i]["input_ids"]))
        token_type_ids_len = max(token_type_ids_len, len(records[i]["token_type_ids"]))
        attention_mask_len = max(attention_mask_len, len(records[i]["attention_mask"]))

    # First row of the records
    first_row = records[0]

    # Create the output arrays, with the default 0 values (no learning mask)
    out_input_ids = torch.zeros((records_len, input_ids_len), dtype=first_row["input_ids"].dtype)
    out_token_type_ids = torch.zeros((records_len, token_type_ids_len), dtype=first_row["token_type_ids"].dtype)
    out_attention_mask = torch.zeros((records_len, attention_mask_len), dtype=first_row["attention_mask"].dtype)
    out_data_ctx_len = torch.zeros((records_len), dtype=torch.int32)

    # Loop through the records and copy the values to the output arrays
    for i in range(records_len):
        out_input_ids[i][:len(records[i]["input_ids"])] = records[i]["input_ids"]
        out_token_type_ids[i][:len(records[i]["token_type_ids"])] = records[i]["token_type_ids"]
        out_attention_mask[i][:len(records[i]["attention_mask"])] = records[i]["attention_mask"]
        out_data_ctx_len[i] = len(records[i]["input_ids"])
    
    # Build & return the output object
    out = {
        'input_ids': out_input_ids,
        'token_type_ids': out_token_type_ids,
        'attention_mask': out_attention_mask,
        'data_ctx_len': out_data_ctx_len
    }
    return out

class RWKVDataModule(LightningDataModule):
    def __init__(
        self, 
        # load_from_disk(dataset_path) param
        data_path: str,
        # load_dataset(path) param
        source: str = None,
        # load_dataset(data_dir) param
        source_data_dir: str = None,
        # Additional dataset params
        source_dataset_params: dict = None,
        # Test split of source data, if it was not already done
        test_split: float = 0.01,
        test_split_shuffle: bool = False,
        # Text rechunking size
        text_rechunk_size: int = 4096,
        text_rechunk_auto: bool = True,
        text_rechunk_force: bool = False,
        # ---
        # Tokenizer settings
        # ---
        tokenizer: str = "neox",
        autoTokenizer = None,

        # Add <|endoftext|> string token to the world tokenizer, at index 0
        # this was missing from the original world trie_tokenizer
        world_add_endoftext_token: bool = True,

        # ---
        # HF dataset conversion helpers
        # ---
        # Min / Max token size filtering
        #
        # default min token size of 1 is chosen, to filter out empty records
        # which causes errors in the trainer
        min_token_size: int = 1,
        max_token_size: int = -1,
        
        # Sort by length
        sort_by_length: bool = False,
        sort_asc: bool = True,

        # Dataloader shuffling, disabled if "sort_by_length" is enabled
        training_dataloader_shuffle_auto: bool = True,

        # Dataset offset and limit controls
        dataset_offset: float = -1,
        dataset_length: float = -1,
        
        # Custom 'text' column to support, mostly used for dataset where the 
        # desired train data is in another column (eg. 'code')
        custom_text_key: str = None,
        # Multi column merging support, used for instruct/input/output datasets
        # or similar varients where the input and output are in different columns
        # and need to be merged
        multi_column_keys: list = None,
        multi_column_prefix: list = None,
        multi_column_suffix: list = None,
        multi_column_train_mask: list = None,
        multi_column_separator: str = None,
        # Conversation format support
        conversation_format: str = None,
        conversation_key: str = None,

        # conversation_format == 'iopairs'
        conversation_input_key_prefix_map: dict = None,
        conversation_input_key_mask: dict = None,

        # conversation_format == 'sender'
        conversation_sender_key: str = None,
        conversation_sender_value_map: dict = None,
        conversation_input_key_map: dict = None,
        conversation_sender_suffix: dict = None,
        conversation_sender_mask: dict = None,
        conversation_end_of_conversation: str = None,

        # prompt/completion format masking support
        disable_prompt_completion_mask: bool = False,

        # ----------------------------
        # Selective loss training
        # ----------------------------

        # Prefix token masking
        #
        # The rationale behind this, is that the first X tokens should not be "backpropped"
        # for any new training record. As its unfair to expect the model (or a human) make
        # any resonable guesses at that stage. As such this is used to "mask" the first X tokens
        # from the loss calculation, and thus not backpropped.
        data_prefix_skip_mask: int = 0,

        # ----------------------------
        # dataset packing support
        # ----------------------------

        # Boolean flag to enable / disable dataset packing
        packing_enable: bool = False,

        # Used to ensure all training samples wihin this batch size is the same length
        # Ideally this should align exactly with your real "batch size"
        #
        # Uses, `8 * (3 * 4 * 5 * 6 * 7) = 20160` for default, as it should align across
        # a large number of batch size combinations. This helps reduce the amount of
        # misaligned batches, and thus reduce the amount of wasted training time.
        packing_batchsize: int = 20160,

        # Chunking size to align within each batch, this ideally should be equal to
        # the training context length used.
        packing_chunksize: int = 4096,

        # Minimum size to pack up to, this should be a multiple of packing_chunksize
        # defautls to -1, which equals to packing_chunksize
        packing_min_ctx_len: int = -1,

        # Pack the data sequentially if possible, in accordance to the dataset sequence
        # this can be used together with sort_by_length, otherwise a shuffle will be done
        packing_in_sequence: bool = False,

        # ----------------------------
        # System tweaks
        # ----------------------------

        # Batch size scanning range, used for deciding the max number of documents
        # to process simultaneously at a time. This is used to prevent OOM errors
        # while rearranging the dataset, etc. Used for both packing / sorting operations
        processing_max_batch_size: int = 100000,

        # Skip database setup checks if datapath exists, ignored if using preload_datapath.py
        skip_datapath_setup: bool = False
    ):
        # Capture the init parameters
        self._init_locals = locals()
        del self._init_locals["self"]
        del self._init_locals["__class__"]
        
        super().__init__()
        self.data_path = data_path
        self._loaded_dataset = None
        self.sort_by_length = sort_by_length
        self.training_dataloader_shuffle_auto = training_dataloader_shuffle_auto

        # Log to wandb
        if wandb.run is not None:
            wandb.config.update({ "data":dict(self._init_locals) })
    
    # Called once for initial setup
    def prepare_data(self):
        prepare_data_static(**self._init_locals)
    
    # Setup process that is universal
    def _internal_setup(self):
        if self._loaded_dataset is None:
            self._loaded_dataset = load_from_disk(self.data_path).with_format('torch')

    # Called once for every process in DDP
    def setup(self, stage):
        self._internal_setup()

    # Return the train dataloader
    def train_dataloader(self):
        self._internal_setup()
        dataset = self._loaded_dataset['train'];
        sampler = DistributedSampler(
            dataset, 
            shuffle=self.training_dataloader_shuffle_auto and not self.sort_by_length,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        microbatch_size = 1
        if hasattr(self, "trainer") and hasattr(self.trainer, "microbatch_size"):
            microbatch_size = self.trainer.microbatch_size

        return DataLoader(
            dataset, 
            sampler=sampler,
            shuffle=False,
            # 4 prefetch workers per GPU
            num_workers=4, 
            # Prefetching 8 batches
            prefetch_factor=8,
            # Of batch size 1 datasets
            batch_size=microbatch_size, 
            # The collation function
            collate_fn=dataloader_collator_fn,
            # Pinned in GPU memory
            pin_memory=True
        )
    
    # Return the validation dataloader
    def val_dataloader(self):
        self._internal_setup()
        dataset = self._loaded_dataset['test'];
        sampler = DistributedSampler(
            dataset, 
            shuffle=False, 
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )
        return DataLoader(
            dataset, 
            sampler=sampler,
            shuffle=False,
            # 4 prefetch workers per GPU
            num_workers=4, 
            # Prefetching 8 batches
            prefetch_factor=8,
            # Of batch size 1 datasets
            batch_size=1, 
            # Pinned in GPU memory
            pin_memory=True
        )