from lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, BatchSampler, SequentialSampler

import math, random, numpy
import wandb
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset, Features, Value, Sequence
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from multiprocessing import cpu_count
import gc, yaml

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
def prepare_data_static(
        # load_from_disk(dataset_path) param
        data_path: str,
        # Data path storage options, this is used to support cloud storage
        # via the huggingface dataset API. See:
        # https://huggingface.co/docs/datasets/v2.16.1/en/filesystems#amazon-s3
        # Note: As of Jan 2023, these options seems very buggy, YMMV
        data_path_storage_options:dict = None,

        # load_dataset(path) param
        source: str = None,
        # load_dataset(data_dir) param
        source_data_dir: str = None,
        # Additional dataset params
        source_dataset_params: dict = None,
        # Source dataset split to use
        source_dataset_split: str = "train",
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
        tokenizer: str = "world",
        autoTokenizer = None,

        # Add <|endoftext|> string token to the world tokenizer, at index 0
        # this was missing from the original world trie_tokenizer
        world_add_endoftext_token: bool = True,

        # ---
        # HF dataset conversion helpers
        # ---
        # Min / Max token size filtering
        #
        # default min token size of 2 is chosen, to filter out empty records
        # which causes errors in the trainer
        min_token_size: int = 2,
        max_token_size: int = -1,
        
        # Sort by length
        sort_by_length: bool = False,
        sort_asc: bool = True,

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
        # Specal use caes flags
        # ----------------------------

        # Reverse the training dataset order before saving, this is useful for,
        # optimizing dataset packing process, when using packing_in_sequence
        # and sort_by_length desc order together
        reverse_train_dataset_before_save: bool = False,

        # ----------------------------
        # System tweaks
        # ----------------------------

        # Skip database setup checks if datapath exists, ignored if using preload_datapath.py
        skip_datapath_setup: bool = False,
        
        # Batch size scanning range, used for deciding the max number of documents
        # to process simultaneously at a time. This is used to prevent OOM errors
        # while rearranging the dataset, etc. Used for both packing / sorting operations
        processing_max_batch_size: int = 100 * 1000,

        # Dataloader shuffling, disabled if "sort_by_length" is enabled
        dataloader_shuffle_training: bool = False,

        # With a total of 4 batches prefetched into memory
        dataloader_prefetch_factor:int = 4,

        # Pin the preloaded documents into GPU memory in advance
        # very small overhead, slight speed bump, disable if your deperate for vram
        dataloader_pin_memory: bool = True,

        # ----------------------------
        # Data oacj soecufuc
        # ----------------------------

        # Dataset name, and index
        # This is only useful for multi dataset packing 
        dataset_name: str = None,
        dataset_index: int = 0,
        is_random_datapack: bool = False,

        # Additional kargs
        **kargs
    ):

    # Capture the init parameters
    kargs = locals()
    
    # Check if skip_datapath_setup is enabled
    # useful for extra large datasets
    if kargs["skip_datapath_setup"] == True:
        return None

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
            if data_prefix_skip_mask_val > 0 and mask_len > 0:
                for i in range(min(data_prefix_skip_mask_val, mask_len)):
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

            # Log the whole load_dataset_params
            # print("load_dataset_params: " + str(load_dataset_params))

            # The split to use
            source_dataset_split = kargs["source_dataset_split"]

            # Load the dataset
            src_dataset = load_dataset(**load_dataset_params)

            # If for some reason the dataset is a "test" only split, and missing a "train" split, we remap it as a "train" split
            if source_dataset_split not in src_dataset.keys():
                raise ValueError('Dataset missing split: ' + source_dataset_split)

            if source_dataset_split != "train":
                src_dataset["train"] = src_dataset[source_dataset_split]
                del src_dataset[source_dataset_split]

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
                                # Get the sender key
                                sender = key
                                # lets get the prefix for this key
                                prefix = conversation_prefix_encoding_map[key] if sender in conversation_prefix_encoding_map else None

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
            if row_length <= 1:
                return False
            if kargs["min_token_size"] > 0 and row_length < kargs["min_token_size"]:
                return False
            if kargs["max_token_size"] > 0 and row_length > kargs["max_token_size"]:
                return False
            if sum(x["attention_mask"]) <= 0:
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

        # =====================================================
                                        
        # Remove the sample_length column, as it is no longer needed / causes problems down the line 
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

        # Dataset flipping (if needed)
        if kargs["reverse_train_dataset_before_save"]:
            train_dataset = src_dataset["train"]
            def reverse_dataset(x, idx):
                return train_dataset[train_dataset.num_rows - idx - 1]
            src_dataset["train"] = src_dataset["train"].map(reverse_dataset, with_indices=True, num_proc=num_cpus)

        # # Convert to iterable datasets (does not support saving to disk???)
        # src_dataset["train"] = src_dataset["train"].to_iterable_dataset()
        # src_dataset["test"] = src_dataset["test"].to_iterable_dataset()
        
        # # @TODO: Fix dataset_index / name labels
        # # Dataset labeling, for custom wandb graphing
        # if kargs["dataset_name"] is not None or kargs["dataset_index"] >= 0:
        #     # Lets label every sample with the dataset name or index
        #     def label_dataset(x):
        #         if kargs["dataset_name"] is not None:
        #             x["dataset_name"] = kargs["dataset_name"]
        #         if kargs["dataset_index"] >= 0:
        #             x["dataset_index"] = kargs["dataset_index"]
        #         return x
            
        #     # Apply the label function
        #     src_dataset["train"] = src_dataset["train"].map(label_dataset, num_proc=num_cpus)
        #     src_dataset["test"] = src_dataset["test"].map(label_dataset, num_proc=num_cpus)

        # Save the dataset to disk (if enabled)
        # For the skip datapath saving string
        # We intentionally used several filesystem illegal characters, to ensure it
        # is not accidentally used by the user for a real file
        if kargs["data_path"] != ".//<#|=@%!$skip_datapath$!%@=|#>//.":
            if kargs["data_path_storage_options"]:
                
                # import s3fs
                # fs = s3fs.S3FileSystem(
                #     key=kargs["data_path_storage_options"]["key"],
                #     secret=kargs["data_path_storage_options"]["secret"],
                #     endpoint_url=kargs["data_path_storage_options"]["endpoint_url"],
                #     client_kwargs={
                #         'region_name': 'sfo3'
                #     },
                #     # asynchronous=True,
                #     config_kwargs={
                #         'signature_version': 's3v4',
                #         's3': {
                #             'addressing_style': 'virtual'
                #         }
                #     }
                # )
                # print("fs.ls", fs.ls(""))

                src_dataset.save_to_disk(
                    kargs["data_path"], 
                    storage_options=kargs["data_path_storage_options"]
                )
            else:
                src_dataset.save_to_disk(
                    kargs["data_path"]
                )

        # Return the dataset object itself
        return src_dataset
    else:
        # there is nothing, return none
        return None

# Dataloader collator for merging multiple dataset records together
# we use token 0 for padding, with a learning mask value of 0
def dataloader_collator_fn(records):
    # Get the maximum number of records 
    # (aka the batch size)
    records_len = len(records)
    
    # Compute the total length of the records
    input_ids_len = 0
    # token_type_ids_len = 0
    # attention_mask_len = 0

    # Loop through the records and compute the max length
    for i in range(records_len):
        input_ids_len = max(input_ids_len, len(records[i]["input_ids"]))
        # token_type_ids_len = max(token_type_ids_len, len(records[i]["token_type_ids"]))
        # attention_mask_len = max(attention_mask_len, len(records[i]["attention_mask"]))

    # First row of the records
    first_row = records[0]

    # Create the output arrays, with the default 0 values (no learning mask)
    out_input_ids = torch.zeros((records_len, input_ids_len), dtype=first_row["input_ids"].dtype)
    out_token_type_ids = torch.zeros((records_len, input_ids_len), dtype=first_row["token_type_ids"].dtype)
    out_attention_mask = torch.zeros((records_len, input_ids_len), dtype=first_row["attention_mask"].dtype)
    out_data_ctx_len = torch.zeros((records_len), dtype=torch.int32)

    out_index = 0
    out_name = None
    # # Add dataset_index if its set
    # if "dataset_index" in records:
    #     out_index = records[0]["dataset_index"]
    # if "dataset_name" in records:
    #     out_name = records[0]["dataset_name"]
    

    # Loop through the records and copy the values to the output arrays
    for i in range(records_len):
        out_input_ids[i][:len(records[i]["input_ids"])] = records[i]["input_ids"]
        out_token_type_ids[i][:len(records[i]["token_type_ids"])] = records[i]["token_type_ids"]
        out_attention_mask[i][:len(records[i]["attention_mask"])] = records[i]["attention_mask"]
        out_data_ctx_len[i] = len(records[i]["input_ids"])

        # if i > 0 and out_index > 0 and out_index != records[i]["dataset_index"]:
        #     out_index = -1
        #     out_name = "mixed"
    
    # Build & return the output object
    out = {
        'input_ids': out_input_ids,
        'token_type_ids': out_token_type_ids,
        'attention_mask': out_attention_mask,
        'data_ctx_len': out_data_ctx_len,
        # 'dataset_index': out_index,
        # 'dataset_name': out_name
    }

    return out

# Build the datapack given the given settings
def prepare_datapack_static(
        # Preload and return without merging
        only_preload=False,
        # Return without counting
        return_without_counting=False,
        # Skip datapath setup, if set
        skip_datapath_setup=False,

        # Additional kargs
        **kargs
    ):

    # Get the config groups
    datapack_config = kargs["datapack"]
    default_config = kargs["default"]
    dataset_config_arr = kargs["dataset"]

    # packing_batchsize
    packing_batchsize = 64
    if "packing_batchsize" in datapack_config:
        packing_batchsize = datapack_config["packing_batchsize"]
    
    # Join the various default settings
    defaultVals = { "packing_batchsize": packing_batchsize, "dataset_weight": 1.0 }
    defaultVals = { **defaultVals, **default_config }

    # Prepare the array of all the datasets to be merged
    datasets_arr = []
    datasets_train_count = []
    datasets_test_count = []
    datasets_train_used_count = []
    datasets_config_merged_arr = []

    # Reset a dataset array
    def reset_dataset_arr_item(i):
        # Get the dataset config
        dataset_in = dataset_config_arr[i]

        # Merge the default config
        one_dataset_config = {
            **{"skip_datapath_setup":skip_datapath_setup}, 
            **defaultVals, 
            **dataset_in, 
            **{ "dataset_index": i } 
        }

        # Insert the "dataset_name" if "name" is set
        if "name" in dataset_in and dataset_in["name"] is not None:
            one_dataset_config = { **one_dataset_config, **{ "dataset_name": dataset_in["name"] } }
        elif "dataset_name" in dataset_in and dataset_in["dataset_name"] is not None:
            one_dataset_config = { **one_dataset_config, **{ "dataset_name": dataset_in["dataset_name"] } }
        else:
            one_dataset_config = { **one_dataset_config, **{ "dataset_name": "dataset_"+str(i) } }

        # Prepapre the datapath, use the dataset overried if set
        if "data_path" in dataset_in and dataset_in["data_path"] is not None:
            one_dataset_config["data_path"] = dataset_in["data_path"]
        elif "data_path" in default_config and  default_config["data_path"] is not None:
            one_dataset_config["data_path"] = default_config["data_path"]+"/"+str(i)+"/"
        else:
            one_dataset_config["data_path"] = ".//<#|=@%!$skip_datapath$!%@=|#>//."

        # Initialize the array indexing if needed
        if i >= len(datasets_arr):
            datasets_arr.append( None )
            datasets_config_merged_arr.append( None )
            datasets_train_count.append(0)
            datasets_test_count.append(0)
            datasets_train_used_count.append(0)
        
        if "source" in one_dataset_config and one_dataset_config["source"] is not None:
            datasets_arr[i] = prepare_data_static(**one_dataset_config)
        elif one_dataset_config["data_path"] != ".//<#|=@%!$skip_datapath$!%@=|#>//.":
            if "data_path_storage_options" in one_dataset_config and one_dataset_config["data_path_storage_options"] is not None:
                datasets_arr[i] = load_from_disk(one_dataset_config["data_path"], storage_options=one_dataset_config["data_path_storage_options"])
            else:
                datasets_arr[i] = load_from_disk(one_dataset_config["data_path"])
        else:
            raise ValueError("Invalid dataset config, missing both source / data_path")

        datasets_config_merged_arr[i] = one_dataset_config

        # Get the dataset lengths
        datasets_train_count[i] = len(datasets_arr[i]["train"])
        datasets_test_count[i] = len(datasets_arr[i]["test"])
        datasets_train_used_count[i] = 0

    # Loop through the dataset config
    # And prepare each dataset seperately
    for i in range(len(dataset_config_arr)):
        if "name" in dataset_config_arr[i]:
            print(">> Preparing dataset - index: ", i, " - name: ", dataset_config_arr[i]["name"])
        else:
            print(">> Preparing dataset - index: ", i)
        reset_dataset_arr_item(i)

        # Perform GC between sets
        gc.collect()

    # If its preload, skip 
    if only_preload:
        print(">> Preload enabled, skipping dataset merging")
        return None

    # The final dataset to build together
    final_dataset = None
    final_trainset = None
    final_testset = None

    # Packing Mode
    mixing_mode = datapack_config["mixing_mode"]
    print(">> Dataset Mixing mode: ", mixing_mode)

    if mixing_mode == "concat" or mixing_mode == "shuffle":
        # ---------------------------------
        # Simple concatenation mode
        # ---------------------------------
        train_dataset = []
        test_dataset = []

        # Loop through the datasets and build the final dataset
        for i in range(len(datasets_arr)):
            train_dataset.append(datasets_arr[i]["train"])
            test_dataset.append(datasets_arr[i]["test"])
        
        # Build the final training dataset
        final_trainset = concatenate_datasets(train_dataset)
        if mixing_mode == "shuffle":
            final_trainset = final_trainset.shuffle(seed=101)

        # And the final test set
        final_testset = concatenate_datasets(test_dataset)
    else:
        # ---------------------------------
        # Complicated batch mixing mode
        # ---------------------------------

        raise ValueError("BATCH mixing mode not yet supported")

        # # Compute the total dataset lengths
        # total_train_count = sum(datasets_train_count)

        # # Get the datapack batchsize
        # datapack_batchsize = datapack_config["batchsize"]
        # mixed_batch_percentage = datapack_config["mixed_batch_percentage"]
        # full_batch_percentage = 1.0 - mixed_batch_percentage

        # # Override batch percentage if needed
        # if datapack_config["mixing_mode"] == "shuffle":
        #     mixed_batch_percentage = 1.0
        #     full_batch_percentage = 0.0

        # # Compute the number of batches for each dataset
        # datasets_train_batches = []
        # datasets_test_batches = []

        # # Compute the number of batches for each dataset
        # for i in range(len(dataset_config_arr)):
        #     dataset_weight = 1.0
        #     if "dataset_weight" in datasets_config_merged_arr[i]:
        #         dataset_weight = datasets_config_merged_arr[i]["dataset_weight"]

        #     # Throw if dataset_weight != 1.0 (not yet supported)
        #     if dataset_weight != 1.0:
        #         raise ValueError("dataset_weight != 1.0 is not yet supported")

        #     # Initialize empty values, if not set
        #     if i >= len(datasets_train_batches):
        #         datasets_train_batches.append(0)
        #         datasets_test_batches.append(0)
            
        #     # Compute the number of batches for each dataset
        #     datasets_train_batches[i] = math.floor( datasets_train_count[i] * dataset_weight * full_batch_percentage / datapack_batchsize )
        #     datasets_test_batches[i] = math.floor( datasets_test_count[i] / datapack_batchsize )

        # # Compute the total number of batches for training
        # total_train_batches = math.floor( total_train_count / datapack_batchsize )
        # total_full_batches = sum( datasets_train_batches )

        # print(">> Total approx train batches ( full | random ) :", total_train_batches, " ( ", total_full_batches, " | ", total_train_batches - total_full_batches, " )")

        # # ---
        # # The full dataset will contain the following columns
        # # input_ids, token_type_ids, attention_mask, sample_length, dataset_index, dataset_name
        # # ---

        # # Build the full train / test split dataset slices
        # train_fullset_arr = []
        # train_randomset_arr = []
        # test_fullset_arr = []

        # # For each dataset, build the full and random sets
        # for i in range(len(datasets_arr)):
        #     if datasets_train_batches[i] > 0:
        #         train_fullset_arr.append( datasets_arr[i]["train"].select(numpy.arange(0, datasets_train_batches[i]*datapack_batchsize)) )
        #         train_randomset_arr.append( datasets_arr[i]["train"].select(numpy.arange(datasets_train_batches[i]*datapack_batchsize, datasets_train_count[i])) )
        #     else:
        #         train_randomset_arr.append( datasets_arr[i]["train"] )
        #     test_fullset_arr.append( datasets_arr[i]["test"] )

        # # Concat the full dataset
        # train_fullset = concatenate_datasets(train_fullset_arr)
        # train_randomset = concatenate_datasets(train_randomset_arr)
        # test_fullset = concatenate_datasets(test_fullset_arr)

        # # Shuffle the train random sets, and merge it
        # train_randomset_len = len(train_randomset)
        # train_randomset = train_randomset.shuffle(seed=101)
        # train_randomset_chunks = math.floor( train_randomset_len / datapack_batchsize )

        # if train_randomset_chunks > 0:
        #     train_fullset = concatenate_datasets([train_fullset,train_randomset.select(numpy.arange(0, train_randomset_chunks*datapack_batchsize))])
        #     train_last_randomset_chunk = train_randomset.select(numpy.arange(train_randomset_chunks*datapack_batchsize, train_randomset_len))
        # else:
        #     train_last_randomset_chunk = train_randomset
        
        # # Get the total fullset chunks
        # train_fullset_chunks = math.floor( len(train_fullset) / datapack_batchsize )

        # # Lets prepare an array of the various dataset indexes
        # dataset_shuffle_index_arr = list( numpy.arange(0, train_fullset_chunks) )
        # random.Random(101).shuffle(dataset_shuffle_index_arr)
        
        # # Label shuffle index
        # def label_shuffle_index(x, idx):
        #     x["shuffle_index"] = idx
        #     return x

        # # Add a shuffled index to the fullset 
        # train_fullset = train_fullset.map(
        #     label_shuffle_index,
        #     with_indices=True,
        #     batched=True,
        #     batch_size=datapack_batchsize,
        #     num_proc=num_cpus,
        # )
        
        # # Sort the fullset by the shuffle index
        # train_fullset = train_fullset.sort("shuffle_index")

        # # Remove the shuffle index
        # train_fullset = train_fullset.remove_columns(["shuffle_index"])

        # # Add the last randomset chunk to the fullset
        # final_trainset = concatenate_datasets([train_fullset, train_last_randomset_chunk])
        # final_testset = test_fullset

    # ---------------------------------
    # Final dataset merger
    # ---------------------------------
    
    # Build the final_dataset
    if final_dataset is None:
        
        # Int type for the dataset (based on index:0 dataset)
        # Note: these are hugging face "Value(dtype=x)", and not the dtype itself
        dataset_input_id_type = datasets_arr[0]["train"].features["input_ids"].feature
        dataset_token_type_id_type = datasets_arr[0]["train"].features["token_type_ids"].feature
        dataset_attention_mask_type = datasets_arr[0]["train"].features["attention_mask"].feature

        # Setup the dataset features
        final_dataset_features = Features({
            'input_ids': Sequence(dataset_input_id_type),
            'token_type_ids': Sequence(dataset_token_type_id_type),
            'attention_mask': Sequence(dataset_attention_mask_type),
            # 'dataset_index': Value(dtype="int16"),
            # 'dataset_name': Value(dtype="string"),
        })
        
        # Build the full train / test split dataset
        final_dataset = Dataset.from_dict({key: [None,None] for key in final_dataset_features}, features=final_dataset_features)
        final_dataset = final_dataset.train_test_split(1,1)

        # Lets override the full dataset
        final_dataset["train"] = final_trainset
        final_dataset["test"] = final_testset

    # Log the saving process

    # Finally save to disk
    if "data_path" in datapack_config and datapack_config["data_path"]:
        print(">> Saving dataset to data_path : ", datapack_config["data_path"])
        if "data_path_storage_options" in datapack_config and datapack_config["data_path_storage_options"]:
            final_dataset.save_to_disk(
                datapack_config["data_path"], 
                storage_options=datapack_config["data_path_storage_options"]
            )
        else:
            final_dataset.save_to_disk(
                datapack_config["data_path"]
            )
        print(">> Dataset saved to data_path")
    else:
        print(">> Skipping dataset saving to disk")

    print(">> -----------------------------------")
    print(">> Performing dataset counting")
    print(">> -----------------------------------")

    # Log the finished dataset sizes
    final_train_len = len(final_dataset["train"])
    final_test_len = len(final_dataset["test"])
    print(">> Final dataset count ( train ) :", "{:,}".format(final_train_len), " samples/chunks/packs")
    print(">> Final dataset count ( test  ) :", "{:,}".format(final_test_len), " samples")
    print(">> -----------------------------------")

    ## Rapid return
    if return_without_counting:
        return final_dataset

    # Compute the total dataset token count
    def compute_lengths(x):
        return {
            'total_tokens': len(x["input_ids"]),
            'valid_tokens': sum(x["attention_mask"]),
        }
    
    # Count the training data
    train_counting = final_dataset["train"].map(compute_lengths, num_proc=num_cpus)
    train_total = sum( train_counting["total_tokens"] )
    train_valid = sum( train_counting["valid_tokens"] )
    train_hidden = train_total - train_valid

    # Count the test data
    test_counting = final_dataset["test"].map(compute_lengths, num_proc=num_cpus)
    test_total = sum( test_counting["total_tokens"] )
    test_valid = sum( test_counting["valid_tokens"] )
    test_hidden = test_total - test_valid

    print(">> -----------------------------------")
    print(">> Final 'train' dataset token count ...")
    print(">> - Total tokens :", "{:,}".format(train_total))
    print(">> - Valid tokens :", "{:,}".format(train_valid))
    print(">> - Hidden tokens :", "{:,}".format(train_hidden))
    print(">> -----------------------------------")
    print(">> Final 'test' dataset token count ...")
    print(">> - Total tokens :", "{:,}".format(test_total))
    print(">> - Valid tokens :", "{:,}".format(test_valid))
    print(">> - Hidden tokens :", "{:,}".format(test_hidden))
    print(">> -----------------------------------")

    # Return the final dataset
    return final_dataset

#
# CheckPointResumeSafeDataLoader
#
# ## Problem Stage 1:
# For some insane reason, neither pytorch, nor pytorch lightning supports a safe way to continue from checkpoint
# with the dataset loader restored to the correct position. This is a INSANE problem, as it means that the training
# will start from the beginning of the dataset, and not from the last checkpointed position.
#
# See: https://discuss.pytorch.org/t/resume-iterating-dataloader-from-checkpoint-batch-idx/60683/14 
#
# ## Problem Stage 2:
# Sound not too hard to solve right?, we can offset the dataset loader as we load it up, or we can skip the first X
# as the dataset loader is being iterated right? Well except that the "dataset", and "model" is loaded first, then
# the checkpoint itself, then the train loop begins. Of which the entire checkpoint -> train loop process is handled
# purely within pytorch lightning code, with no reliable way to hook into the process.
#
# ## Solution:
# This custom dataloader, which has a hook for the model varaible itself. When the model loads with the learning rate
# scheduler - it will "install" itself into the dataloader. After which the normal pytorch lightning process will
# continue, and restore the checkpoint. From which we will use the undocumented variable "_batches_that_stepped" to
# read the current batch status that was "stored/restored" into the pytorch lightning training loop process, to figure
# out the current batch status. When the dataset iteration begins, we will immediately skip all the respective datapoints
# that is below the batch status, and continue from there.
#
class CheckPointResumeSafeDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self._model = None

    def _set_model_self(self, model):
        self._model = model

    def __iter__(self):
        batch_iterator = super().__iter__()
        
        i = 0
        for batch in batch_iterator:
            i += 1

            skip_offset = -1
            if self._model is not None:
                skip_offset = self._model.trainer.fit_loop.epoch_loop._batches_that_stepped * self._model.trainer.accumulate_grad_batches
            
            # We skip the first X steps, which should not be iterated on
            if i <= skip_offset:
                continue

            yield batch

class RWKVDataModule(LightningDataModule):
    def __init__(
        self, 

        # Skip database setup checks if datapath exists, ignored if using preload_datapath.py
        skip_datapath_setup: bool = False,
        
        # Datapack config yaml to use instead, this overwrites all other settings below
        datapack_config_path:str = None,

        # ---

        # load_from_disk(dataset_path) param
        data_path: str = None,
        # Data path storage options, this is used to support cloud storage
        # via the huggingface dataset API. See:
        # https://huggingface.co/docs/datasets/v2.16.1/en/filesystems#amazon-s3
        # Note: As of Jan 2023, these options seems very buggy, YMMV
        data_path_storage_options:dict = None,

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
        tokenizer: str = "world",
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
        min_token_size: int = 2,
        max_token_size: int = -1,
        
        # Sort by length
        sort_by_length: bool = False,
        sort_asc: bool = True,

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
        # Specal use caes flags
        # ----------------------------

        # Reverse the training dataset order before saving, this is useful for,
        # optimizing dataset packing process, when using packing_in_sequence
        # and sort_by_length desc order together
        reverse_train_dataset_before_save: bool = False,

        # ----------------------------
        # System tweaks
        # ----------------------------

        # Batch size scanning range, used for deciding the max number of documents
        # to process simultaneously at a time. This is used to prevent OOM errors
        # while rearranging the dataset, etc. Used for both packing / sorting operations
        processing_max_batch_size: int = 100 * 1000,

        # Dataloader shuffling, disabled if "sort_by_length" is enabled
        dataloader_shuffle_training: bool = False,

        # With a total of 4 batches prefetched into memory
        dataloader_prefetch_factor:int = 4,

        # Pin the preloaded documents into GPU memory in advance
        # very small overhead, slight speed bump, disable if your deperate for vram
        dataloader_pin_memory: bool = True,

        # Swap the train / test split, used for debugging purposes
        dataloader_swap_train_test_split: bool = False,
    ):
        # Capture the init parameters
        self._init_locals = locals()
        del self._init_locals["self"]
        del self._init_locals["__class__"]
        
        super().__init__()
        self.datapack_config_path = datapack_config_path
        self.data_path = data_path
        self.data_path_storage_options = data_path_storage_options
        self.dataloader_prefetch_factor = dataloader_prefetch_factor
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_shuffle_training = dataloader_shuffle_training
        self.sort_by_length = sort_by_length
        self.dataloader_swap_train_test_split = dataloader_swap_train_test_split

        self._loaded_dataset = None

        # Log to wandb
        if wandb.run is not None:
            wandb.config.update({ "data":dict(self._init_locals) })
    
    # Called once for initial setup
    def prepare_data(self):
        prepare_data_static(**self._init_locals)
    
    # Setup process that is universal
    def _internal_setup(self):
        if self._loaded_dataset is None:

            # Load from a datapack
            if self.datapack_config_path is not None:
                # Get the yaml config
                yaml_config = None
                with open(self.datapack_config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)

                # Check if the data is configured, else throw error (default assertion)
                assert 'datapack' in datapack_config, "`datapack` is not configured in the config file"
                assert 'default'  in datapack_config, "`default` is not configured in the config file"
                assert 'dataset'  in datapack_config, "`dataset` is not configured in the config file"

                # Disable preloadonly mode, etc
                yaml_config["datapack"] = { **yaml_config["datapack"], **{ "only_preload": Fals }}

                # Get the loaded datapack
                self._loaded_dataset = prepare_datapack_static(
                    skip_datapath_setup=skip_datapath_setup,
                    return_without_counting=True,
                    **yaml_config
                )

            if self.data_path_storage_options:
                self._loaded_dataset = load_from_disk(self.data_path, storage_options=self.data_path_storage_options).with_format('torch')
            else:
                self._loaded_dataset = load_from_disk(self.data_path).with_format('torch')

    # Called once for every process in DDP
    def setup(self, stage):
        self._internal_setup()

    # Return the train dataloader
    def train_dataloader(self):
        self._internal_setup()

        if self.dataloader_swap_train_test_split == False:
            dataset = self._loaded_dataset['train'];
        else:
            dataset = self._loaded_dataset['test'];

        microbatch_size = 1
        if hasattr(self, "trainer") and hasattr(self.trainer, "microbatch_size"):
            microbatch_size = self.trainer.microbatch_size

        # # Batched sampler
        # batch_sampler = BatchSampler(
        #     SequentialSampler(dataset),
        #     batch_size=microbatch_size,
        #     drop_last=True
        # )

        # # Distributed sampler
        # distributed_sampler = DistributedSampler(
        #     dataset, 
        #     shuffle=self.dataloader_shuffle_training and not self.sort_by_length,
        #     num_replicas=self.trainer.world_size,
        #     rank=self.trainer.global_rank,
        #     ## This is required due to multi node alignment errors
        #     drop_last=True
        # )

        _train_sampler = DistributedSampler(
            dataset, 
            shuffle=self.dataloader_shuffle_training and not self.sort_by_length,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            ## This is required due to multi node alignment errors
            drop_last=True
        )
        self._train_sampler = _train_sampler
        self._train_sampler.set_epoch(self.trainer.current_epoch)

        _train_dataloader = CheckPointResumeSafeDataLoader(
            dataset, 
            sampler=_train_sampler,
            shuffle=False,
            # prefetch workers per GPU
            num_workers=self.dataloader_prefetch_factor,
            # Prefetching of X batches
            prefetch_factor=self.dataloader_prefetch_factor,
            # Of batch sizeed datasets
            batch_size=microbatch_size, 
            # The collation function
            collate_fn=dataloader_collator_fn,
            # Pinned in GPU memory
            pin_memory=self.dataloader_pin_memory
        )

        return _train_dataloader
    
    # Return the validation dataloader
    def val_dataloader(self):
        self._internal_setup()
        if self.dataloader_swap_train_test_split == False:
            dataset = self._loaded_dataset['test'];
        else:
            dataset = self._loaded_dataset['train'];
        
        sampler = DistributedSampler(
            dataset, 
            shuffle=False, 
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            ## This is required due to multi node alignment errors
            drop_last=True
        )
        self._val_sampler = sampler

        microbatch_size = 1
        if hasattr(self, "trainer") and hasattr(self.trainer, "microbatch_size"):
            microbatch_size = self.trainer.microbatch_size

        return DataLoader(
            dataset, 
            sampler=sampler,
            shuffle=False,
            # prefetch workers per GPU
            num_workers=self.dataloader_prefetch_factor,
            # Prefetching 8 batches
            prefetch_factor=self.dataloader_prefetch_factor,
            # Of batch sized datasets
            batch_size=microbatch_size, 
            # The collation function
            collate_fn=dataloader_collator_fn,
            # Pinned in GPU memory
            pin_memory=self.dataloader_pin_memory
        )