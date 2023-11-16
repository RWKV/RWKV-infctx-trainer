from lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

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
        
        # Special handling for binidx
        #--------------------------------

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
                        'attention_mask': [1] * len(tokens)
                    }

            # Load the huggingface dataset from the generator
            src_dataset = Dataset.from_generator(gen)

            # Train/test split
            test_split = kargs["test_split"]
            # The minimum test size is 1, if not we will get errors in the trainer?
            if test_split <= 0 or test_split <= 0.0:
                test_split = 1
            split_dataset = src_dataset.train_test_split(
                test_size=test_split,shuffle=kargs["test_split_shuffle"],
                seed=42 #Fixed seed, to prevent train/test reshuffling between test runs
            )

            # Save the dataset to disk
            split_dataset.save_to_disk(kargs["data_path"])
            # Does nothing else (done)
            return

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
                        'attention_mask': attention_mask
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
                    'attention_mask': attention_mask,
                }
            
            # Fallback to standard text tokenization
            if 'text' in x:
                return encodeTokens(x['text'])
            
            raise ValueError('Invalid dataset format, must contain either the configured "multi column" or prompt/completion or text')

        # Map the dataset to the tokenizer, removing the old text column
        src_dataset = src_dataset.map(map_tokenizer, batched=False, num_proc=num_cpus)
        
        # Remove all features, except input_ids, token_type_ids and attention_mask
        # as the metadata/etc columns may cause problems down the line (when passed to the trainer)
        dataset_features = src_dataset["train"].features
        dataset_features_to_remove = {k: v for k, v in dataset_features.items() if k not in ["input_ids", "token_type_ids", "attention_mask"]}
        src_dataset = src_dataset.remove_columns(list(dataset_features_to_remove.keys()))
        
        # Get the newline token
        endOfDoc_tokenSet = encodeTokens(["\n"])
        endOfDoc_tokenSet["input_ids"][0][0] = 0

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
                full_attention_mask += x["attention_mask"][i] + endOfDoc_tokenSet["attention_mask"][0]
            
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
                out_attention_mask.append(full_attention_mask[start:end])
            
            # Prepare and return the output object
            ret = {
                'input_ids': out_input_ids,
                'token_type_ids': out_token_type_ids,
                'attention_mask': out_attention_mask,
            }
            return ret

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
        
        # Perform rechunking if needed for "text" based datasets
        if kargs["source"] == "text" and kargs["text_rechunk_size"] > 0 and kargs["text_rechunk_auto"]:
            src_dataset = src_dataset.map(rechunk_text, batched=True, 
                                        batch_size=kargs["text_rechunk_size"]*10,
                                        num_proc=num_cpus)
        
        # Perform rechunking after filtering, if source is not a "text" based 
        # dataset and text_rechunk_force is enabled
        if kargs["source"] != "text" and kargs["text_rechunk_size"] > 0 and kargs["text_rechunk_force"]:
            src_dataset = src_dataset.map(rechunk_text, batched=True, 
                                        batch_size=kargs["text_rechunk_size"]*2,
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
        
        # Perform a sort by length, only after test split
        if kargs["sort_by_length"]:
            sort_asc = kargs["sort_asc"]
            
            def add_length(example):
                example["input_length"] = len(example['input_ids'])
                return example
            
            src_dataset['train'] = src_dataset['train'].map(add_length, batched=False, num_proc=num_cpus)
            
            # sort by length (not sorting the columns, just the rows)
            src_dataset['train'] = src_dataset['train'].sort("input_length", reverse=not sort_asc)

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
        min_token_size: int = -1,
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
        # prompt/completion format masking support
        disable_prompt_completion_mask: bool = False,
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
