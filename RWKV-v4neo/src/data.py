from lightning import LightningDataModule

from torch.utils.data import DataLoader

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
                        enc_str = world_tokenizer_encode(x[i])
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
                enc_str = world_tokenizer_encode(x)
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
            multi_column_train_mask = [True, False, True]
            multi_column_separator = '\n\n'
        else:
            multi_column_keys = kargs["multi_column_keys"]
            multi_column_prefix = kargs["multi_column_prefix"]
            multi_column_train_mask = kargs["multi_column_train_mask"]
            multi_column_separator = kargs["multi_column_separator"]
        
        # Tokenized encodings for multi column keys
        multi_column_enabled = len(multi_column_keys) > 0
        multi_column_prefix_encodings = []
        multi_column_separator_encodings = None

        # Process the multi column settings
        if multi_column_enabled:
            # Check if the multi column keys lengths are valid (only if it is enabled)
            if len(multi_column_keys) != len(multi_column_prefix) or len(multi_column_keys) != len(multi_column_train_mask):
                raise ValueError('Multi column keys, prefix and masking must be the same length')
            # Tokenize the multi column strings
            for i in range(len(multi_column_keys)):
                multi_column_prefix_encodings.append(encodeTokens(multi_column_prefix[i]))
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
                            input_ids += multi_column_prefix_encodings[i]['input_ids']
                            token_type_ids += multi_column_prefix_encodings[i]['token_type_ids']
                            attention_mask += multi_column_prefix_encodings[i]['attention_mask']

                            # Tokenize the column
                            column_encodings = encodeTokens(x[multi_column_keys[i]])

                            # Add the column
                            input_ids += column_encodings['input_ids']
                            token_type_ids += column_encodings['token_type_ids']

                            # Override the attention mask if masking is enabled
                            if multi_column_train_mask[i]:
                                attention_mask += ([1] * len(column_encodings['input_ids']))
                            else:
                                attention_mask += ([0] * len(column_encodings['input_ids']))
                    
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
        newline_tokenSet = encodeTokens(["\n"])

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
                full_input_ids += x["input_ids"][i] + newline_tokenSet["input_ids"][0]
                full_token_type_ids += x["token_type_ids"][i] + newline_tokenSet["token_type_ids"][0]
                full_attention_mask += x["attention_mask"][i] + newline_tokenSet["attention_mask"][0]
            
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

        # Perform rechunking if needed for "text" based datasets
        if kargs["source"] == "text" and kargs["text_rechunk_size"] > 0 and kargs["text_rechunk_force"]:
            src_dataset = src_dataset.map(rechunk_text, batched=True, 
                                        batch_size=kargs["text_rechunk_size"]*10,
                                        num_proc=num_cpus)
        
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
        
        # Save the dataset to disk
        src_dataset.save_to_disk(kargs["data_path"])


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
        text_rechunk_force: bool = False,
        # ---
        # Tokenizer settings
        # ---
        tokenizer: str = "neox",
        autoTokenizer = None,
        # ---
        # HF dataset conversion helpers
        # ---
        # Min / Max token size filtering
        min_token_size: int = -1,
        max_token_size: int = -1,
        # Custom 'text' column to support, mostly used for dataset where the 
        # desired train data is in another column (eg. 'code')
        custom_text_key: str = None,
        # Multi column merging support, used for instruct/input/output datasets
        # or similar varients where the input and output are in different columns
        # and need to be merged
        multi_column_keys: list = None,
        multi_column_prefix: list = None,
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
        return DataLoader(self._loaded_dataset['train'], num_workers=num_workers)
    
    # Return the validation dataloader
    def val_dataloader(self):
        self._internal_setup()
        return DataLoader(self._loaded_dataset['test'], num_workers=num_workers)
