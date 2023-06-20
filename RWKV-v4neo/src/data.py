from lightning import LightningDataModule

from torch.utils.data import Dataset

from datasets import load_from_disk, load_dataset
from transformers import PreTrainedTokenizerFast
from multiprocessing import cpu_count

def get_data_module(
        # load_from_disk(dataset_path) param
        data_path: str,
        # load_dataset(path) param
        source: str = None,
        # load_dataset(data_dir) param
        source_data_dir: str = None,
        # Test split of source data, if it was not already done
        test_split: float = 0.1,
        test_split_shuffle: bool = False,
        # Text rechunking size
        text_chunk_size: int = 2048,
        # Custom tokenizer settings
        tokenizer: str = "neox",
        disablePromptCompletionMasking: bool = False
    ) -> LightningDataModule:
    if source is not None:
        if tokenizer is None:
            raise ValueError('Tokenizer must be specified if source is specified')

        # Setup the basic load_dataset params
        num_cpus = cpu_count()
        load_dataset_params = {
            'path': data_path,
            'num_proc': num_cpus
        }

        # Handle advance params (if set)
        if source_data_dir is not None:
            load_dataset_params['data_dir'] = source_data_dir

        # Load the dataset
        src_dataset = load_dataset(**load_dataset_params)

        # Load the tokenizer
        # according to either its predefined name or its path
        # (defaults to neox)
        if tokenizer == "neox":
            tokenizer_file = "./20B_tokenizer.json"
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        elif tokenizer == "world":
            raise NotImplementedError("World tokenizer not implemented yet")
        else:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer)

        # Maps the dataset to the tokenizer
        # handles prompt / completion if its present, otherwise just tokenizes the text
        def map_tokenizer(x):
            if 'prompt' in x and 'completion' in x:
                # Array of output valeus we will return
                input_ids = []
                token_type_ids = []
                attention_mask = []

                # Tokenize both prompt and completion
                # Note that the tokenizer will process and return the input_ids in batches
                prompt_encodings = tokenizer(x['prompt'])
                completion_encodings = tokenizer(x['completion'])

                # Important note, prompt_encodings['input_ids'] are list, containing list of the actual values
                # so we need to process them accordingly (batch processing)
                for i in range(len(prompt_encodings['input_ids'])):
                    # Join the two input_ids lists
                    input_ids.append(prompt_encodings['input_ids'][i] + completion_encodings['input_ids'][i])
                    # Join the two token_type_ids lists
                    token_type_ids.append(prompt_encodings['token_type_ids'][i] + completion_encodings['token_type_ids'][i])
                    # Setup the attention mask, 0 for prompt, 1 for completion, if masking is enabled
                    if disablePromptCompletionMasking:
                        attention_mask.append([1] * len(prompt_encodings['input_ids'][i]) + [1] * len(completion_encodings['input_ids'][i]))
                    else:
                        attention_mask.append([0] * len(prompt_encodings['input_ids'][i]) + [1] * len(completion_encodings['input_ids'][i]))

                # Prepare and return the output object
                ret = {
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                }
                return ret
            else:
                # Fallback to standard text tokenization
                return tokenizer(x['text'])

        # Map the dataset to the tokenizer, removing the old text column
        train_features = src_dataset['train'].features
        if 'prompt' in train_features and 'completion' in train_features:
            src_dataset = src_dataset.map(map_tokenizer, batched=True, num_proc=num_cpus, remove_columns=['prompt', 'completion'])
        else:
            src_dataset = src_dataset.map(map_tokenizer, batched=True, num_proc=num_cpus, remove_columns=['text'])

        # See if rechunking is needed, this is useful only for text based datasets
        # where we would need to split them into "digestable" context length sizes
        if source == "text" and text_chunk_size > 0:
            # Get the newline token
            newline_tokenSet = tokenizer(["\n"])

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
                total_samples = total_len // text_chunk_size

                # The output arrays
                out_input_ids = []
                out_token_type_ids = []
                out_attention_mask = []

                # Generate the output arrays
                for i in range(total_samples):
                    # Calculate the start and end of the sample
                    start = i * text_chunk_size
                    end = start + text_chunk_size

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

            # Perform the rechunking
            src_dataset = src_dataset.map(rechunk_text, batched=True, 
                                          batch_size=text_chunk_size*10,
                                          num_proc=num_cpus)

        # Check if the dataset does not have a test split
        # and if so, perform the split
        if 'test' not in src_dataset.keys():
            src_dataset = src_dataset['train'].train_test_split(test_size=test_split,shuffle=test_split_shuffle)
        
        # Save the dataset to disk
        src_dataset.save_to_disk(data_path)

    # Load the dataset as per normal 
    dataset = load_from_disk(data_path).with_format('torch')
    return LightningDataModule.from_datasets(dataset['train'], dataset['test'])
