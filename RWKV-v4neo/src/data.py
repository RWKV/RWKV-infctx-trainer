from lightning import LightningDataModule

from torch.utils.data import Dataset

from datasets import load_from_disk, load_dataset
from transformers import PreTrainedTokenizerFast


def get_data_module(data_path: str,
                    source: str = None,
                    tokenizer: str = None) -> LightningDataModule:
    if source is not None:
        src_dataset = load_dataset(source, split='train')
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer)
        src_dataset = src_dataset.map(lambda x: tokenizer(x['text']),
                                      batched=True)
        src_dataset = src_dataset.train_test_split(test_size=0.1,
                                                   shuffle=False)
        src_dataset.save_to_disk(data_path)

    dataset = load_from_disk(data_path).with_format('torch')
    return LightningDataModule.from_datasets(dataset['train'], dataset['test'])
