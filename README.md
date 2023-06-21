# RWKV Implementation for Infinite Context

This branch contains my experimental attempts to achieve infinite context training in RWKV.
With this implementation you can train on arbitrarily long context within (near) constant VRAM consumption; the increasing should be, take RWKV 7B as an example, about 2MB per 1024/2048 tokens (depending on your chosen `ctx_len`) in the training sample, which will enable training on sequences over 1M tokens.
Yet directly tune to such long sequences might be problematic; so `ctx_len_cutoff` is provided so longer sequences are sliced into multiple pieces of the specified cutoff size and learnt by the model separately.
It can be later increased until no cutoff presents.

The training code is by the way tremendously refactored into using PyTorch 2.0, Lightning 2.0 and DeepSpeed 2.0, and the starting script now relies on LightningCLI so you will see the [config.yaml](RWKV-v4neo/config-7B.yaml) containing all the switches, mostly standard ones that Lightning processes by itself.

The data loading is also rewritten so that it no longer accepts binidx format (I'm a bit lazy!), instead it accepts HuggingFace Datasets through the `data.source` configuration, tokenize with `data.tokenizer` (e.g. `20B_tokenizer.json` provided in the repo for RWKV-4-pile series models; I haven't work on the new RWKV-4-world models yet) and stores (also in HuggingFace format) at `data.data_path`.
Once you have prepared the tokenized data, you can leave the `data.{source, tokenizer}` empty to skip the preprocessing.
Please notice that it currently supports only batch size of 1, and accepts no `bsz` parameter.

Besides, I removed the weight initialization for the sake of simplicity, so it doesn't support training ground up now.
The weight has to be loaded from a model file, compatible with the original format by @BlinkDL used in [RWKV-LM](https://github.com/BlinkDL/RWKV-LM).

To use this repo, go into `RWKV-v4neo` directory and do

```sh
python3 new_train.py -c {your_config}.yaml
```

Remember to modify the configuration for your own need. 

See [RWKV-v4neo/config-example.yaml](./RWKV-v4neo/config-example.yaml) for documentation on the various options

## Environment setup

The following venv setup using conda, modify for your use case respectively
```
# ninja-build is required for the new trainer
sudo apt-get install ninja-build

# Virtual env, with python 3.11
conda create -n rwkv-exp python=3.11 pip
conda activate rwkv-exp

# Install pytorch
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# We use python -m pip, instead of pip directly, as it resolve issues with venv not loading the right pip
python -m pip install datasets transformers 
python -m pip install lightning==2.0.2 deepspeed==0.9.3 
python -m pip install ninja numexpr jsonargparse 'jsonargparse[signatures]'
python -m pip install lm-dataformat ftfy sentencepiece tokenizers wandb
```

Due to issues with [deepspeed on windows](https://github.com/microsoft/DeepSpeed/issues/2427). Only linux environments are supported. WSl2 with windows is not recommended, due to heavy performance penalities in the process (cannot use deepspeed offload, ~50% slower)