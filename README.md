# RWKV-X-PLAYGROUND

This is the infctx playground branch, for various highly experimental ideas for RWKV.

THIS BRANCH IS NOT STABLE, AND IS NOT MEANT TO BE USED DIRECTLY / SERIOUSLY. 

I CAN AND WILL BREAK THIS BRANCH AT WILL FOR AN EXPERIMENT.

For the official infctx branch, find it here : https://github.com/Blealtan/RWKV-LM-LoRA/tree/dev-infctx

If you want the bleeding edge infctx branch from picocreator, find it here : https://github.com/PicoCreator/RWKV-LM-LoRA/tree/picocreator-dev-infctx

---

# RWKV Implementation for Infinite Context

RWKV trainer with
- no training context limit (via BPTT)
- deepspeed 3
- HF dataset integration

With this implementation you can train on arbitrarily long context within (near) constant VRAM consumption; this increasing should be, about 2MB per 1024/2048 tokens (depending on your chosen `ctx_len`, with RWKV 7B as an example) in the training sample, which will enable training on sequences over 1M tokens.

The training code is by the way tremendously refactored into using PyTorch 2.0, Lightning 2.0 and DeepSpeed 2.0, and the starting script now relies on LightningCLI so you will see the [config-example.yaml](RWKV-v4neo/config-example.yaml) containing all the switches, mostly standard ones that Lightning processes by itself. And new ones for RWKV and the dataset parser.

To use this repo, go into `RWKV-v4neo` directory and do

```sh
python3 lightning_trainer.py fit -c {your_config}.yaml
```

Remember to modify the configuration for your own need. 

See [RWKV-v4neo/config-example.yaml](./RWKV-v4neo/config-example.yaml) for documentation on the various options

## Environment setup

> Note: There is a known issue with CUDA 12.0 and multi-gpu at this point of writing. Upgrade to CUDA 12.1 or 12.2 atleast Or downgrade to 11.8

The following venv setup using conda, modify for your use case respectively

```shell
# ninja-build is required for the new trainer
sudo apt-get install ninja-build

# Update conda & its package listings
conda update conda

# Virtual env, with python 3.11
conda create -n rwkv-infctx python=3.11 pip
conda activate rwkv-infctx

# Install pytorch (>=2.1)
conda install -y pytorch==2.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python3 -m pip install lightning==2.1 deepspeed==0.12

# Currently for torch.compile + 3.11 to work, for some platform, you will need the nightly build
# if so you may need to try the following instead - this is considered "unstable"
# ---
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia
# python -m pip install lightning==2.0.5 deepspeed==0.10.0

# Verify your pytorch version 
python3 -c "import torch; print(torch.__version__)"

# Install all the other various dependencies
# PS: We use python -m pip, instead of pip directly, as it resolve issues with venv not loading the right pip
python3 -m pip install datasets transformers 
python3 -m pip install ninja numexpr jsonargparse 'jsonargparse[signatures]'
python3 -m pip install lm-dataformat ftfy sentencepiece tokenizers wandb

# Optional dependencies, useful for running notebooks, etc
python3 -m pip install papermill
```

Alternatively you could use the requirements.txt (this may not install pytorch-cuda properly)

```shell
python3 -m pip install -r requirements.txt
```

> Due to issues with [deepspeed on windows](https://github.com/microsoft/DeepSpeed/issues/2427). Only linux environments are supported. WSl2 with windows is not recommended, due to heavy performance penalities in the process (cannot use deepspeed offload, ~50% slower)

## Overall training process

- Either init a new model, or download an existing model
    - To initialize a new model use `python3 ./init_model.py --n_layer {number-of-layers} --n_embd {embedding-size} --vocab_size {vocab-size/neox/world} --skip-if-exists ../model/file/path.pth`
- Setup the [config.yaml](./RWKV-v4neo/config-example.yaml) file, customized for your foundation model / finetune use case
- Preload the dataset using the `python3 preload_datapath.py {you-config}.yaml`
- Start the training process `python3 lightning_trainer.py fit -c {your_config}.yaml`
- Export the checkpoint after training is complete with `python3 export_checkpoint.py ../path/to/checkpoint ../path/to/export/model.pth`
- optional, run the dragon prompt as a quick sanity check `python3 dragon_test.py ../path/to/export/model.pth`
- You should probably convert this to an fp16 model (todo script)

In summary with code, from the trainer directory (eg. RWKV-v4neo)

```shell
# Initialize the blank model (or download a pretrained model)
python3 init_model.py --n_layer {number-of-layers} --n_embd {embedding-size} --vocab_size {vocab-size/neox/world} --skip-if-exists ../model/file/path.pth

# Preload your dataset
python3 preload_datapath.py {you-config}.yaml

# Run the training process
python3 lightning_trainer.py fit -c {your_config}.yaml

# Export the checkpoint to model code
python3 export_checkpoint.py ../path/to/checkpoint ../path/to/export/model.pth

# Quick test the model with the dragon prompt
python3 dragon_test.py ../path/to/export/model.pth

# @TODO, convert the model to bf16 format (instead of the huge fp32 format now)
#        for now you will have to use the RWKV pip package to do this with python code: 
#        https://pypi.org/project/rwkv/
```

## Examples of configuration files

You can find the following notebook/examples at the following ...
- fully annotation of various configs at [./RWKV-v4neo/config-example.py](./RWKV-v4neo/config-example.py)
- minimal config example at [./RWKV-v4neo/config-example.py](./RWKV-v4neo/config-example.py)
- [configuration / notebooks for various dataset usecases here](./notebook/dataset-config/)
- @TODO: training scenerios specific examples

For configuration issues, please review through the examples listed above first, before asking questions on discord.

You can find the training channel on our discord here: https://discord.com/channels/992359628979568762/992362252269256815

## Should I use the official RWKV-LM trainer or the infctx trainer?

Generally if your training a foundation model from scratch - with a fixed context size, and you need the absolute highest throughput across multiple nodes (ie. 10 nodes filled with A100 servers), the [official trainer](https://github.com/BlinkDL/RWKV-LM) should perform better.

If you need deepspeed 3 support, or you deal with dynamic datasets, this trainer is much more flexible, for most nearly all other use cases.

## Some long term architecture goals

- CUDA should be optional
    - Moving forward, this allows us to potentially train (even if its at a perf cost) on other architectures like AMD ROCM, TPU, or Apple M1 architecture.
- No dependency on the official RWKV pip package
    - This is an intentional choice, to help facilitate easy iteration on model architecture in `#rwkv-x` development. So that the entire train-test-validation of design changes can be done in this repository.

## Existing limitations

The following features are not yet supported (that may exist in [blinks original repo](https://github.com/BlinkDL/RWKV-LM))
- numpy file dataset
- model resize weights (init from smaller to bigger model)
- helper script to add new tokens to existing model
- torch compile is NOT supported, as this has been unstable on nightly build

## Designated maintainer

[@picocreator](https://github.com/PicoCreator) - is the current maintainer of the project, you can ping him on the RWKV discord if you have any questions on this project

## Credits (for v4neo and v5 code)

- The bulk of the first infctx trainer was originally rewritten by @Blealtan at : [https://github.com/Blealtan/RWKV-LM-LoRA/tree/dev-infctx](https://github.com/Blealtan/RWKV-LM-LoRA/tree/dev-infctx)
- RWKV-LM and the original trainer code is credited to @BlinkDL at : [https://github.com/BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- Special credit to @Yuzaboto and @bananaman via our RWKV discord, whose assistance was crucial to help debug and fix the repo to work with RWKVv4 and RWKVv5 code respectively.
- PyTorch Lightning team @lantiga and @Adrian via Pytorch LIghtning AI discord - who assisted in clarifying questions on pytorch lightning
- [@picocreator](https://github.com/PicoCreator) for getting the project feature complete for RWKV mainline release

> This project was intentionally a hard fork, as it has too many conflicting changes to the official RWKV-LM repo
