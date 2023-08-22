# We start from the CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install ninja, and several common stuff
RUN apt-get update && \
    apt-get install -y \
        ninja-build \
        htop wget curl git vim \
        python3.11 python3-pip

# Install pytorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all the various other required packages
RUN pip3 install lightning==2.0.5 deepspeed==0.10.0 \
    datasets transformers \
    ninja numexpr jsonargparse 'jsonargparse[signatures]' \
    lm-dataformat ftfy sentencepiece tokenizers wandb

# Install the misc packages we might need for the various experiments
RUN pip3 install \
    papermill aiocsv aiofiles

# Configure default dir, to the home directory
WORKDIR /root
