# We start from the CUDA base image
# todo: consider other base images?
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install ninja, and several common stuff
RUN apt-get update && \
    apt-get install -y \
        ninja-build \
        htop wget curl git vim \
        python3.11 python3-pip \
        python-is-python3

# Install pytorch
RUN pip3 install torch torchvision torchaudio

# Install all the various other required packages
RUN pip3 install lightning==2.1.3 deepspeed==0.12.6 \
    datasets transformers \
    ninja numexpr jsonargparse 'jsonargparse[signatures]' \
    lm-dataformat ftfy sentencepiece tokenizers wandb

# Install the misc packages we might need for the various experiments
RUN pip3 install \
    papermill aiocsv aiofiles huggingface_hub[cli]

# Python subkernel stuff (for papermill notebook)
RUN pip3 install ipython ipykernel
RUN ipython kernel install --name "python3" --user

# Configure default dir, to the home directory
WORKDIR /root
