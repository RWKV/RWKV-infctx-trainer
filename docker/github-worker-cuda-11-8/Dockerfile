# Temporary, until the rwkv package is public
FROM ghcr.io/picocreator/rwkv-lm-lora:env-cuda-11-8
# FROM ghcr.io/rwkv/rwkv-infctx-trainer:env-cuda-11-8

RUN mkdir actions-runner && cd actions-runner && \
    curl -o actions-runner-linux-x64-2.308.0.tar.gz \
    -L https://github.com/actions/runner/releases/download/v2.308.0/actions-runner-linux-x64-2.308.0.tar.gz && \
    tar xzf ./actions-runner-linux-x64-2.308.0.tar.gz
