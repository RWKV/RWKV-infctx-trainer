# Temporary, until the rwkv package is public
FROM ghcr.io/picocreator/rwkv-lm-lora:env-cuda-11-8
# FROM ghcr.io/rwkv/rwkv-infctx-trainer:env-cuda-11-8

# Install the github runner
RUN cd / && mkdir actions-runner && cd actions-runner && \
    curl -o actions-runner-linux-x64-2.308.0.tar.gz \
    -L https://github.com/actions/runner/releases/download/v2.308.0/actions-runner-linux-x64-2.308.0.tar.gz && \
    tar xzf ./actions-runner-linux-x64-2.308.0.tar.gz && \
    rm ./actions-runner-linux-x64-2.308.0.tar.gz

# Install dependencies
RUN cd actions-runner && ./bin/installdependencies.sh 

# Copy the entrypoint script, and set it up
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Configure default env variables
ENV RUNNER_LABELS="cuda-11-8"
ENV RUNNER_NAME=""
ENV RUNNER_TOKEN=""
ENV RUNNER_REPO_URL="https://github.com/RWKV/RWKV-infctx-trainer"