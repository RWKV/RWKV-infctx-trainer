# Temporary, until the rwkv package is public
FROM ghcr.io/picocreator/rwkv-lm-lora:env-cuda-11-8
# FROM ghcr.io/rwkv/rwkv-infctx-trainer:env-cuda-11-8

# Install the github runner
RUN cd / && mkdir actions-runner && cd actions-runner && \
    curl -o actions-runner-linux-x64-2.308.0.tar.gz \
    -L https://github.com/actions/runner/releases/download/v2.308.0/actions-runner-linux-x64-2.308.0.tar.gz && \
    tar xzf ./actions-runner-linux-x64-2.308.0.tar.gz && \
    rm ./actions-runner-linux-x64-2.308.0.tar.gz

# Clone the runner, for lane2 track
RUN cd / && cp -r /actions-runner /actions-runner-lane2

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

# Runner with lane2 track
# ---
# this helps setup dual runs on the same machine
# to help ensure better utilization of GPUs.
#
# In general DS2/3_offload should be used.
#
# Tags should be adjusted to be half their original spec
# to account for the fact that we are running two runners
#
# This is only useful for high GPU, and high ram count machines
ENV RUNNER_LANE2="false"