#!/bin/bash

export RUNNER_ALLOW_RUNASROOT="1"
cd /actions-runner

# Check the URL, token, and name of the runner from the container ENV vars
# and if they are not set, provide default values
if [[ -z "${RUNNER_NAME}" ]]; then
    export RUNNER_NAME=$(hostname)
fi
if [[ -z "${RUNNER_TOKEN}" ]]; then
    echo "# [WARNING] RUNNER_TOKEN is missing, skipping github runner setup"
else
    # Configure unattended
    ./config.sh \
        --unattended \
        --url "${RUNNER_REPO_URL}" \
        --token "${RUNNER_TOKEN}" \
        --name "${RUNNER_NAME}" \
        --replace \
        --labels "${RUNNER_LABELS}"

    # Run it in background, and get the PID
    ./run.sh &
    RUNNER_PID=$!
fi

# Follow up on any forwarded command args
if [[ $# -gt 0 ]]; then
    exec "$@"
fi

# Wait for everything to exit
# wait $RUNNER_PID
wait 