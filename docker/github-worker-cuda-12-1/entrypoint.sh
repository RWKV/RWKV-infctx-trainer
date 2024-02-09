#!/bin/bash

export RUNNER_ALLOW_RUNASROOT="1"
cd /actions-runner

# CUDA version for label
CUDA_VER="cuda-12-1"

# Check if nvidia-smi is available
SMI_CALL=$(nvidia-smi)
if [ $? -ne 0 ]; then
    echo "# [ERROR] nvidia-smi is not available, shutting down (after long delay)"
    echo "# ---"
    echo "$SMI_CALL"
    echo "# ---"

    # Sleep for 1 hour, a failure on start is a sign of a bigger problem
    echo "# Performing a long sleep, to stagger container flow ..."
    sleep 3600
    echo "# Exiting now"
    exit 1
fi

# Check the URL, token, and name of the runner from the container ENV vars
# and if they are not set, provide default values
if [[ -z "${RUNNER_NAME}" ]]; then
    export RUNNER_NAME=$(hostname)
fi
if [[ -z "${RUNNER_TOKEN}" ]]; then
    echo "# [WARNING] RUNNER_TOKEN is missing, skipping github runner setup"
else
    echo "# [INFO] lane1 starting up ... "

    # If lane2 runner is enabled, start it
    # this is enabled with RUNNER_LANE2=true
    if [ "$RUNNER_LANE2" != true ]; then

        # Configure unattended
        ./config.sh \
            --unattended \
            --url "${RUNNER_REPO_URL}" \
            --token "${RUNNER_TOKEN}" \
            --name "${RUNNER_NAME}" \
            --replace \
            --labels "nolane,${CUDA_VER},${RUNNER_LABELS}"

        # Run it in background, and get the PID
        ./run.sh &

        echo "# [INFO] lane2 runner is disabled"
    else
        # Configure unattended
        ./config.sh \
            --unattended \
            --url "${RUNNER_REPO_URL}" \
            --token "${RUNNER_TOKEN}" \
            --name "${RUNNER_NAME}-lane1" \
            --replace \
            --labels "lane1,${CUDA_VER},${RUNNER_LABELS}"

        # Run it in background, and get the PID
        ./run.sh &

        echo "# [INFO] lane2 starting up ... "

        cd /actions-runner-lane2
        ./config.sh \
            --unattended \
            --url "${RUNNER_REPO_URL}" \
            --token "${RUNNER_TOKEN}" \
            --name "${RUNNER_NAME}-lane2" \
            --replace \
            --labels "lane2,${CUDA_VER},${RUNNER_LABELS}"

        # Run it in background, and get the PID
        ./run.sh &
    fi
fi

# Follow up on any forwarded command args
if [[ $# -gt 0 ]]; then
    cd /root
    exec "$@" &
fi

# Wait for everything to exit
# wait $RUNNER_PID
while true; do
    # Check if any background process is still running
    # Break if there is no background process
    if [ -z "$(jobs -p)" ]; then
        echo "# [INFO] All runners have exited, shutting down"
        break
    fi

    # Call nvidia-smi, check if any error orccured
    SMI_CALL=$(nvidia-smi)

    # If nvidia-smi failed, exit
    if [ $? -ne 0 ]; then
        echo "# [ERROR] nvidia-smi failed, shutting down now!"
        echo "# ---"
        echo "$SMI_CALL"
        echo "# ---"
        break
    fi

    # Performs a small sleep wait
    sleep 10
done