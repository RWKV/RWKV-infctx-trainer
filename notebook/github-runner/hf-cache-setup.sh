#!/bin/bash

# -----
# Required ARGS check
# -----

# Check if HUGGING_FACE_HUB_TOKEN & WANDB_API_KEY is set
if [[ -z "${HUGGING_FACE_HUB_TOKEN}" ]]; then
    echo "[ERROR]: HUGGING_FACE_HUB_TOKEN is not set"
    exit 1
fi
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "[ERROR]: WANDB_API_KEY is not set"
    exit 1
fi

# Get the current script directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NOTEBOOK_DIR="$(dirname "$SCRIPT_DIR")"
PROJ_DIR="$(dirname "$NOTEBOOK_DIR")"

# Assume the ACTION dir, is 4 dir levels up
ACTION_DIR="$(dirname "$PROJ_DIR/../../../../")"
ACTION_DIR="$(cd "$ACTION_DIR" && pwd)"

# Cache dir to use when possible
CACHE_DIR="$ACTION_DIR/.cache/"
mkdir -p "$CACHE_DIR"

# Log the proj dir
echo "# ------------------------------"
echo "# Starting HF cache"
echo "# CACHE_DIR: $CACHE_DIR"
echo "# ------------------------------"

# -----
# Cache dir size check
# -----

# Convert size to bytes
convert_to_bytes() {
    local size=$1
    if [[ $size == *G ]]; then
        size=${size%G}
        size=$((size*1024*1024*1024)) # 1G = 1073741824 bytes
    elif [[ $size == *M ]]; then
        size=${size%M}
        size=$((size*1024*1024))    # 1M = 1048576 bytes
    elif [[ $size == *K ]]; then
        size=${size%K}
        size=$((size*1024))       # 1K = 1024 bytes
    fi
    echo $size
}

if [[ -z "${RUNNER_CACHE_SIZE_LIMIT}" ]]; then
    RUNNER_CACHE_SIZE_LIMIT="100G"
fi
RUNNER_CACHE_SIZE_LIMIT_BYTES=$(convert_to_bytes $RUNNER_CACHE_SIZE_LIMIT)

# Get the cache directory size
CACHE_SIZE=$(du -sh $CACHE_DIR | awk '{print $1}')
CACHE_SIZE_BYTES=$(convert_to_bytes $CACHE_SIZE)

# If the cache dir is larger then RUNNER_CACHE_SIZE_LIMIT, then delete the cache dir
if [[ "$CACHE_SIZE_BYTES" -gt "$RUNNER_CACHE_SIZE_LIMIT_BYTES" ]]; then
    echo "# [NOTE] Cache dir size ($CACHE_SIZE) is larger/equal to RUNNER_CACHE_SIZE_LIMIT ($RUNNER_CACHE_SIZE_LIMIT)"
    echo "# [NOTE] Resetting cache dir: $CACHE_DIR"
    rm -rf "$CACHE_DIR"
    mkdir -p "$CACHE_DIR"
else
    echo "# [NOTE] Cache dir size: ~$CACHE_SIZE"
fi

# Cofigure the HF cache dir
export HF_HOME="$CACHE_DIR/huggingface"
mkdir -p "$HF_HOME"

# Get the cache directory size
CACHE_SIZE=$(du -sh $CACHE_DIR | awk '{print $1}')
echo "# [NOTE] Final Cache dir size: ~$CACHE_SIZE"