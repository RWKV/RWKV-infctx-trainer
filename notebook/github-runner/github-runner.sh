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

# The HF repo directory to use
if [[ -z "${HF_REPO_SYNC}" ]]; then
    HF_REPO_SYNC="rwkv-x-dev/rwkv-x-playground"
fi

# Get the notebook script from the first arg
NOTEBOOK_FILE=$1
NOTEBOOK_FILE="$(echo -e "${NOTEBOOK_FILE}" | sed -e 's/[[:space:]]*$//')"

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
echo "# ------"
echo "# Starting github notebook runner"
echo "#"
echo "# PROJ_DIR: $PROJ_DIR"
echo "# NOTEBOOK_DIR: $NOTEBOOK_DIR"
echo "# NOTEBOOK_FILE: $NOTEBOOK_FILE"
echo "#"
echo "# CACHE_DIR: $CACHE_DIR"
echo "# ------"

# Check if the notebook file exists, in the notebook directory
if [[ ! -f "$NOTEBOOK_DIR/$NOTEBOOK_FILE" ]]; then
    echo "[ERROR]: Notebook file does not exist ($NOTEBOOK_FILE)"
    exit 1
fi

# Cofigure the HF cache dir
export HF_HOME="$CACHE_DIR/huggingface"
mkdir -p "$HF_HOME"

# -----
# Ensuring HF CLI / wandb is installed
# -----

echo "# [NOTE] Ensuring huggingface_hub[cli] / wandb is updated"
python3 -m pip install huggingface_hub[cli] wandb 
python3 -m pip install ipython ipykernel
ipython kernel install --name "python3" --user

# -----
# Project dir resets setup
# -----

rm -rf $PROJ_DIR/checkpoint
mkdir -p $PROJ_DIR/checkpoint

rm -rf $PROJ_DIR/datapath
mkdir -p $PROJ_DIR/datapath

rm -rf $PROJ_DIR/output
mkdir -p $PROJ_DIR/output

rm -rf $PROJ_DIR/model
mkdir -p $PROJ_DIR/model

# -----
# Run the notebook, and store a copy into the output dir
# -----

INPUT_FILE_PATH="$PROJ_DIR/notebook/$NOTEBOOK_FILE"
INPUT_FILE_DIR="$(dirname "$INPUT_FILE_PATH")"
OUTPUT_FILE_PATH="$PROJ_DIR/output/$NOTEBOOK_FILE"
OUTPUT_FILE_DIR="$(dirname "$OUTPUT_FILE_PATH")"
mkdir -p "$OUTPUT_FILE_DIR"

echo "# [NOTE] Running notebook: $NOTEBOOK_FILE"
cd "$INPUT_FILE_DIR"
papermill \
    -k python3 --log-output \
    "$INPUT_FILE_PATH" "$OUTPUT_FILE_PATH" 
