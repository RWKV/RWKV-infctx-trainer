#!/bin/bash

# Check if HUGGING_FACE_HUB_TOKEN & WANDB_API_KEY is set
if [[ -z "${HUGGING_FACE_HUB_TOKEN}" ]]; then
    echo "[ERROR]: HUGGING_FACE_HUB_TOKEN is not set"
    exit 1
fi
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "[ERROR]: WANDB_API_KEY is not set"
    exit 1
fi

# Get the notebook script from the first arg
NOTEBOOK_FILE=$1

# Get the current script directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NOTEBOOK_DIR="$(dirname "$SCRIPT_DIR")"
PROJ_DIR="$(dirname "$NOTEBOOK_DIR")"

# Log the proj dir
echo "#"
echo "# Starting github notebook runner"
echo "#"
echo "# PROJ_DIR: $PROJ_DIR"
echo "# NOTEBOOK_DIR: $NOTEBOOK_DIR"
echo "# NOTEBOOK_FILE: $NOTEBOOK_FILE"
echo "#"

# Check if the notebook file exists, in the notebook directory
if [[ ! -f "$NOTEBOOK_DIR/$NOTEBOOK_FILE" ]]; then
    echo "[ERROR]: Notebook file does not exist ($NOTEBOOK_FILE)"
    exit 1
fi

# Setup the common folders
mkdir -p $PROJ_DIR/checkpoint
mkdir -p $PROJ_DIR/datapath
mkdir -p $PROJ_DIR/output
mkdir -p $PROJ_DIR/model