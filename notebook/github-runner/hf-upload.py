# Get the Hugging Face Hub API
from huggingface_hub import HfApi
api = HfApi()

# Get the NOTEBOOK_FILE from the script first arg
import sys
NOTEBOOK_FILE = sys.argv[2]

# Compute the notebook subdir from the notebook parent
NOTEBOOK_SUBDIR = os.path.dirname(NOTEBOOK_FILE)

# Get the repo ID from the second arg
# if its not set/blank, defaults to rwkv-x-dev/rwkv-x-playground
REPO_URI = sys.argv[1] if len(sys.argv) > 1 else ""
if REPO_URI == "":
    REPO_URI = "rwkv-x-dev/rwkv-x-playground"

# Get the current script dir
import os

RUNNER_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.dirname(RUNNER_SCRIPT_DIR)
PROJ_DIR = os.path.dirname(NOTEBOOK_DIR)
MODEL_DIR = os.path.join(PROJ_DIR, "model")
OUTPUT_DIR = os.path.join(PROJ_DIR, "output")

# Upload the models
api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=REPO_URI,
    path_in_repo=NOTEBOOK_SUBDIR,
    repo_type="model",
    multi_commits=True,
    allow_patterns=["*.pth"],
    commit_message=f"[GHA] {NOTEBOOK_FILE} result models"
)

# Upload the ipynb files
api.upload_folder(
    folder_path=f"{OUTPUT_DIR}/${NOTEBOOK_SUBDIR}",
    repo_id=REPO_URI,
    path_in_repo=NOTEBOOK_SUBDIR,
    repo_type="model",
    multi_commits=True,
    allow_patterns=["*.ipynb"],
    commit_message=f"[GHA] {NOTEBOOK_FILE} result notebooks"
)
