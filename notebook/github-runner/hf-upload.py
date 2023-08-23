# Get the Hugging Face Hub API
from huggingface_hub import HfApi
api = HfApi()

# Get the repo path from the script first arg
import sys
REPO_PATH = sys.argv[1]
REPO_SUBDIR = sys.argv[2]

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
    repo_id=REPO_PATH,
    path_in_repo=REPO_SUBDIR,
    repo_type="model",
    multi_commits=True,
    allow_patterns=["*.pth"],
    commit_message=f"[GHA] {REPO_SUBDIR}.ipynb result models"
)

# Upload the ipynb files
api.upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=REPO_PATH,
    path_in_repo=REPO_SUBDIR,
    repo_type="model",
    multi_commits=True,
    allow_patterns=["*.ipynb"],
    commit_message=f"[GHA] {REPO_SUBDIR}.ipynb result notebooks"
)