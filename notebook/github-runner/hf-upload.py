import os
import sys
from huggingface_hub import HfApi

# Get the Hugging Face Hub API
api = HfApi()

# Get the NOTEBOOK_FILE from the script first arg
NOTEBOOK_FILE = sys.argv[1]

# Compute the notebook subdir from the notebook parent
NOTEBOOK_SUBDIR = os.path.dirname(NOTEBOOK_FILE)

# Get the repo ID from the HF_REPO_SYNC env var
REPO_URI = os.getenv("HF_REPO_SYNC", "rwkv-x-dev/rwkv-x-playground")

# Directory paths
RUNNER_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.dirname(RUNNER_SCRIPT_DIR)
PROJ_DIR = os.path.dirname(NOTEBOOK_DIR)
MODEL_DIR = os.path.join(PROJ_DIR, "model")
OUTPUT_DIR = os.path.join(PROJ_DIR, "output")

# Generate the URL where all the items will be uploaded
hf_url = f"https://huggingface.co/{REPO_URI}/tree/main/{NOTEBOOK_SUBDIR}"
print(f"# ------------------------------------")
print(f"# Uploading to: {hf_url}")
print(f"# ------------------------------------")

# Upload the models
print("# Uploading the models ... ")
try:
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_URI,
        path_in_repo=NOTEBOOK_SUBDIR,
        repo_type="model",
        multi_commits=True,
        allow_patterns=["*.pth"],
        commit_message=f"[GHA] {NOTEBOOK_FILE} result models"
    )
except Exception as e:
    eStr = str(e)
    if "must have at least 1 commit" in eStr:
        print("# Skipping model upload due to error ... ")
        print(e)
    else:
        raise e
    
# Upload the ipynb files
print("# Uploading the notebooks / output files ... ")
api.upload_folder(
    folder_path=f"{OUTPUT_DIR}/{NOTEBOOK_SUBDIR}",
    repo_id=REPO_URI,
    path_in_repo=NOTEBOOK_SUBDIR,
    repo_type="model",
    multi_commits=True,
    allow_patterns=["*.ipynb", "*.csv", "*/*.csv"],
    commit_message=f"[GHA] {NOTEBOOK_FILE} result notebooks"
)

print(f"# ------------------------------------")
print(f"# Uploaded to: {hf_url}")
print(f"# ------------------------------------")
