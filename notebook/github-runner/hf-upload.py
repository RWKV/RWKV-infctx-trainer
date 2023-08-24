import os
import sys
from huggingface_hub import HfApi

# Get the Hugging Face Hub API
api = HfApi()

# Get the NOTEBOOK_FILE from the script first arg
NOTEBOOK_FILE = sys.argv[1]

# Compute the notebook subdir from the notebook parent
NOTEBOOK_SUBDIR = os.path.dirname(NOTEBOOK_FILE)

# Get the repo ID from the second arg
# if its not set/blank, defaults to rwkv-x-dev/rwkv-x-playground
REPO_URI = sys.argv[2] if len(sys.argv) > 1 else ""
if REPO_URI == "":
    REPO_URI = "rwkv-x-dev/rwkv-x-playground"

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

print("# Uploading the notebooks ... ")

# Upload the ipynb files
api.upload_folder(
    folder_path=f"{OUTPUT_DIR}/{NOTEBOOK_SUBDIR}",
    repo_id=REPO_URI,
    path_in_repo=NOTEBOOK_SUBDIR,
    repo_type="model",
    multi_commits=True,
    allow_patterns=["*.ipynb"],
    commit_message=f"[GHA] {NOTEBOOK_FILE} result notebooks"
)

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
    print("# Skipping model upload due to error ... ")
    print(e)
    
print(f"# ------------------------------------")
print(f"# Uploaded to: {hf_url}")
print(f"# ------------------------------------")