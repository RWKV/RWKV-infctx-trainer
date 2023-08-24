import os
import sys
# from huggingface_hub import HfApi
from huggingface_hub import snapshot_download

# # Get the Hugging Face Hub API
# api = HfApi()

# Get the NOTEBOOK_FILE from the script first arg
NOTEBOOK_FILE = sys.argv[1]

# Compute the notebook subdir from the notebook parent
NOTEBOOK_SUBDIR = os.path.dirname(NOTEBOOK_FILE)

# Get the repo ID from the HF_REPO_SYNC env var
REPO_URI = os.getenv("HF_REPO_SYNC", "rwkv-x-dev/rwkv-x-playground")

RUNNER_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.dirname(RUNNER_SCRIPT_DIR)
PROJ_DIR = os.path.dirname(NOTEBOOK_DIR)
MODEL_DIR = os.path.join(PROJ_DIR, "model")
OUTPUT_DIR = os.path.join(PROJ_DIR, "output")

# Temporary HF download directory within the project (due to API limitation)
HF_DOWNLOAD_DIR = os.path.join(PROJ_DIR, ".hf-download")
os.makedirs(HF_DOWNLOAD_DIR, exist_ok=True)

# The HF cache directory for models
HF_HOME=os.getenv("HF_HOME", "")
if HF_HOME == "":
    raise Exception("HF_HOME is not set")
# Setup the model cache
HF_MODEL_CACHE = os.path.join(HF_HOME, f"model_cache/${REPO_URI}/${NOTEBOOK_SUBDIR}")
os.makedirs(HF_MODEL_CACHE, exist_ok=True)

# Generate the URL where all the items will be uploaded
hf_url = f"https://huggingface.co/{REPO_URI}/tree/main/{NOTEBOOK_SUBDIR}"

# Check if the URL is a 404 (meaning no files exists)
# If it doesn't exists - skip the download process
import requests
r = requests.get(hf_url)
if r.status_code == 404:
    print(f"# ------------------------------------")
    print(f"# [Finished] No files to download from: {hf_url}")
    print(f"# ------------------------------------")
    exit(0)

# Start the downloading process
print(f"# ------------------------------------")
print(f"# Downloading from: {hf_url}")
print(f"# ------------------------------------")

# Download the existing models
# due to the limtation of the API, we cannot download from a specific folder
# and instead use the allow_patterns to filter the files
snapshot_download(
    repo_id=REPO_URI,
    local_dir=HF_DOWNLOAD_DIR,
    local_dir_use_symlinks=False,
    allow_patterns=[f"{NOTEBOOK_SUBDIR}/*.pth"],
    cache_dir=HF_MODEL_CACHE
)

print(f"# ------------------------------------")
print(f"# Syncing to model directory: {MODEL_DIR}")
print(f"# ------------------------------------")

# Move the downloaded files the NOTEBOOK_SUBDIR
# within the HF_DOWNLOAD_DIR to the MODEL_DIR
import shutil

# Create the MODEL_DIR if it doesn't exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Move the files from the NOTEBOOK_SUBDIR inside HF_DOWNLOAD_DIR to the MODEL_DIR
for file in os.listdir(os.path.join(HF_DOWNLOAD_DIR, NOTEBOOK_SUBDIR)):
    shutil.move(
        os.path.join(HF_DOWNLOAD_DIR, NOTEBOOK_SUBDIR, file),
        os.path.join(MODEL_DIR, file)
    )

