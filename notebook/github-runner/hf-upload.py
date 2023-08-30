import os
import sys
from huggingface_hub import HfApi

# This hopefully fix some issues with the HF API
import os
os.environ['CURL_CA_BUNDLE'] = ''

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

# ------------------------------------
# Uploading code, with work around some unstable HF stuff
# ------------------------------------

# List of errors, to throw at the end of all upload attempts
UPLOAD_ERRORS = []

# Fallback upload method, upload the files one by one
# with retry on failure (up to 3 attempts)
def upload_folder_fallback(folder_path, file_type="model"): 
    # Get the files to upload in the folder_path, including nested files
    file_list = []

    # Walk the folder and get all the files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    # and filter by supported file type
    file_list = [f for f in file_list if f.endswith(".pth") or f.endswith(".ipynb") or f.endswith(".csv")]

    # Log the fallback logic
    print(f"# Fallback {file_type} upload method ... ")

    # Upload the files one by one
    for file in file_list:
        print(f"# Uploading {file_type} file: {file} ... ")
        for i in range(3):
            try:
                api.upload_file(
                    path_or_fileobj=os.path.join(folder_path, file),
                    repo_id=REPO_URI,
                    path_in_repo=f"{NOTEBOOK_SUBDIR}/{file}",
                    repo_type="model",
                    commit_message=f"[GHA] {NOTEBOOK_FILE} result {file_type} (fallback single file upload)"
                )
            except Exception as e:
                print(f"# Error uploading {file_type} file: {file} ... ")
                print(e)
                if i == 2:
                    UPLOAD_ERRORS.append(e)
                continue
            break
    
    # Upload finished !
    print(f"# Upload of {file_type} files, via fallback finished !")

# Because multi-stage upload is "unstable", we try to upload the models with fallback handling
def upload_folder(folder_path, file_type="model"): 
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=REPO_URI,
            path_in_repo=NOTEBOOK_SUBDIR,
            repo_type="model",
            multi_commits=True,
            allow_patterns=["*.pth", "*.ipynb", "*.csv", "*/*.csv"],
            commit_message=f"[GHA] {NOTEBOOK_FILE} result {file_type}"
        )
    except Exception as e:
        eStr = str(e)
        if "must have at least 1 commit" in eStr:
            print("# Skipping {file_type} upload due to error ... ")
            print(e)
        else:
            upload_folder_fallback(folder_path, file_type)

# ------------------------------------
# Actual upload (and error handling)
# ------------------------------------

# Upload the models
print("# Uploading the models ... ")
upload_folder( MODEL_DIR, file_type="model" )
    
# Upload the ipynb files
print("# Uploading the notebooks / output files ... ")
upload_folder( f"{OUTPUT_DIR}/{NOTEBOOK_SUBDIR}", file_type="notebook & reports" )

print(f"# ------------------------------------")
print(f"# Uploaded finished to: {hf_url}")
print(f"# ------------------------------------")

# Print out the errors, if any
if len(UPLOAD_ERRORS) > 0:
    print("# ------------------------------------")
    print("# Upload errors:")
    print("# ------------------------------------")
    for e in UPLOAD_ERRORS:
        print(e)
    print("# ------------------------------------")
    print("# Upload errors, logged - hard exit")
    print("# ------------------------------------")
    sys.exit(1)
