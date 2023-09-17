import modal, os
from modal import gpu, Mount, Stub, Image, Volume, Secret

# -----------------------
# Input vars handling
# -----------------------

# -----------------------
# Dir path detection
# -----------------------

# Runners Dir
RUNNER_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.dirname(RUNNER_DIR)

# Lets include the project
TARGET_DIR_NAME="RWKV-v5"

# Target file path
NOTEBOOK_SUBPATH = "notebook/experiment/infctx-math-and-name"
TARGET_FILE_PATH = os.path.join(PROJ_DIR, "notebook")

# Mountied training dir
TRAINING_DIR = os.path.join(PROJ_DIR, TARGET_DIR_NAME)

# -----------------------
# Modal download setup
# -----------------------

# Setup the DockerImage
DOCKERFILE_IMAGE = Image.from_dockerfile(os.path.join(PROJ_DIR, "runners/modal-docker/Dockerfile"))
DOCKER_IMAGE=DOCKERFILE_IMAGE.pip_install("jupyter").apt_install("curl").run_commands("curl https://sh.rustup.rs -sSf | bash -s -- -y").run_commands(". $HOME/.cargo/env && cargo install bore-cli")

# Setup the RWKV infctx trainer
stub = modal.Stub(
    "RWKV-infctx-ModalTrainer",
    image=DOCKER_IMAGE
)
stub.workspace_volume = Volume.persisted("RWKV-infctx-workspace")

# Start the downloads
@stub.function(
    volumes={
        "/workspace": stub.workspace_volume
    },
    # RAM usage and timeout
    memory=1024 * 100,
    timeout=3600 * 4,
)
def download(model_name: str = None):
    return ""



# -----------------------
# Input vars handling
# -----------------------


@stub.function()
def square(x):
    print("This code is running on a remote worker!")

    # Get the list of files
    files = modal.get_files()
    return files

@stub.local_entrypoint()
def main():
    print("Running the download sequence", download.remote())
    # print("the square is", square.remote(42))