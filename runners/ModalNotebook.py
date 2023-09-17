import modal, os
import subprocess, time
from modal import gpu, Mount, Stub, Image, Volume, Secret

# -----------------------
# Input vars handling
# -----------------------

JUPYTER_TOKEN = "P@ssw0rd!"

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
# Juypyter setup
# -----------------------

# Setup the DockerImage
DOCKERFILE_IMAGE = Image.from_dockerfile(os.path.join(PROJ_DIR, "runners/modal-docker/Dockerfile"))
DOCKER_IMAGE=DOCKERFILE_IMAGE.pip_install("jupyter").apt_install("curl").run_commands("curl https://sh.rustup.rs -sSf | bash -s -- -y").run_commands(". $HOME/.cargo/env && cargo install bore-cli")

# Setup the RWKV infctx trainer
stub = modal.Stub(
    "RWKV-infctx-ModalNotebook",
    image=DOCKER_IMAGE
)
stub.nfs_volume = modal.NetworkFileSystem.persisted("RWKV-infctx-nfs-x")
stub.workspace_volume = Volume.persisted("RWKV-infctx-workspace")


@stub.function(
    concurrency_limit=1,
    network_file_systems={
        "/nfs": stub.nfs_volume
    },
    volumes={
        "/workspace": stub.workspace_volume
    },
    # 8 hours
    timeout=3600 * 8,
    # 8 * A100
    gpu=gpu.A100(count=8, memory=40),
)
def run_jupyter(timeout: int):
    jupyter_process = subprocess.Popen(
        [
            "jupyter",
            "notebook",
            "--no-browser",
            "--allow-root",
            "--notebook-dir=/workspace",
            "--port=8888",
            "--NotebookApp.allow_origin='*'",
            "--NotebookApp.allow_remote_access=1",
        ],
        env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
    )
    bore_process = subprocess.Popen(
        ["/root/.cargo/bin/bore", "local", "8888", "--to", "bore.pub"],
    )

    try:
        end_time = time.time() + timeout
        while time.time() < end_time:
            time.sleep(5)
        print(f"Reached end of {timeout} second timeout period. Exiting...")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        bore_process.kill()
        jupyter_process.kill()


# -----------------------
# Input vars handling
# -----------------------

# @stub.function()
# def square(x):
#     print("This code is running on a remote worker!")

#     # Get the list of files
#     files = modal.get_files()
#     return files

@stub.local_entrypoint()
def main():
    # Run for 8 hours batches in a loop
    while True:
        try:
            print("Running the jupyter env", run_jupyter.remote(3600 * 8))
        except Exception as e:
            print("An error occurred while running the script: ", e)
    