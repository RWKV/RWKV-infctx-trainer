import sys, os, yaml

# # Lets configure PYTORCH_CUDA_ALLOC_CONF to use `backend:cudaMallocAsync` 
# # unless backend is already configured, to optimize memory allocations.
# #
# # This has to be done before any torch related modules are imported
# #
# # See: https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
# #
# # UPDATE: This is found to have issues with deepspeed 3, and is disabled for now
# # ---
# PYTORCH_CUDA_ALLOC_CONF = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', "")
# if len(PYTORCH_CUDA_ALLOC_CONF) > 0 and PYTORCH_CUDA_ALLOC_CONF.find("backend") == -1:
#     PYTORCH_CUDA_ALLOC_CONF = "backend:cudaMallocAsync," + PYTORCH_CUDA_ALLOC_CONF
# elif len(PYTORCH_CUDA_ALLOC_CONF) == 0:
#     PYTORCH_CUDA_ALLOC_CONF = "backend:cudaMallocAsync"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF
# print(f"[RWKV.lightning_trainer.py] Running with PYTORCH_CUDA_ALLOC_CONF={PYTORCH_CUDA_ALLOC_CONF}")

# Global RWKV flags
global RWKV_JIT_ON, RWKV_TORCH_COMPILE, RWKV_NO_CUDA

# Get the JIT / torch compile option flags default specific for lightning training mode
# This enables torch compile by default
RWKV_JIT_ON         = os.getenv("RWKV_JIT_ON", "1").lower() in ("1", "true", "yes")
RWKV_TORCH_COMPILE  = os.getenv("RWKV_TORCH_COMPILE", f"0").lower() in ("1", "true", "yes")

# Disable CUDA for RWKV
RWKV_NO_CUDA        = os.getenv("RWKV_NO_CUDA", "1").lower() in ("1", "true", "yes")

# Set back to the env vars (so that the rest of the code can use it)
os.environ["RWKV_JIT_ON"] = str(RWKV_JIT_ON)
os.environ["RWKV_TORCH_COMPILE"] = str(RWKV_TORCH_COMPILE)
os.environ["RWKV_NO_CUDA"] = str(RWKV_NO_CUDA)

# Parse the global args, we have to do this manually
# because argparse do not support --trainer.strategy
# ---
CLI_ARGS_MAP = {}
current_key = None
for arg in sys.argv[1:]:
    if arg.startswith('-'):
        if '=' in arg:
            key, value = arg.split('=', 1)
            CLI_ARGS_MAP[key] = value
            current_key = None
        else:
            current_key = arg
    elif current_key:
        CLI_ARGS_MAP[current_key] = arg
        current_key = None     

# Check for the config file
CONFIG_FILE_PATH = None
if "-c" in CLI_ARGS_MAP:
    CONFIG_FILE_PATH = CLI_ARGS_MAP["-c"]
elif "--config" in CLI_ARGS_MAP:
    CONFIG_FILE_PATH = CLI_ARGS_MAP["--config"]
assert CONFIG_FILE_PATH is not None, "Config file is not specified (use --config <config.yaml>, or -c <config.yaml>)"
assert os.path.exists(CONFIG_FILE_PATH), "Config file does not exist: "+CONFIG_FILE_PATH

# Read the config file, for the trainer.strategy
LIGHTNING_CONFIG = None
with open(CONFIG_FILE_PATH, 'r') as f:
    LIGHTNING_CONFIG = yaml.safe_load(f)
assert LIGHTNING_CONFIG is not None, "Failed to load config file: "+CONFIG_FILE_PATH

# We need to detect if deepspeed 3 is being used, either as defined
# by the config file, or by the command line arguments. 
# Before loading the respective RWKV modules, with the required env vars
# ---
def disable_jit_if_deepspeed_3():
    # Get the configured deepspeed strat
    assumed_deepspeed_strategy = LIGHTNING_CONFIG.get("trainer", {}).get("strategy", "")

    # Check if there is a trainer.strategy in the command line arguments
    if "--trainer.strategy" in CLI_ARGS_MAP:
        assumed_deepspeed_strategy = CLI_ARGS_MAP["--trainer.strategy"]

    # Finally lets check if the assumed_deepspeed_strategy contains the text "deepspeed_stage_3"
    # And disable JIT, as its not supported by deepspeed_stage_3
    if "deepspeed_stage_3" in assumed_deepspeed_strategy:
        print(f"[RWKV.lightning_trainer.py] Detected {assumed_deepspeed_strategy}, disabling JIT using RWKV_JIT_ON=0")
        os.environ["RWKV_JIT_ON"] = "0"
        os.environ["RWKV_TORCH_COMPILE"] = "0"
        

# Perform the deepspeed 3 check
disable_jit_if_deepspeed_3()

#
# Handle --auto-resume-ckpt-dir and --auto-resume-ckpt-mode parameters
# by modifyiing the argv list if needed.
#
# --auto-resume-ckpt-dir  : is used to specify a directory where the various checkpoints will be saved to
#                           including the last.ckpt checkpoint directory.
# --auto-resume-ckpt-mode : supports either "last" or "2nd-last", with "2nd-last" being default
#
# ---
#
# ckpt-mode, "2nd-last" uses the 2nd last checkpoint, instead of the truely latest checkpoint. 
# And is meant to be used with the checkpoint ooption `save_last: true`
#
# this is used to mitigate potential checkpoint file corruption issues that occur when the training
# is forcefully interupted while actively saving the checkpoint. The following are the 3 major scenerios
#
# 1. Training interrupted, between 2 checkpoint. Since last.ckpt, and the latest non "last.ckpt" checkpoint are saved correctly.
#    Training resumes correctly from the checkpoint
#
# 2. Training interrupted, while saving the new checkpoint (not the last.ckpt). As this new checkpoint is newer then last.ckpt
#    the uncorrupted last.ckpt is used instead
#
# 3. Training interrupted, while saving the last.ckpt, after saving the latest checkpoint. While the last.ckpt is corrupted
#    no issue occur when using the latest non last.ckpt checkpoint.
#
# If there is no last.ckpt, the 2nd latest checkpoint is used instead, followed by the latest checkpoint
#
# Technically there is a potential error still when crashing while saving the first checkpoint, but I consider that an acceptable compromise
# as not too much "work" would be lost in the process
#
# ---

# Get all the args used
PYTORCH_CLI_ARGV = sys.argv[1:]

# (utility function) Safely get mtime if the file exists, else -1
def safe_getmtime(path):
    if os.path.exists(path):
        return os.path.getmtime(path)
    else:
        return -1

# Handle the --auto-resume-ckpt-dir and --auto-resume-ckpt-mode parameters
def process_auto_resume_ckpt():
    # Get the --auto-resume-ckpt-dir and --auto-resume-ckpt-mode values
    auto_resume_ckpt_dir = None
    auto_resume_ckpt_mode = "2nd-last"
    if "--auto-resume-ckpt-dir" in CLI_ARGS_MAP:
        auto_resume_ckpt_dir = CLI_ARGS_MAP["--auto-resume-ckpt-dir"]
    if "--auto-resume-ckpt-mode" in CLI_ARGS_MAP:
        auto_resume_ckpt_mode = CLI_ARGS_MAP["--auto-resume-ckpt-mode"]

    # Check if the --auto-resume-ckpt-dir is set, return and exit if its not
    if auto_resume_ckpt_dir is None:
        return

    # Abort if --checkpoint is configured
    if "--checkpoint" in CLI_ARGS_MAP:
        print(f"[RWKV.lightning_trainer.py][warning] --auto-resume-ckpt-dir is not compatible with --checkpoint (ignoring --auto-resume-ckpt-dir)")
        return

    # Handle auto_resume_ckpt_dir if its true or auto
    if auto_resume_ckpt_dir.lower() == "true" or auto_resume_ckpt_dir.lower() == "auto":
        print(f"[RWKV.lightning_trainer.py] Extracting checkpoint dir from config, for --auto-resume-ckpt-dir={auto_resume_ckpt_dir}")

        # Handle the auto resume overwrite, via CLI
        if CLI_ARGS_MAP["--trainer.callbacks.init_args.dirpath"] is not None:
            auto_resume_ckpt_dir = CLI_ARGS_MAP["--trainer.callbacks.init_args.dirpath"]
        else:
            # Try to get as an object, then an object in an array
            auto_resume_ckpt_dir = LIGHTNING_CONFIG.get("trainer", {}).get("callbacks", {}).get("init_args", {}).get("dirpath", None)
            if auto_resume_ckpt_dir is None:
                auto_resume_ckpt_dir = LIGHTNING_CONFIG.get("trainer", {}).get("callbacks", [{}])[0].get("init_args", {}).get("dirpath", None)

        # Safety check on the dir
        assert auto_resume_ckpt_dir is not None, "Failed to extract checkpoint dir from config, for --auto-resume-ckpt-dir=True"
        
    # Log the setting flag
    print(f"[RWKV.lightning_trainer.py] Enabling --auto-resume-ckpt-dir={auto_resume_ckpt_dir} --auto-resume-ckpt-mode={auto_resume_ckpt_mode}")

    # Check if the --auto-resume-ckpt-dir exists, if it does not initialize it and return
    # In some rare cases, path can "not exists" but exists when "created" 	
    if not os.path.exists(auto_resume_ckpt_dir):
        try:
            os.makedirs(auto_resume_ckpt_dir)
            print(f"[RWKV.lightning_trainer.py] Created '{auto_resume_ckpt_dir}' directory (did not exist previously)")
        except FileExistsError:
            print(f"[RWKV.lightning_trainer.py] Directory '{auto_resume_ckpt_dir}' already exists.")
        return


    #  if not os.path.exists(auto_resume_ckpt_dir):
    #      os.makedirs(auto_resume_ckpt_dir)
    #      print(f"[RWKV.lightning_trainer.py] Created '{auto_resume_ckpt_dir}' directory (did not exist previously)")
    #      return
    
    # Get the list of directories in the --auto-resume-ckpt-dir
    auto_resume_ckpt_dir_list = os.listdir(auto_resume_ckpt_dir)

    # Filter only for directories that end with .ckpt
    auto_resume_ckpt_dir_list = [x for x in auto_resume_ckpt_dir_list if x.endswith(".ckpt")]

    # No directories, return
    if len(auto_resume_ckpt_dir_list) == 0:
        print(f"[RWKV.lightning_trainer.py] No checkpoints found in '{auto_resume_ckpt_dir}', starting from scratch")
        return

    # Get the last 2 directories, by their 'latest' file (inside the dir) if said file exists
    auto_resume_ckpt_dir_list.sort(key=lambda x: safe_getmtime(os.path.join(auto_resume_ckpt_dir, x, 'latest')), reverse=True)

    # Lets figure out the checkpoint to use
    checkpoint_to_use = None

    # Checkpoint count
    checkpoint_count = len(auto_resume_ckpt_dir_list)

    # If there is only 1 checkpoint, use it
    if checkpoint_count == 1:
        checkpoint_to_use = auto_resume_ckpt_dir_list[0]
    else:
        # There are at least 2 checkpoints, lets figure out which one to use
        # Check if the mode is "last"
        if auto_resume_ckpt_mode == "last":
            # Use the first checkpoint
            checkpoint_to_use = auto_resume_ckpt_dir_list[0]
        elif auto_resume_ckpt_mode == "2nd-last":
            # Use the second checkpoint
            checkpoint_to_use = auto_resume_ckpt_dir_list[1]
        else:
            # Use the second checkpoint as default behaviour, log a warning for invalid mode
            checkpoint_to_use = auto_resume_ckpt_dir_list[1]
            print(f"[RWKV.lightning_trainer.py][warning] Invalid --auto-resume-ckpt-mode={auto_resume_ckpt_mode}, using '2nd-last' instead")
    
    # Log the chekpoint to use
    print(f"[RWKV.lightning_trainer.py] Found {checkpoint_count} checkpoints in '{auto_resume_ckpt_dir}', using '{checkpoint_to_use}'")

    # Lets append the --checkpoint argument to the PYTORCH_CLI_ARGV
    PYTORCH_CLI_ARGV.append("--ckpt_path")
    PYTORCH_CLI_ARGV.append(os.path.join(auto_resume_ckpt_dir, checkpoint_to_use))

# Process the args
process_auto_resume_ckpt()

# Add a quick warning that pytorch lightning timing estimates are COMPLETELY INACCURATE
# when resuming from a checkpoint
if "--ckpt_path" in PYTORCH_CLI_ARGV:
    print("[RWKV.lightning_trainer.py][warning] Pytorch Lightning timing estimates can be very inaccurate when resuming from a checkpoint, due to the way it calculates the time left. See: https://github.com/Lightning-AI/lightning/issues/18220")

# Utility for remove an argument from the PYTORCH_CLI_ARGV list
def remove_arg(argList, argCommand):
    # Remove the --argCommand=argValue varient
    # we do this by prefix searchig `argCommand=`
    for arg in argList:
        if arg.startswith(argCommand+"="):
            argList.remove(arg)
    
    # Remove the --argCommand argValue varient
    # we do this by searchig for `argCommand`, and the next arg
    for arg in argList:
        if arg == argCommand:
            argIndex = argList.index(arg)
            if argIndex < len(argList)-1:
                argList.remove(argList[argIndex+1])
            argList.remove(arg)
    
    # Return the modified argList
    return argList

# Remove the --auto-resume-ckpt-dir and --auto-resume-ckpt-offset
PYTORCH_CLI_ARGV = remove_arg(PYTORCH_CLI_ARGV, "--auto-resume-ckpt-dir")
PYTORCH_CLI_ARGV = remove_arg(PYTORCH_CLI_ARGV, "--auto-resume-ckpt-mode")

# ---

from lightning.pytorch.cli import LightningCLI
from src.model import RWKV
from src.data import RWKVDataModule
from src.trainer import RWKVLightningTrainer

def cli_main():
    LightningCLI(
        RWKV, RWKVDataModule, 
        save_config_kwargs={"overwrite": True},
        trainer_class=RWKVLightningTrainer,

        # Overwrite several trainer default configs
        trainer_defaults={
            "accelerator": "gpu",
            "precision": "bf16-mixed",
            "strategy": "deepspeed_stage_2_offload",

            # num_sanity_val_steps is disabled, as they seem
            # to hang during initial sanity check for unknown reasons
            # for larger model sizes randomly on multi-gpus
            "num_sanity_val_steps": 0,

            # Disable default distributed sampler, 
            # so that we can control shuffle logic on our side instead
            "use_distributed_sampler": False
        },
        seed_everything_default=True,
        args=PYTORCH_CLI_ARGV
    )

if __name__ == "__main__":
    cli_main()
