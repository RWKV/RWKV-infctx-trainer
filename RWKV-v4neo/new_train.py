from lightning.pytorch.cli import LightningCLI
import sys, os, yaml

# We need to detect if deepspeed 3 is being used, either as defined
# by the config file, or by the command line arguments. 
# Before loading the respective RWKV modules, with the required env vars
# ---
def disable_jit_if_deepspeed_3():
    assumed_deepspeed_strategy=""

    # Parse the global args, we have to do this manually
    # because argparse do not support --trainer.strategy
    cli_args = {}
    current_key = None
    for arg in sys.argv[1:]:
        if arg.startswith('-'):
            if '=' in arg:
                key, value = arg.split('=', 1)
                cli_args[key] = value
                current_key = None
            else:
                current_key = arg
        elif current_key:
            cli_args[current_key] = arg
            current_key = None     

    # Check for the config file
    config_file = None
    if "-c" in cli_args:
        config_file = cli_args["-c"]
    elif "--config" in cli_args:
        config_file = cli_args["--config"]
    assert config_file is not None, "Config file is not specified (use --config <config.yaml>, or -c <config.yaml>)"
    assert os.path.exists(config_file), "Config file does not exist: "+config_file

    # Read the config file, for the trainer.strategy
    with open(config_file, 'r') as f:
        lightning_config = yaml.safe_load(f)
        assumed_deepspeed_strategy = lightning_config.get("trainer", {}).get("strategy", "")

    # Check if there is a trainer.strategy in the command line arguments
    if "--trainer.strategy" in cli_args:
        assumed_deepspeed_strategy = cli_args["--trainer.strategy"]

    # Finally lets check if the assumed_deepspeed_strategy contains the text "deepspeed_stage_3"
    # And disable JIT, as its not supported by deepspeed_stage_3
    if "deepspeed_stage_3" in assumed_deepspeed_strategy:
        print(f"[RWKV.new_train.py] Detected {assumed_deepspeed_strategy}, disabling JIT using RWKV_JIT_ON=0")
        os.environ["RWKV_JIT_ON"] = "0"

# Perform the check
disable_jit_if_deepspeed_3()

# ---

# Load the respective RWKV module
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
            "precision": "bf16",
            "strategy": "deepspeed_stage_2_offload",
        },
        seed_everything_default=True
    )

if __name__ == "__main__":
    cli_main()
