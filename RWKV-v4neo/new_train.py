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
            "precision": "bf16",
            "strategy": "deepspeed_stage_2_offload",
        },
        seed_everything_default=True
    )

if __name__ == "__main__":
    cli_main()
