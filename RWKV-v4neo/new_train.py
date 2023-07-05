from lightning.pytorch.cli import LightningCLI

from src.model import RWKV
from src.data import RWKVDataModule
from src.trainer import RWKVLightningTrainer

def cli_main():
    LightningCLI(
        RWKV, RWKVDataModule, 
        save_config_kwargs={"overwrite": True},
        trainer_class=RWKVLightningTrainer
    )

if __name__ == "__main__":
    cli_main()
