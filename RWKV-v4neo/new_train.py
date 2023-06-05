from lightning.pytorch.cli import LightningCLI

from src.model import RWKV
from src.data import get_data_module


def cli_main():
    LightningCLI(RWKV, get_data_module, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
