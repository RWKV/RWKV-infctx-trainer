from lightning.pytorch import Trainer
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
import lightning as Lightning

# We extend the native pytorch lightning trainer to add the following
#
# - local "fabric" support, as the trainer object is one of the few
#   objects that is available to all the processess
class RWKVLightningTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fabric_instance = None

    def getFabric(self):
        if self._fabric_instance is not None:
            return self._fabric_instance
        
        strat = self.strategy
        if strat is None:
            raise ValueError("Trainer strategy config is missing")

        # Map the pytorch lightning strat to fabric strat string
        stratStr = "auto"
        if isinstance(strat, DeepSpeedStrategy):
            stratStr = "deepspeed"
        
        self._fabric_instance = Lightning.Fabric(
            accelerator=self.accelerator,
            devices=self.num_devices,
            num_nodes=self.num_nodes,
            strategy=stratStr
        )
        return self._fabric_instance
