from lightning.pytorch import Trainer
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
import lightning as Lightning
import torch
import math
import wandb

global RWKV_JIT_ON, RWKV_TORCH_COMPILE, RWKV_NO_CUDA

# We extend the native pytorch lightning trainer to add the following
#
# - local "fabric" support, as the trainer object is one of the few
#   objects that is available to all the processess
# - target_batch_size, which automatically computes the accumulate_grad_batches
class RWKVLightningTrainer(Trainer):
    def __init__(
            self,
            *args, 
            # Replaces the accumulate_grad_batches, if set
            # automatically compute the accumulate_grad_batches
            #
            # According to the microbatch_size, num_nodes, 
            # and num_devices configured
            target_batch_size=-1,
            # Microbatch sizing, to be used with
            # each training step per GPU. 
            # 
            # This is the same as pytorch dataset batch size.
            microbatch_size=1,
            # Handle the rest of args, as per normal
            **kwargs,
        ):
        # Use the global RWKV_NO_CUDA flag
        global RWKV_NO_CUDA

        # trainer_config args (used for wndb logging)
        trainer_config = dict(kwargs)

        # target batch size logging
        target_batch_size_log_msg = ""

        # Compute the microbatch_size
        self.microbatch_size = microbatch_size
        assert microbatch_size > 0, "microbatch_size must be greater than 0"

        # Compute the accumulate_grad_batches, using the target_batch_size
        self.target_batch_size = target_batch_size
        if target_batch_size > 0:

            # Check if the accumulate_grad_batches is already set
            # (note that it seems that pytorch lightning defaults to 1)
            if "accumulate_grad_batches" in kwargs and kwargs["accumulate_grad_batches"] > 1:
                raise ValueError(f"Cannot set both 'target_batch_size' ({target_batch_size}) and 'accumulate_grad_batches' ({kwargs['accumulate_grad_batches']}))")

            # Extract the num_nodes and devices
            num_nodes = kwargs.get("num_nodes", 1)
            devices = kwargs.get("devices", "auto")
            
            # Compute the number of devices
            if devices == "auto":
                num_devices = torch.cuda.device_count()
            elif isinstance(devices, int):
                num_devices = devices
            elif isinstance(devices, list):
                num_devices = len(devices)
            else:
                raise ValueError(f"Unsupported devices config '{devices}', unable to compute device count for 'target_batch_size'")
            
            # Compute the accumulate_grad_batches
            accumulate_grad_batches = max( 1, math.floor(target_batch_size / (num_nodes * num_devices * microbatch_size)) )
            kwargs["accumulate_grad_batches"] = accumulate_grad_batches
            effective_batch_size = accumulate_grad_batches * num_nodes * num_devices * microbatch_size

            # Log the applied accumulate_grad_batches
            trainer_config["__accumulate_grad_batches"] = accumulate_grad_batches
            trainer_config["__effective_batch_size"] = effective_batch_size

            # Log the computed accumulate_grad_batches
            # this is done after _init_ so we can confirm local rank
            target_batch_size_log_msg = ("\n"+
                f"\n[RWKV.Trainer] Applying 'target_batch_size' with the following:\n"+
                f"   - target_batch_size:       {target_batch_size}\n"+
                f"   - num_nodes:               {num_nodes}\n"+
                f"   - num_devices:             {num_devices}\n"+
                f"   - microbatch_size:         {microbatch_size}\n"+
                f"   - accumulate_grad_batches: {accumulate_grad_batches}\n"
                f"   - effective_batch_size:    {effective_batch_size}\n")
            
            # Disable CUDA, if the device type is NOT auto / cuda
            # or if no CUDA devices was detected
            if devices != "auto" and devices != "cuda":
                RWKV_NO_CUDA = True
            if num_devices <= 0:
                RWKV_NO_CUDA = True

        # Update WANDB config
        # ---
        if wandb.run is not None:
            trainer_config["target_batch_size"] = target_batch_size
            del trainer_config["logger"]
            del trainer_config["callbacks"]
            wandb.config.update({
                "trainer": trainer_config
            })

        # Call the parent constructor
        super().__init__(*args, **kwargs)
        self._fabric_instance = None

        # Log the target_batch_size_log_msg
        # if local rank is 0
        if target_batch_size_log_msg != "" and self.local_rank == 0:
            print(target_batch_size_log_msg)
    
    # Fabric instance, useful for coordinating between processes
    # when `self.trainer.strategy.reduce` is not possible
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
