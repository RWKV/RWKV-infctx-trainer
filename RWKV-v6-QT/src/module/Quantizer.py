# Dependencies
from .CoreDependencies import *

# Quantized training support
import bitsandbytes as bnb
import torch
from torch.nn import functional as F

# Quantize the given module, for training purpose
# return both the quentized data, and state
@torch.jit.ignore
def quantize_training_module(mod_weight, type="4bit"):
    # Get the device of the module
    device = mod_weight.device

    if type == "4bit":
        qData, opt = bnb.functional.quantize_4bit((mod_weight).to("cuda"))
        return qData, opt
    elif type == "nf4":
        qData, opt = bnb.functional.quantize_nf4((mod_weight).to("cuda"))
        return qData, opt
    elif type == "fp4":
        qData, opt = bnb.functional.quantize_fp4((mod_weight).to("cuda"))
        return qData, opt
    else:
        raise ValueError(f"Unknown quantization type {type}")

# Dequantize the given module, for training purpose
@torch.jit.ignore
def dequantize_training_module(mod_weight, mod_optimizer, type="4bit"):
    if type == "4bit":
        return bnb.functional.dequantize_4bit(mod_weight,quant_state=mod_optimizer)
    elif type == "nf4":
        return bnb.functional.dequantize_nf4(mod_weight,quant_state=mod_optimizer)
    elif type == "fp4":
        return bnb.functional.dequantize_fp4(mod_weight,quant_state=mod_optimizer)
    else:
        raise ValueError(f"Unknown quantization type {type}")

# Quantizer module wrapper, used for training
class QuantizedLinearModule(JITModClass):

    # Setup the quantized module representation, which can be used for training
    def __init__(self, module, quantize_type="4bit"):
        super().__init__()

        self.qType = quantize_type
        self.optState = None
        self.qData = None
        self.bias = module.bias
        self.device = module.weight.device

        # Quantize the initialized module
        self.qData, self.optState = quantize_training_module(module.weight, quantize_type)

        # assert self.qData is not None, "Quantized data is not initialized (pre move)"
        # assert self.optState is not None, "Quantized optimizer state is not initialized (pre move)"

        # self.qData = self.qData.to(self.device)

        #### DO NOT MOVE THIS - the optimizer state can be viewed as a pointer of sorts? that when moved breaks the quantized optimizer, etc
        #### I have no idea why this happens, its probably just a deisng artifact of the quantized training module
        # self.optState = self.optState.to(self.device)

        # assert self.qData is not None, f"Quantized data is not initialized - {self.device}"
        # assert self.optState is not None, f"Quantized optimizer state is not initialized = {self.device}"

        # # Holding variable, for the full dequantized weights
        # self._fullDataCache = None


    # Get the dequentized module, for training purpose
    def dequantize_weights(self, device):

        # # Return the dequantized weights if present
        # if self._fullDataCache is not None:
        #     return self._fullDataCache.to(device).to(torch.bfloat16)
        
        assert self.qData is not None, "Quantized data is not initialized"
        assert self.optState is not None, "Quantized optimizer state is not initialized"

        # # Rebuild the tensor, and cache it?
        # self._fullDataCache = dequantize_training_module(self.qData, self.optState, self.qType).to(device).to(torch.bfloat16)
        # return self._fullDataCache

        # Return and use it directly
        return dequantize_training_module(self.qData, self.optState, self.qType).to(device).to(torch.bfloat16)

    # # Clear the cache, if set
    # def cache_clear(self):
    #     self._fullDataCache = None

    # Forward operations, for linear modules
    def forward(self, x: torch.Tensor):
        return F.linear(x, self.dequantize_weights(x.device), self.bias)
