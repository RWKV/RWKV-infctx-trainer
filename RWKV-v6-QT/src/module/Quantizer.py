# Dependencies
from .CoreDependencies import *

# Quantized training support
import bitsandbytes as bnb
import torch
from torch.nn import functional as F

# Quantize the given module, for training purpose
# return both the quentized data, and state
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
        self.qData = self.qData.to(self.device)
        self.optState = self.optState.to(self.device)

        assert self.qData is not None, "Quantized data is not initialized"
        assert self.optState is not None, "Quantized optimizer state is not initialized"


    # Get the dequentized module, for training purpose
    def dequantize_weights(self):
        
        assert self.qData is not None, "Quantized data is not initialized"
        assert self.optState is not None, "Quantized optimizer state is not initialized"

        # Rebuild the tensor
        return dequantize_training_module(self.qData, self.optState, self.qType).to(self.device).to(torch.bfloat16)

    # Handle the module operations, with dequantized computation
    def __call__(self, *args, **kwargs):
        return F.linear(*args, self.dequantize_weights(), self.bias)

    # # Forward operation, with dequantized computation
    # def forward(self, *args, **kwargs):
    #     return self.dequantize().forward(*args, **kwargs)

    # Forward operations, for linear modules
    def forward(self, x):
        return F.linear(x, self.dequantize_weights(), self.bias)
