# Dependencies
from .CoreDependencies import *

# Quantized training support
import bitsandbytes as bnb

# Quantize the given module, for training purpose
# return both the quentized data, and state
def quantize_training_module(mod_weight, type="4bit"):
    if type == "4bit":
        return bnb.functional.quantize_4bit((mod_weight).to('cuda'))
    elif type == "nf4":
        return bnb.functional.quantize_nf4((mod_weight).to('cuda'))
    elif type == "fp4":
        return bnb.functional.quantize_fp4((mod_weight).to('cuda'))
    else:
        raise ValueError(f"Unknown quantization type {type}")

# Dequantize the given module, for training purpose
def dequantize_training_module(mod_weight, mod_optimizer, type="4bit"):
    if type == "4bit":
        return bnb.functional.dequantize_4bit(mod_weight, mod_optimizer)
    elif type == "nf4":
        return bnb.functional.dequantize_nf4(mod_weight, mod_optimizer)
    elif type == "fp4":
        return bnb.functional.dequantize_fp4(mod_weight, mod_optimizer)
    else:
        raise ValueError(f"Unknown quantization type {type}")

# Quantizer module wrapper, used for training
class QuantizedModule(JITModClass):

    # Setup the quantized module representation, which can be used for training
    def __init__(self, module, quantize_type="4bit"):
        self.qType = quantize_type
        self.optState = None
        self.qData = None

        # Quantize the initialized module
        self.qData, self.optState = quantize_training_module(module.weight, quantize_type)

    # Get the dequentized module, for training purpose
    def dequantize(self):
        return dequantize_training_module(self.qData, self.optState, self.qType).to(torch.bfloat16)

    # Handle the module operations, with dequantized computation
    def __call__(self, *args, **kwargs):
        return self.dequantize()(*args, **kwargs)
