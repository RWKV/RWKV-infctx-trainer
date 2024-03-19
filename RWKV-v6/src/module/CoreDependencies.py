### ---
# External shared pytorch dependencies, used across multiple modules
#
# Other modules in this codebase should not be imported in this file, 
# only external dependencies, and shared dependencies should be imported here
### ---

import gc, math, os
from random import randint
import time
from typing import List, Optional

import numpy as np
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F

import lightning as L
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from lightning.pytorch.strategies import DeepSpeedStrategy

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed.runtime.lr_schedules
import wandb

from torch.utils.cpp_extension import load

# Script dir for various files
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CUDA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../cuda"))

### ---
# JIT / torch compile special handling
### ---

# Because we are using APIs avaliable only to pytorch 2.1.0
# We will throw an error if the user is using a lower version
from packaging import version
def is_torch_version_above(required_version):
    torch_version = version.parse(torch.__version__.split('+')[0])
    return torch_version >= version.parse(required_version)

# Torch versioning flags
IS_TORCH_2_1_COMPATIBLE = is_torch_version_above("2.1.0")
# IS_TORCH_2_1_2_COMPATIBLE = is_torch_version_above("2.1.2")

# Get the JIT / torch compile option flags from the environment
# This default is FOR inference mode, the trainer mode default is configured in the lightning_trainer.py
global RWKV_JIT_ON, RWKV_TORCH_COMPILE, RWKV_TORCH_RUN_MODE
RWKV_TORCH_RUN_MODE = None
if 'RWKV_JIT_ON' not in globals():
    RWKV_JIT_ON         = os.getenv("RWKV_JIT_ON", "1").lower() in ("1", "true", "yes")
if 'RWKV_TORCH_COMPILE' not in globals():
    RWKV_TORCH_COMPILE  = os.getenv("RWKV_TORCH_COMPILE", f"0").lower() in ("1", "true", "yes")

# The RWKV_NO_CUDA global
global RWKV_NO_CUDA
if 'RWKV_NO_CUDA' not in globals():
    RWKV_NO_CUDA = os.getenv("RWKV_NO_CUDA", f"1").lower() in ("1", "true", "yes")

# Enforce no cuda, if there is no cuda
if torch.cuda is None or torch.cuda.is_available() == False or torch.cuda.device_count() <= 0:
    print(f"[RWKV.model] No CUDA device found, enforcing RWKV_NO_CUDA=True")
    RWKV_NO_CUDA = True

# Disable torch compile if its not atleast v2.1.0
if not IS_TORCH_2_1_COMPATIBLE:
    RWKV_TORCH_COMPILE = False

# We enable JITMod*/Function when supporting torch.jit
# We use TorchCompile* when supporting torch compile
# based on the current runtime settings
if RWKV_TORCH_COMPILE:
    RWKV_TORCH_RUN_MODE = "torch-compile"

    JITModClass  = nn.Module
    JITModMethod = lambda x: x
    JITFunction  = lambda x: x

    # PS: i have tried mode="max-autotune", and mode="reduce-overhead", however they crash
    #     for now (8th July 2023). I may introduce them in the future once they are stable
    #
    #     Additionally, torch.compile has issues with the pytorch.lightning module directly
    # ---

    # We generally have 2 major options, either we use torch.compile
    # onto the key top level functions (train, val, test, predict, etc)
    # and let the compiler handle all the decision making on how to optimize
    #
    # However this was found to basically just match JIT level of performance exactly
    # ---
    # TCompileMax          = lambda x: x
    # TCompileBaseline     = lambda x: torch.compile(x, backend='default', fullgraph=False)

    # Alternatively, we can perform a much more aggressive optimization on critical functions
    # that we know are compatible with torch.compile(fullgraph=True) - which provides the highest
    # level of optimization possible with torch.compile
    # ---

    # mode="max-autotune" gives issues presently?
    TCompileMax        = lambda x: torch.compile(x, mode="default", fullgraph=True)
    TCompileBaseline   = lambda x: torch.compile(x, mode='default', fullgraph=False)

    # Running in eager mode?   
    torch._dynamo.config.suppress_errors = True

    # ---
    # Because torch.compile is expected to change overtime, the two options should 
    # be tested every now and then, for any performance changes
    #
    # and we should switch over to the broaded automated approach if its "faster"
    # ---

    # ---
    # For debugging, disable everything
    # ---

    # TCompileMax        = lambda x: x
    # TCompileBaseline   = lambda x: x
    # TCompileDisable    = lambda x: x

    # Used to wrap functions which are **not** torch.compile compatible
    TCompileDisable    = torch._dynamo.disable

    # The following are known warnings in the nightly build, that can be safely ignored for stable release
    #
    # `torch._inductor.utils: [WARNING] DeviceCopy in input program` 
    # https://discuss.pytorch.org/t/what-can-cause-warning-devicecopy-in-input-program/175566

    # Added warning
    print(f"[RWKV.model][WARNING] - torch.compile is enabled, but this has been observed to perform worse, or even crash in some setup. Ensure to test if you actually measure speedups over JIT before using for large training runs'")

elif RWKV_JIT_ON:
    RWKV_TORCH_RUN_MODE = "torch-jit"
    JITModClass  = torch.jit.ScriptModule
    JITModMethod = torch.jit.script_method
    JITFunction  = torch.jit.script

    # JITModClass  = nn.Module
    # JITModMethod = lambda x: x
    # JITFunction  = lambda x: x

    TCompileMax        = lambda x: x
    TCompileBaseline   = lambda x: x
    TCompileDisable    = lambda x: x
else:
    RWKV_TORCH_RUN_MODE = "torch-native"
    JITModClass  = nn.Module
    JITModMethod = lambda x: x
    JITFunction  = lambda x: x

    TCompileMax        = lambda x: x
    TCompileBaseline   = lambda x: x
    TCompileDisable    = lambda x: x

print(f"[RWKV.model] Running RWKV infctx using '{RWKV_TORCH_RUN_MODE}' with torch '{torch.__version__}'")
