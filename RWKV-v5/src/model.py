### ---
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
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

# Currently the features we need for torch compile, is avaliable only in
# 2.1 nightly build (and is expected to be in 2.1 official release)
#
# However because the nightly build torch.compile behaviour has been unstable
# the versioning code to check for enabling toch compile will not be used, 
# until we confirm a stable version of torch.compile
from packaging import version
def is_torch_version_above(required_version):
    torch_version = version.parse(torch.__version__.split('+')[0])
    return torch_version >= version.parse(required_version)
IS_TORCH_2_1 = is_torch_version_above("2.0.9999")

# Get the JIT / torch compile option flags from the environment
RWKV_JIT_ON        = os.getenv("RWKV_JIT_ON", "1").lower() in ("1", "true", "yes")
RWKV_TORCH_COMPILE = os.getenv("RWKV_TORCH_COMPILE", f"0").lower() in ("1", "true", "yes")
RWKV_TORCH_RUN_MODE = None

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
    # TCompileBaseline     = lambda x: torch.compile(x, fullgraph=False)

    # Alternatively, we can perform a much more aggressive optimization on critical functions
    # that we know are compatible with torch.compile(fullgraph=True) - which provides the highest
    # level of optimization possible with torch.compile
    # ---
    TCompileMax        = lambda x: torch.compile(x, fullgraph=True)
    TCompileBaseline   = lambda x: x

    # ---
    # Because torch.compile is expected to change overtime, the two options should 
    # be tested every now and then, for any performance changes
    #
    # and we should switch over to the broaded automated approach if its "faster"
    # ---

    # Used to wrap functions which are **not** torch.compile compatible
    TCompileDisable    = torch._dynamo.disable

    # The following are known warnings in the nightly build, that can be safely ignored for stable release
    #
    # `torch._inductor.utils: [WARNING] DeviceCopy in input program` 
    # https://discuss.pytorch.org/t/what-can-cause-warning-devicecopy-in-input-program/175566

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

print(f"[RWKV.model] Running RWKV model using '{RWKV_TORCH_RUN_MODE}' with torch '{torch.__version__}'")

# ---
# Isolating out known operations that **does not work** with torch.compile
# and wrapping them within a torch._dynamo.disable, this is required to get
# the baseline torc.compile to work
# ---

@TCompileDisable
def deepspeed_checkpoint(*args, **kwargs):
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)

### ---
# RWKV: State Blocks
### ---

class TimeMixState:

    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:

    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:

    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    # @ TCompileMax (no difference)
    @staticmethod
    def create(N, B, C, n_head, head_size, device, dtype):
        result = BlockStateList.empty(N, B, C, n_head, head_size, device, dtype)
        result.wkv_states[:] = 0
        # result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    # @ TCompileMax (no difference)
    @staticmethod
    def empty(N, B, C, n_head, head_size, device, dtype):
        # @TODO: confirm if dtype can be changed from .flaot to dtype=dtype (when bf16)
        wkv_states = torch.empty((N, B, 1, n_head, head_size, head_size),
                                 device=device,
                                #  dtype=dtype)
                                 dtype=torch.float)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state

### ---
# RWKV: RWKV Time-mix + RWKV Channel-mix
### ---

class RWKV_TimeMix(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att):
        super().__init__()
        
        self.dim_att = dim_att
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.layer_id = layer_id

        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8

        # Optimized timemix chunk length is fixed for now
        self.chunk_len = 512
        # assert ctx_len % self.chunk_len == 0

        # V5-R4 changes
        # https://github.com/BlinkDL/RWKV-LM/commit/5aab658f945ba80745d36c2ab411fb43df3a74f9    
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(dim_att)
            for n in range(dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.key = nn.Linear(n_embd, dim_att, bias=False)

        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, dim_att, bias=False)
        self.ln_x = nn.GroupNorm(n_head, dim_att)

    @TCompileMax
    def forward(self, x, last_state: TimeMixState):
        # Get the x sizing
        B, TT, C = x.size()

        # Perform the tokenshift, and get the respective state
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1)
    
        # Get the xk, xv, xr, xg
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr).view(B, TT, self.n_head, 1, -1)
        k = self.key(xk).view(B, TT, self.n_head, -1, 1)
        v = self.value(xv).view(B, TT, self.n_head, 1, -1)
        g = F.silu(self.gate(xg))


        u = self.time_faaaa.view(1,1,self.n_head, 1, -1)
        w = self.time_decay.exp().neg().exp().reshape(1,1, self.n_head,-1,1)
        # The WKV state to update
        if last_state.wkv_state is None:
            wkv_state = torch.zeros((B, 1, self.n_head, self.head_size, self.head_size),dtype=r.dtype)
        else:
            # Clone is required, due to the way backprop works
            wkv_state = last_state.wkv_state.clone().to(r.dtype)

        # Compute attent and the initial output tensor
        at = k @ v
        
         
        # Slightly inefficent, but it works, lets compute all the tokens
        ms = [wkv_state]
        
        
        for t in range(1,TT+1):
            ms = ms + [at[:,t-1] + ms[t-1] * w]
            
        
        out = (u * r ) @ at + (r @ torch.cat(ms[:-1],1))
        # Compute the final x output
        x_logits = out.view(-1, C)
        x_logits = self.ln_x(x_logits / 8).view(B, TT, C)
        x_logits = self.output(x_logits * g)

        
        # Return the logits and the state
        return x_logits, TimeMixState(x[:,-1],ms[-1])
    
        # print(f"B: {B}, TT: {TT}, C: {C}, chunk_len: {chunk_len}")
        # print(f"Original Shapes - r: {r.shape}, k: {k.shape}, v: {v.shape}, g: {g.shape}, x: {x.shape}")

        # # Split the input by TT chunks, and process it
        # for i in range(0, TT, chunk_len):
        #     x_chunk = x[:, i:i+chunk_len, :]
        #     r_chunk = r[:, i:i+chunk_len, :]
        #     k_chunk = k[:, i:i+chunk_len, :]
        #     v_chunk = v[:, i:i+chunk_len, :]
        #     g_chunk = g[:, i:i+chunk_len, :]

        #     print(f"Chunked shapes - r: {r_chunk.shape}, k: {k_chunk.shape}, v: {v_chunk.shape}, g: {g_chunk.shape}, x: {x_chunk.shape}")
        #     print(f"WKV state size : {last_state.wkv_state.shape}")
        #     print(f"X logits size : {x_logits.shape}")
        #     print(f"X chunk size : {x_chunk.shape}")

        #     chunk_logits, last_state = self._forward_chunk(x_chunk, last_state, r_chunk, k_chunk, v_chunk, g_chunk)
        #     x_logits[:, i:i+chunk_len, :] = chunk_logits

        # # Chunk length to split by, 
        # # we probably can reoptimize this at some point
        # chunk_len = self.chunk_len


    # # this is based on jit_func(self,x)
    # @JITModMethod
    # def _get_rwkvg(self, x, last_state: TimeMixState, B:int, TT:int, C:int):
    #     # Mix x with the previous timestep to produce xk, xv, xr
    #     # this is the token shift
    #     xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1)

    #     xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    #     xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
    #     xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    #     xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

    #     r = self.receptance(xr).view(B, TT, self.n_head, 1, -1)
    #     k = self.key(xk).view(B, TT, self.n_head, -1, 1)
    #     v = self.value(xv).view(B, TT, self.n_head, -1, 1)
    #     g = F.silu(self.gate(xg))

    #     return r, k, v, g

    # def _forward_chunk(self, x_chunk, last_state: TimeMixState, r, k, v, g):
    #     B, T_chunk, C = x_chunk.size()
    #     H = self.n_head  # Number of heads
    #     head_size = self.head_size

    #     # If last_state.wkv_state is not provided or its shape doesn't match, initialize it
    #     if last_state.wkv_state is None or last_state.wkv_state.size() != (B, H, self.n_head):
    #         wkv_state = torch.zeros((B, H, self.n_head), device=x_chunk.device, dtype=x_chunk.dtype)
    #     else:
    #         # Or clone the state accordingly
    #         wkv_state = last_state.wkv_state.clone()

    #     # Compute the attention k/v
    #     at = k @ v

    #     # Compute u expended, and the initial output tensor
    #     u_expanded = self.time_faaaa.view(1,1,self.n_head, 1, -1)
    #     out = (u_expanded * r) @ at

    #     # Update the output with r @ wkv_state
    #     w_expanded = self.time_decay.double().exp().neg().exp().reshape(1, self.n_head,-1,1)

    #     # Compute the full output tensor
    #     for t in range(T_chunk):
    #         out[:,t] += r[:,t] @ wkv_state
    #         wkv_state *= w_expanded
    #         wkv_state += at[:,t]

    #     # Compute the final x_chunk output
    #     x_chunk = out.view(-1, C)
    #     x_chunk = self.ln_x(x_chunk / 8).view(B, T_chunk, C)
    #     x_chunk = self.output(x_chunk * g)

    #     # Save the updated state back to last_state.wkv_state for the next chunk
    #     last_state.wkv_state = wkv_state

    #     # Return with x distribution logits, and last state
    #     return x_chunk, last_state







    # def _forward_chunk(self, x, last_state: TimeMixState, r, k, v, g):
    #     # Forward sizings (Batch, Time/ContextLength, Tokens)
    #     B, TT, C = x.size()
    #     B = torch.tensor(B, device=x.device, dtype=torch.int32)
    #     TT = torch.tensor(TT, device=x.device, dtype=torch.int32)

    #     # Get w, wk, wb, ws (self.jit_func_2)
    #     w, wk, wb, ws = self._forward_wkbs_chunk(TT, r, k, v)

    #     # Does the state forwarding
    #     return self._forward_state_chunk(r, k, v, g, w, wk, wb, ws, x[:, -1], last_state)

    # @JITModMethod
    # def _forward_state_chunk(self, r, k, v, g, w, wk, wb, ws, x_l, last_state: TimeMixState):
    #     B, H, TT, S = r.size()
    #     T = TT

    #     # s = torch.zeros(B, H, S, S, device=r.device, dtype=r.dtype)  # state
    #     s = last_state.wkv_state
    #     if r.dtype == torch.bfloat16 and s.dtype != torch.bfloat16:
    #         s = s.contiguous().to(torch.bfloat16)

    #     x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype) # output

    #     ########################################################################
    #     for i in range(TT // T):
        
    #         rr = r[:, :, i*T:i*T+T, :]
    #         kk = k[:, :, :, i*T:i*T+T]
    #         vv = v[:, :, i*T:i*T+T, :]

    #         x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv + (rr @ s) * wb

    #         s = ws * s + (kk * wk) @ vv
    #     ########################################################################
        
    #     x = x.transpose(1, 2).contiguous().view(B * TT, H*S) # BHTS -> BTHS -> BTC

    #     # x = self.ln_x(x/self.head_size_divisor).view(B, TT, H*S)
    #     x = self.ln_x(x/8).view(B, TT, H*S)

    #     # Fix missing *g for output as per :
    #     # https://github.com/RWKV/RWKV-infctx-trainer/commit/beb46d599042b77d53db9c7fa59a5966e7d33719#r126730367
    #     return self.output(x*g), TimeMixState(x_l, s)

    # def _forward_wkbs_chunk(self, T, r, k, v):
    #     H = self.n_head

    #     w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1)

    #     # V5-R2 changes
    #     u = self.time_faaaa.float().unsqueeze(-1)
    #     # u = torch.exp(self.time_first.float()).unsqueeze(-1)

    #     ws = w.pow(T).reshape(1, H, 1, 1)
    #     ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
    #     w = w.repeat(1, T).pow(ind)

    #     wk = w.reshape(1, H, 1, T)
    #     wb = wk.transpose(-2, -1).flip(2)

    #     w = torch.cat([w[:, 1:], u], dim=1)
    #     w = F.pad(w, (0, T))
    #     w = torch.tile(w, [T])
    #     w = w[:, :-T].reshape(-1, T, 2 * T - 1)
    #     w = w[:, :, T-1:].reshape(1, H, T, T)

    #     w = w.to(dtype=r.dtype)
    #     wk = wk.to(dtype=r.dtype)
    #     wb = wb.to(dtype=r.dtype)
    #     ws = ws.to(dtype=r.dtype)

    #     return w, wk, wb, ws


### ---


class RWKV_ChannelMix(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
        super().__init__()
        # self.layer_id = layer_id
        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(dim_ffn, n_embd, bias=False)

    @JITModMethod
    @TCompileMax
    def forward(self, x, last_state: ChannelMixState):
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]),
                          dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return (torch.sigmoid(self.receptance(xr)) * kv,
                ChannelMixState(x[:, -1]))


### ---
# The RWKV Model blocks
### ---

class Block(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dropout, dim_att, dim_ffn):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, n_head, head_size, dim_att)
        self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)

        # Setup droupout at block level
        self.dropout = dropout
        if dropout > 0:            
            self.drop0 = nn.Dropout(p = dropout)
            self.drop1 = nn.Dropout(p = dropout)

    def forward(self, x, last_state: BlockState):
        if self.layer_id == 0:
            x = self.ln0(x)

        att_out, att_state = self.att(
            self.ln1(x),
            last_state.time_mix_state,
        )

        if self.dropout > 0.0:
            # Handle with dropout
            x = self.drop0(x + att_out)
            ffn_out, ffn_state = self.ffn(
                self.ln2(x),
                last_state.channel_mix_state,
            )
            x = self.drop1(x + ffn_out)
        else:
            # Handle without dropout
            x = x + att_out
            ffn_out, ffn_state = self.ffn(
                self.ln2(x),
                last_state.channel_mix_state,
            )
            x = x + ffn_out
        
        return x, BlockState(att_state, ffn_state)


class L2Wrap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, y, token_amount, currentMask):
        # Currently (8th July 2023), save_for_backward, causes an issue with
        # pytorch.compile (see: https://github.com/pytorch/pytorch/blob/e600505e3209eaf539e8bc99870ea55236cefbf5/torch/_dynamo/variables/higher_order_ops.py#L735)
        # 
        # Due to L2Wrap being a major hotspot, we should monitor this for future support.
        # so that once its resolved, we can include the L2Wrap step in the torch.compile path
        #
        # See also:
        # - checkpointed_step
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
        ctx.currentMask = currentMask
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        token_amount = ctx.token_amount
        # to encourage the logits to be close to 0
        factor = 1e-4 / token_amount
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        gy = gy * ctx.currentMask[:, None][None, :]
        return (grad_output, gy, None, None)

### ---
# Static optimized functions
### ---

# @ TCompileMax (no speed improvement)
# def F_cross_entropy_reduction_none_optimized(logits, targets):
#     return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")

### ---
# Core RWKV module
### ---
class RWKV(L.LightningModule):

    def __init__(self,
                 # Model file path to load from
                 load_model: str,
                 # Model size settings, which we either
                 # "auto detect", or use the user specified settings
                 n_embd: int = -1,
                 n_layer: int = -1,
                 vocab_size: int = -1,
                 # Context length size for the model
                 ctx_len: int = 2048,
                 # Context length schedule
                 ctx_len_cutoffs: List[int] = [],
                 ctx_len_warmup_steps: List[int] = [],
                 # Learning rate schedule
                 # use only target_lr_init / lr_init
                 # to configure a constant learning rate
                 lr_init: float = -1.0,
                 lr_final: float = -1.0,
                 lr_period: int = -1,
                 lr_period_type: str = 'epoch',
                 # Dropout rate
                 dropout: float = 0.0,
                 # Adam optimizer settings
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 adam_eps: float = 1.0e-08,
                 weight_decay: float = 0.01,
                 warmup_steps: int = -1,
                 # loss bias start
                 position_loss_bias: float = 1.0,
                 position_loss_bias_in_validation: bool = False,
                 # Backprop settings
                 grad_cp: bool = True,
                 bptt_learning: bool = True,
                 bptt_learning_range: int = -1,
                 bptt_truncated_learning: bool = False,
                 layerwise_lr: bool = True,
                 dim_att: Optional[int] = None,
                 dim_ffn: Optional[int] = None,
                 substep_cuda_cache_clear: bool = False,
                 substep_logging: bool = False,
                 torch_set_float32_matmul_precision:str = 'high'
                 ):

        # Lets save everything in one shot
        # (this is used for wandb logging)
        self.setup_args = locals()
        del self.setup_args["self"]
        del self.setup_args["__class__"]

        # Setup the parent class
        super().__init__()

        # Load the model, unless its the special ".//<#|=@%!$init_model$!%@=|#>//." path
        # which is reserved to be used with the `init_model.py`
        #
        # We intentionally used several filesystem illegal characters, to ensure it
        # is not accidentally used by the user for a real file
        model_weights = None
        model_keys = None
        if load_model != ".//<#|=@%!$init_model$!%@=|#>//.":
            # Check if the load_model path exists, and is a file
            if not os.path.isfile(load_model):
                raise ValueError(f"load_model file '{load_model}' does not exist")

            # Load the model weights
            model_weights = torch.load(load_model, map_location='cpu')

            # Get the model keys
            model_keys = list(model_weights.keys())

        # Lets compute the model various sizes, if they are not provided
        if n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                    max_block_id = max(max_block_id, block_id)
            n_layer = max_block_id + 1

        if n_embd < 0:
            n_embd = model_weights['head.weight'].shape[1]
        
        if vocab_size < 0:
            vocab_size = model_weights['head.weight'].shape[0]

        # Save the various other params for later
        self.ctx_len = ctx_len
        self.ctx_len_cutoffs = ctx_len_cutoffs
        self.ctx_len_warmup_steps = ctx_len_warmup_steps
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.layerwise_lr = layerwise_lr
        self.grad_cp = grad_cp
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_period = lr_period
        self.lr_period_type = lr_period_type
        self.dropout = dropout
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.bptt_learning = bptt_learning
        self.bptt_learning_range = bptt_learning_range
        self.bptt_truncated_learning = bptt_truncated_learning
        self.substep_cuda_cache_clear = substep_cuda_cache_clear
        self.substep_logging = substep_logging

        # Save the position loss params
        self.position_loss_bias = position_loss_bias
        self.position_loss_bias_in_validation = position_loss_bias_in_validation

        dim_att = dim_att or n_embd
        dim_ffn = dim_ffn or int((n_embd * 3.5) // 32 * 32)
        self.dim_att = dim_att
        self.dim_ffn = dim_ffn

        # Compute the RWKV-v5 n_head / headsize
        head_size = 64
        self.head_size = head_size
        self.head_size_divisor = 8

        n_head = dim_att // head_size
        self.n_head = n_head
        assert dim_att % n_head == 0 ,  f"dim_att must be divisible by head_size ({self.head_size})"

        # Validate various sizes
        assert n_embd  % 32 == 0, f"n_embd must be divisible by 32"
        assert dim_att % 32 == 0, f"dim_att must be divisible by 32"
        assert dim_ffn % 32 == 0, f"dim_ffn must be divisible by 32"
        
        # Matmu precision check
        if torch_set_float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(torch_set_float32_matmul_precision)

        self.emb = nn.Embedding(vocab_size, n_embd)

        # load(name=f"wkv_{self.ctx_len}_bf16",
        #      sources=[
        #         os.path.join(CUDA_DIR, "wkv_op_bf16.cpp"),
        #         os.path.join(CUDA_DIR, "wkv_cuda_bf16.cu")
        #     ],
        #      verbose=True,
        #      extra_cflags=["-std=c++17", "-O3", f"-DTmax={self.ctx_len}"],
        #      extra_cuda_cflags=[
        #          "-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60",
        #          "--use_fast_math", "-O3", "-Xptxas -O3",
        #          "--extra-device-vectorization", f"-DTmax={self.ctx_len}"
        #      ],
        #      is_python_module=False)

        self.blocks = nn.ModuleList([
            Block(i, n_layer, n_embd, n_head, head_size, dropout, dim_att, dim_ffn) for i in range(n_layer)
        ])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Dropout handling
        if dropout > 0:
            self.drop0 = nn.Dropout(p = dropout)

        # load the state, and GC the original cpu copy
        if model_weights != None:
            self.load_state_dict(model_weights)
            del model_weights
            gc.collect()

        # Training based timings to track, and initialize
        self._counting_tokens = 0
        self._counting_time_start = 0

    def configure_optimizers(self):
        if self.bptt_learning == False:
            if self.deepspeed_stage >= 2 or self.deepspeed_offload:
                print(f"[WARNING]: it is highly recommended to enable bptt_learning when used to deepspeed 2/3/offloading, otherwise an exception will occur when training with dataset records, larger then the configured context length ({self.ctx_len})")
        else:
            if self.trainer.num_devices > 1:
                if self.bptt_learning_range <= 0:
                    print("[WARNING]: unlimited bptt_learning_range across multiple GPU's has a performance penalty with datasets of mixed sizes due to its constant need to keep all GPU's in sync (consider using bptt_learning_range=1 instead)")
        
        # Get the learning rate used for the optimizer
        lr_init = self.lr_init
        lr_final = self.lr_final
        # If the final learning rate is not specified, use the initial learning rate
        if lr_final < 0:
            lr_final = self.lr_init

        # Log the learning rate, and various other parameters
        if self.trainer.local_rank == 0:

            # Add the important notes, for informing users of common gotchas
            print((
                "#\n"
                "# RWKV lighting_trainer.py important notes \n"
                "# https://github.com/RWKV/RWKV-infctx-trainer \n"
                "#\n"
                "# - Ensure your host is not running cuda 12.0 (use either 11.8, or >=12.1), as this is known to have freeze issues\n"
                "# - The terms used in wandb / the progress bar can be confusing, see the github README.md for beter clarifications\n"
                "# - When resuming from checkpoint, the estimated time is inaccurate\n"
                "#"
            ))

            lr_init_e = "{:.3e}".format(lr_init)
            lr_final_e = "{:.3e}".format(lr_final)
            print(f"\n[RWKV.model] Configuring optimizer with\n"+
                  f"    - lr_init:  {lr_init_e} ({lr_init})\n"+
                  f"    - lr_final: {lr_final_e} ({lr_final})\n")

            # Get the setup args
            model_args = dict(self.setup_args)
            model_args["__lr_init"] = lr_init
            model_args["__lr_final"] = lr_final

            # Update WANDB
            if wandb.run is not None:
                wandb.config.update({ "model": model_args })

        # Setup layerwise learning rate
        if self.layerwise_lr:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    lr_1x.add(n)
                elif "time_decay" in n:
                    lr_2x.add(n)
                # V5-R2 changes
                elif "time_faaaa" in n:
                    lr_2x.add(n)
                # elif "time_first" in n:
                #     lr_3x.add(n)
                else:
                    lr_1x.add(n)
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            # print('1x', lr_1x)
            # print('2x', lr_2x)
            # print('3x', lr_3x)
            param_dict = {n: p for n, p in self.named_parameters()}
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 2.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.0,
                    "lr": 3.0 * lr_init
                },
            ]
        else:
            optim_groups = [
                {
                    "params": [p for n, p in self.named_parameters()],
                    "weight_decay": 0.0
                },
            ]

        # Setup the adam optimizers
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(self.beta1, self.beta2),
                                         eps=self.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=False,
                                         weight_decay=self.weight_decay,
                                         amsgrad=False)
        else:
            optimizer = FusedAdam(optim_groups,
                                  lr=lr_init,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.adam_eps,
                                  bias_correction=True,
                                  adam_w_mode=False,
                                  weight_decay=self.weight_decay,
                                  amsgrad=False)
            
        # Throw if wramup_steps and lr_period are both set (not supported)
        if self.warmup_steps > 0 and self.lr_period > 0:
            raise ValueError(
                "Use either warmup_steps or lr_period, not both.")

        if self.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.lr_init,
                warmup_max_lr=self.lr_init,
                warmup_num_steps=self.warmup_steps,
                warmup_type='linear')

            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
            }

        else:
            # Skip the lr_scheduler process if lr_init and lr_final are the same
            if lr_init == lr_final:
                return optimizer

            # The total number of steps to perform training rate decay with
            lr_total_step = 0

            # Handle lr_period -1 default behaviour of using the max_step / max_epoch
            if self.lr_period == -1:
                # Get trainer max_step / max_epoch
                trainer_max_step = self.trainer.max_steps
                trainer_max_epoch = self.trainer.max_epochs
                if trainer_max_step > 0:
                    lr_total_step = trainer_max_step
                elif trainer_max_epoch > 0:
                    lr_total_step = trainer_max_epoch * self.num_step_per_epoch()
                else :
                    print("Warning: max_step/max_epoch not set, we would be performing lr_init to lr_final shift assuming 10 epoch")
                    lr_total_step = 10 * self.num_step_per_epoch()
            else:
                # Calculate lr_total_step based on lr_period
                if self.lr_period_type == "step":
                    lr_total_step = self.lr_period
                elif self.lr_period_type == "epoch":
                    lr_total_step = self.lr_period * self.num_step_per_epoch()
                else:
                    raise ValueError(f"lr_period_type {self.lr_period_type} not supported.")

            # Lets initialize the lr_scheduler
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor= lr_final / lr_init,
                total_iters=lr_total_step
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
                
    
    # We have to compute the number of steps per epoch ourselves
    # as this value is not provided directly by pytorch lightning
    # https://github.com/Lightning-AI/lightning/issues/5449#issuecomment-1501597319
    def num_step_per_epoch(self) -> int:
        # Estimated number of steps in total, added as the following
        # https://github.com/Lightning-AI/lightning/pull/11599
        #
        # This MUST be called before len(self.trainer.train_loader)
        # otherwise there is a bug in which the train_dataloader is not
        # fully initialized, which seems to be resolved by computing the 
        # self.trainer.estimated_stepping_batches
        estimated_stepping_batches = self.trainer.estimated_stepping_batches

        # Get the number of epochs, 
        # use estimated_stepping_batches if max_epochs is set
        max_epochs = self.trainer.max_epochs
        if max_epochs > 0:
            return estimated_stepping_batches // max_epochs

        # Get the train_dataloader
        train_dataloader = self.trainer.train_dataloader
        if train_dataloader is None:
            train_dataloader = self.trainer.fit_loop._data_source.dataloader()

        # Max epoch is not set, use the train_dataloader
        dataset_size = len(train_dataloader)

        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size // (self.trainer.accumulate_grad_batches * num_devices)
        return num_steps
    
    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return "offload_optimizer" in cfg or "offload_parameters" in cfg
        return False
    
    @property
    def deepspeed_stage(self) -> int:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return "stage" in cfg
        return -1

    @TCompileBaseline
    def forward(self, idx: torch.Tensor, last_shift_states: torch.Tensor = None,
                last_wkv_states: torch.Tensor = None):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        # Handle dropout (input)
        if self.dropout > 0.0:
            x = self.drop0(x)

        new_states = BlockStateList.empty(self.n_layer, B, self.n_embd, 
                                          self.n_head, self.head_size,
                                          x.device, x.dtype)
        
        # last_shift_states can be None, when we are performing direct inference
        if last_shift_states is None:
            cur_bs_list = BlockStateList.create(
                self.n_layer, B, self.n_embd, 
                self.n_head, self.head_size,
                x.device, x.dtype
            )
        else:
            cur_bs_list = BlockStateList(last_shift_states, last_wkv_states)

        # Avoid using the zip operation, as torch.compile throws an exception on it
        # with `zip not reconized as a valid function`
        # ---
        # for i, (block, last_state) in enumerate(
        #         zip(self.blocks,
        #             BlockStateList(last_shift_states, last_wkv_states))):
        # ---
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = cur_bs_list[i]
            if self.grad_cp:
                x, new_state = deepspeed_checkpoint(
                    block, x, last_state)
            else:
                x, new_state = block(x, last_state)
            new_states[i] = new_state

        x = self.ln_out(x)

        x = self.head(x)

        return x, new_states.shift_states, new_states.wkv_states

    #
    # Custom overwrite of manual_backwards operation, to skip the "manual_backwards"
    # safety check, so we can perform manual backward operation step, while using
    # the default trainer loop. This is modified from the original code found here:
    # https://github.com/Lightning-AI/lightning/blob/37c244f94be365496def82870b22c2faf0ab889e/src/lightning/pytorch/core/module.py#L999
    #
    # ---
    # 
    # This allow us to avoid disabling the "automatic_optimization" flag
    #
    # Which would have been required to do "segmented learning", or "Backpropagation Through Time"
    # where we would need to implement manual optimization as per
    # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    #
    # Otherwise an error will be thrown if we call `self.manual_backward`
    #
    # However this would mean that we would need to do a full reimplementation
    # of several features that were handled by the automatic optimization.
    # - accumulate_grad_batches
    # - gradient_clip_val
    # - logging behaviour
    # - distributed training co-ordination
    # - (And probably other features that I am not aware of)
    #
    # So this is a hacky work around, to avoid reimplementing all of the above.
    # 
    # From the current code implementatiion, it seem like this is blocked only by 
    # automatic_optimization flag - and has no adverse side effect otherwise
    # https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule.manual_backward
    #
    # If anyone have a better idea, let me know
    # (have experimented with, reimplementing the above, but it is not trivial, unfortunately)
    #
    def manual_backward(self, loss: torch.Tensor, *args, **kwargs):
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            # self._verify_is_manual_optimization("manual_backward")
            self.trainer.strategy.backward(loss, None, *args, **kwargs)

    #
    # Main compute_loss function, this is called by the trainer loop
    #
    def compute_loss(self, batch, batch_idx, is_training_run: bool):

        # Used for token/second performance tracking
        if self._counting_tokens is None or batch_idx == 0:
            self._counting_tokens = 0
        if self._counting_time_start is None or batch_idx == 0:
            self._counting_time_start = time.time()
            
        seq = batch['input_ids']
        assert isinstance(seq, torch.Tensor) and seq.ndim == 2
        ori_seq_mask = batch['attention_mask']

        # Check if attent mask is set, if not initialize it
        if ori_seq_mask is None or ori_seq_mask.ndim != 2:
            ori_seq_mask = torch.ones_like(seq[:, 1:])

        # Get the starting and ending loss bias
        loss_bias_start = self.position_loss_bias
        loss_bias_end   = 2.0 - loss_bias_start

        # total_mask_sum
        total_mask_sum = torch.sum(ori_seq_mask)

        # Skip loss bias calculation, if loss_bias_start is 1.0
        if loss_bias_start == 1.0 or (is_training_run == False and self.position_loss_bias_in_validation == False):
            seq_mask = ori_seq_mask
        else:
            # Lets get a linear multiplier for the loss bias
            # seq_mask_sum = torch.sum(ori_seq_mask)
            bias_mask = torch.linspace(loss_bias_start, loss_bias_end, int(total_mask_sum.item()), device=ori_seq_mask.device)

            # Boolean flag of seq_mask > 0
            seq_mask_index = ori_seq_mask[0] > 0

            # Apply the bias mask only to positive seq_mask values
            final_mask = torch.zeros(ori_seq_mask.shape[1], device=ori_seq_mask.device)
            final_mask[seq_mask_index] = ori_seq_mask[0][seq_mask_index] * bias_mask

            # And save it as seq_mask
            seq_mask = final_mask.unsqueeze(0)

        # Perform cutoff for training run
        if is_training_run:
            prev_step = 0

            # Avoid using the zip operation, as torch.compile throws an exception on it
            # with `zip not reconized as a valid function`
            # ---
            # for step, len_cut in zip(self.ctx_len_warmup_steps,
            #                          self.ctx_len_cutoffs):
            # ---
            for i in range(min(len(self.ctx_len_warmup_steps), len(self.ctx_len_cutoffs))):
                step = self.ctx_len_warmup_steps[i]
                len_cut = self.ctx_len_cutoffs[i]

                if prev_step <= self.global_step < step and len_cut < seq.shape[
                        1] - 1:
                    pos = randint(0, seq.shape[1] - len_cut - 1)

                    # Original
                    # seq = seq[:, pos:pos + len_cut + 1]

                    # Changed to use masking for prefix cutoff (i do not know if this makes sense)
                    seq = seq[:, :pos + len_cut + 1]
                    seq_mask = seq_mask[:, :pos + len_cut + 1]
                    # Set the attention mask to 0 for the skipped tokens
                    seq_mask[:, :pos] = 0
                    break
                prev_step = step
                
        do_bptt_learning = self.bptt_learning and is_training_run
        idx, targets = seq[:, :-1], seq[:, 1:]

        B, T = idx.shape
        C = self.n_embd
        total_mask_sum = torch.sum(seq_mask)

        # If total_mask_sum, we skip, as there is no tokens of value to learn from anyway
        if total_mask_sum == 0:
            return 0
        
        def checkpointed_step(idx, targets, mask, prev_loss, last_shift_states,
                              last_wkv_states, prev_steps):
            logits, new_shift_states, new_wkv_states = self(
                idx, last_shift_states, last_wkv_states)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    targets.view(-1),
                                    reduction="none")

            submask = mask.view(-1)[:loss.shape[0]]
            submask_sum = torch.sum(submask)
            loss = torch.sum(loss * submask) / total_mask_sum  

            loss = L2Wrap.apply(loss, logits, total_mask_sum, submask)
            new_steps = prev_steps + submask_sum
            new_loss = prev_loss + loss
            return new_loss, new_shift_states, new_wkv_states, new_steps

        total_loss = torch.tensor(
            0, dtype=self.emb.weight.dtype).requires_grad_()
        steps = 0
        states = BlockStateList.create(self.n_layer, B, C, 
                                       self.n_head, self.head_size,
                                       seq.device, self.emb.weight.dtype)
        segment_count = math.ceil(T / self.ctx_len)

        #
        # BPTT learning, we split the sequence into segments
        # and perform a backward pass for each segment, on its own.
        #
        # Allowing us to perform backpropagation across context sizes much larger
        # then what is supported by the current GPU memory.
        #
        # This reduces the need for the checkpointing process, and mitigate
        # a known error where multiple backwards pass throws an exception.
        #
        # While not mathematically equivalent to full context size learning,
        # it makes "infctx" size training possible with deepspeed 2/3
        #
        # ---
        # 
        # See the following, for more details on "Gradient computed twice" error:
        # https://github.com/microsoft/DeepSpeed/issues/988#issuecomment-1549417269
        #
        # Other possibly related issues on the topic:
        # https://github.com/microsoft/DeepSpeed/pull/677
        # https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-766366413
        #
        if do_bptt_learning:

            gradient_accumulation_steps = max(1, self.trainer.accumulate_grad_batches)
            optimizer = self.optimizers()
            cur_device = self.device
            
            # We use the average segment size, instead of ctx length size.
            # this helps ensure that the segment cutoffs do not make the last segment too small.
            # (eg, the last chunk having only 1 token)
            #
            # it also helps ensure the segment cutoff points are more varied, across mixed dataset sizes
            # and avoid potentially undesired training behaviour at fixed cutoff points
            # (this only applies for segmented learning)
            segment_size = min(math.ceil(T / segment_count), self.ctx_len)

            # Dummy 2D tenros of shape [1,1], are used to do "dummy checkpoint/forward/backprop" to keep everything in sync
            dummy_2d_zero = torch.tensor([[0]], dtype=torch.long, device=cur_device)

            # Get the max segment count across all GPUs, in the current batch, which is used to keep all devices are in sync
            # Once a thread has completed all its segments, it will do dummy checkpoint/forward/backprop with one token,
            # and stay in sync with the thread that are still working on their segments
            #
            # This is used to work around an undocumented behaviour for either lightning / deepspeed loss.backward multi-gpu handling
            # where the `self.manual_backward()` / `loss.backward()` call will block / freeze / hang when being too "out of sync"
            #
            # This can be viewed as a form of `fabric.barrier()` which is invoked implicitly by each `self.manual_backward()` call
            # except that it isn't exactly a `fabric.barrier()` - because it does not block immediately and instead blocks in 
            # the next `self.manual_backward()` call if the previous ones are too far out of sync. 
            # (its confusing, but makes sense for efficency)
            #
            # Additionally because the "code line position" and params actually matter for the 'barrier' code effect,
            # we cant work around this issue by doing dummy `self.manual_backward()` calls, in another if/else branch or loop
            #
            # Combined, this makes this issue very hard to trace and debug, as it will manifest itself as randomly "freezing"
            # when out of sync during `self.manual_backward()` calls. Without the current work around put in place
            #
            # We only do this, if we are doing bptt learning on all segments (-1), and gpu count > 1
            # otherwise we just use the segment count as it is
            if self.trainer.num_devices > 1:
                if self.bptt_learning_range <= 0:
                    # We perform forward/backward on the shared max segment count across all GPUs
                    # ---
                    # we map it to be a tensor, instead of the int directly, as this is more reliable across certain versions of torch/lightning
                    # https://discord.com/channels/992359628979568762/1148755392638234697/1148821863749931008
                    forward_segment_count  = self.trainer.strategy.reduce(torch.Tensor([segment_count]).to(torch.int), reduce_op="max")

                    # Convert to int, if its a torch tensor
                    if isinstance(forward_segment_count, torch.Tensor):
                        forward_segment_count = forward_segment_count.item()
                    # We perform as many backward pass as we need to be equal or more then bptt_learning_range
                    backward_segment_count = forward_segment_count
                else:
                    # We perform as many forward pass as we need to be equal or more then bptt_learning_range
                    # and perform an equal amount of backward pass
                    forward_segment_count  = max(segment_count, self.bptt_learning_range)
                    backward_segment_count = self.bptt_learning_range
            else:
                if self.bptt_learning_range <= 0:
                    # Since we do not need to sync GPUs here, we perform as much forward as we exactly need
                    forward_segment_count  = segment_count
                    backward_segment_count = forward_segment_count
                else:
                    # We clamp the backward segment count to the forward count, and bptt_learning_range
                    forward_segment_count  = segment_count
                    backward_segment_count = min(self.bptt_learning_range, segment_count)

            # We compute when we start the segmented learning process
            if forward_segment_count != backward_segment_count:
                start_learning_segment = max(segment_count - self.bptt_learning_range, 0);
            else:
                start_learning_segment = 0;

            # # Segment loss array to track (and reduce later)
            # # of size equal to forward_segment_count
            # segment_loss_arr = [0] * forward_segment_count

            # Lets go through and forward all the segments 
            # (including dummy ones)
            for i in range(forward_segment_count):
                # Apply state truncation, if truncated learning is enabled
                # this limits the backprop process, reduces loss learning rate, 
                # but save vram across extreamly large backpropagation steps
                if self.bptt_truncated_learning:
                    prv_shift_states = states.shift_states.clone().detach().requires_grad_(False)
                    prv_wkv_states = states.wkv_states.clone().detach().requires_grad_(False)
                else:
                    prv_shift_states = states.shift_states
                    prv_wkv_states = states.wkv_states
                
                # We use a dummy masked token 0, to do additional dummy checkpoint/forward/backprop when needed
                # for each additional call after the current "segment_count" max
                if i <= segment_count - 1:
                    cur_idx = idx[:, i * segment_size:(i + 1) * segment_size]
                    cur_tar = targets[:, i * segment_size:(i + 1) * segment_size]
                    cur_msk = seq_mask[:, i * segment_size:(i + 1) * segment_size]
                else:
                    cur_idx = dummy_2d_zero
                    cur_tar = dummy_2d_zero
                    cur_msk = dummy_2d_zero

                # Segmented learning, applies the forward/pass over each chunk seperately
                segment_loss, new_shift_states, new_wkv_states, steps = checkpointed_step(
                    cur_idx,
                    cur_tar,
                    cur_msk,
                    torch.tensor(0, dtype=self.emb.weight.dtype, device=cur_device).requires_grad_(True),
                    prv_shift_states,
                    prv_wkv_states,
                    steps,
                )
                states = BlockStateList(new_shift_states, new_wkv_states)

                # # Keep the segment loss (for backpassing in reverse)
                # segment_loss_arr[i] = segment_loss

                # Perform the backward pass accordingly, for valid segments (besides the last segment)
                # In this version, we do backward passes together the forward passes in the main segment loop
                # Instead of after all segment losses are computed
                if i >= start_learning_segment and i < start_learning_segment + backward_segment_count:
                    # The learning loss, should be normalized against the accumulation steps
                    # as we are bypassing the pytorch lightning normalization
                    # https://lightning.ai/docs/pytorch/2.0.4/common/lightning_module.html#backward
                    learning_loss = segment_loss / gradient_accumulation_steps

                    # Perform the backward pass accordingly, for valid segments (besides the last segment)
                    if i == start_learning_segment + backward_segment_count - 1:
                        # This is the last backward pass, we let the default pytorch lightning handle the backward pass
                        # and return the segment loss as part of the total loss
                        total_loss = total_loss + segment_loss
                    else:
                        # Undocumented multiple backward pass support
                        # https://github.com/Lightning-AI/lightning/blob/678f642808c54e4c490caee4df5d357301c976bb/tests/trainer/optimization/test_manual_optimization.py#L251
                        self.manual_backward(learning_loss, optimizer, retain_graph=True)
            
                        # Accumulate without gradient, as we already did the backward pass
                        total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)
                else:
                    # Even if its not the segments we use for backward pass, we still need to accumulate the loss
                    total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)
                
                # GC collect unused memory
                # gc.collect()
                # torch.cuda.empty_cache()

            # # Lets backpass the respective segments, in reverse
            # # (including dummy backpass)
            # for i in range(forward_segment_count-1, -1, -1):
            #     # Get the segment loss
            #     segment_loss = segment_loss_arr[i]
            #
            #     # Compute the backward pass for the segment
            #     if i >= start_learning_segment and i < start_learning_segment + backward_segment_count:
            #         # The learning loss, should be normalized against the accumulation steps
            #         # as we are bypassing the pytorch lightning normalization
            #         # https://lightning.ai/docs/pytorch/2.0.4/common/lightning_module.html#backward
            #         learning_loss = segment_loss / gradient_accumulation_steps
            #
            #         # Perform the backward pass accordingly, for valid segments (besides the start_learning_segment)
            #         if i > start_learning_segment:
            #             # Undocumented multiple backward pass support
            #             # https://github.com/Lightning-AI/lightning/blob/678f642808c54e4c490caee4df5d357301c976bb/tests/trainer/optimization/test_manual_optimization.py#L251
            #             self.manual_backward(learning_loss, optimizer, retain_graph=True)
            #
            #             # Accumulate without gradient, as we already did the backward pass
            #             total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)
            #         else:
            #             # This is the last backward pass, we let the default pytorch lightning handle the backward pass
            #             # and return the segment loss as part of the total loss
            #             total_loss = total_loss + segment_loss
            #     else:
            #         # Even if its not the segments we use for backward pass, we still need to accumulate the loss
            #         total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)
            #
            #    # GC collect unused memory
            #    gc.collect()
            #    # torch.cuda.empty_cache()
        else:

            # Normal operations without BPTT
            segment_size = self.ctx_len
            for i in range(segment_count):
                if i < segment_count-1 and is_training_run:
                    total_loss, new_shift_states, new_wkv_states, steps = deepspeed_checkpoint(
                        checkpointed_step,
                        idx[:, i * segment_size:(i + 1) * segment_size],
                        targets[:, i * segment_size:(i + 1) * segment_size],
                        seq_mask[:, i * segment_size:(i + 1) * segment_size],
                        total_loss,
                        states.shift_states,
                        states.wkv_states,
                        steps,
                    )
                else:
                    total_loss, new_shift_states, new_wkv_states, steps = checkpointed_step(
                        idx[:, i * segment_size:(i + 1) * segment_size],
                        targets[:, i * segment_size:(i + 1) * segment_size],
                        seq_mask[:, i * segment_size:(i + 1) * segment_size],
                        total_loss,
                        states.shift_states,
                        states.wkv_states,
                        steps,
                    )

                states = BlockStateList(new_shift_states, new_wkv_states)
                gc.collect()
                # torch.cuda.empty_cache()

        # Wandb logging only, if an active run exists (only applies for training)
        if wandb.run is not None and is_training_run:
            global_rank = self.global_rank
            global_device_count = self.trainer.num_devices * self.trainer.num_nodes

            # Increment the counting tokens, and log it accordingly
            self._counting_tokens += T

            # Log the line values
            wandb.log({
                'global_rank': global_rank, 
                'real_ctx_len': T, 
                'train/loss': total_loss,
                f'perf/tokens_total.gpu.{global_rank}': self._counting_tokens,
                f'perf/tokens_per_sec.gpu.{global_rank}': self._counting_tokens / max(time.time() - self._counting_time_start, 1),
                'substep': (batch_idx * global_device_count + global_rank),
                'trainer/global_step':self.global_step,
                'trainer/learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
                'batchidx': batch_idx
            })

        return total_loss

    @TCompileBaseline
    def training_step(self, batch, batch_idx):
        total_loss = self.compute_loss(batch, batch_idx, True)

        self.log('train/loss', total_loss, prog_bar=True)
        # If set - forces the above train/loss log line to always be on a new line
        if self.substep_logging:
            print("")
        
        if self.substep_cuda_cache_clear:
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss

    @TCompileBaseline
    def validation_step(self, batch, batch_idx):
        total_loss = self.compute_loss(batch, batch_idx, False)
        self.log('validation/loss', total_loss, prog_bar=True, sync_dist=True)
        return total_loss

### ---
# SimpleRWKV, a wrapper for RWKV that allows for simple usage of the model
### ---

# SimpleRWKV specific imports
from transformers import PreTrainedTokenizerFast

# Current script dir
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../'))

# SimpleRWKV is a wrapper for RWKV that allows for simple usage of the model
#
# it is not meant to be highly performant, but rather a simple minimal way to run the RWKV trainer module
# in inference mode, and can be used to validate the model trainer code / its changes
class SimpleRWKV():

    def __init__(
            self,
            model_path: str,
            ctx_len:int = 1024,
            device:str = "cuda",
            dtype:str = "fp32"
        ):

        # Log the mismatch dtype
        if dtype != "fp32":
            print("[SimpleRWKV] Warning: dtype mismatch, only fp32 is supported (for now)")

        # Prepare the model config with the model path, and custom torch load
        model_config = {}
        model_config["load_model"] = model_path
        model_config["ctx_len"] = ctx_len

        # This feature depends on deepspeed
        model_config["grad_cp"] = False
        # model_config["_torch_load_state"] = loaded_state

        # Save the config settings
        self.ctx_len = ctx_len
        self.device = device

        # Lets actually load the model
        self.model = RWKV(**model_config)

        # Lets map it over to the respective device type
        # and set it to run as eval/inference mode
        self.model.to(device)
        self.model.eval()

        # Get the model detected vocab size
        vocab_size = self.model.vocab_size

        # The tokenizer object values
        self.fastTokenizer = None
        self.worldTokenizer = None

        # Setup the tokenizer
        if vocab_size == 50277:
            # Use the neox tokenizer
            tokenizer_file = os.path.join(SCRIPT_DIR,"./dataflow/20B_tokenizer.json")
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            self.fastTokenizer = tokenizer
        elif vocab_size == 65536:
            # Use the world tokenizer
            from .dataflow.trie_tokenizer import MT_TRIE_TOKENIZER
            world_tokenizer = MT_TRIE_TOKENIZER(os.path.join(SCRIPT_DIR, "./dataflow/rwkv_vocab_v20230424.txt"))
            self.worldTokenizer = world_tokenizer
        else:
            raise NotImplementedError(f"Unsupported vocab size ({vocab_size}) - custom tokenizer not supported")

    # Encoding strings
    def encode(self, text: str):
        if self.worldTokenizer != None:
            return self.worldTokenizer.encode(text)
        return self.fastTokenizer.encode(text)

    # Decoding strings
    def decode(self, tokens: list):
        if self.worldTokenizer != None:
            return self.worldTokenizer.decode(tokens)
        return self.fastTokenizer.decode(tokens)

    # Forwarding logic, withoout torch._no_grad() context
    def _forward(
            self, tokens, 
            stateObj = None,
            all_logits = False
        ):

        logits_arr = None
        token_len = len(tokens)

        # Get the shift/wkv state
        if stateObj is None:
            shift_states = None
            wkv_states = None
        else:
            shift_states = stateObj["shift_states"]
            wkv_states = stateObj["wkv_states"]
        
        # The all_logits array, if requested
        all_logits_arr = None

        # For each token, process the state, in batches up to ctx_len
        for i in range(0, token_len, self.ctx_len):
            # Token set
            token_set = tokens[i:i+self.ctx_len]

            # Check if tokens are already tensors
            batch_tokens = torch.tensor(
                token_set, 
                dtype=torch.long, device=self.device
            ).unsqueeze(0)
            
            # Compute the logits and state
            logits_arr, shift_states, wkv_states = self.model.forward(
                batch_tokens, shift_states, wkv_states
            )

            # Build the all_logits array
            if all_logits:
                if all_logits_arr is None:
                    all_logits_arr = logits_arr[0]
                else:
                    all_logits_arr = torch.cat([all_logits_arr, logits_arr[0]], dim=0)

        # Return the logits and state
        if all_logits:
            return all_logits_arr, { "shift_states": shift_states, "wkv_states": wkv_states }
        else:
            return logits_arr[0][-1], { "shift_states": shift_states, "wkv_states": wkv_states }
    
    # Forwarding logic, with torch._no_grad() context
    def forward(
            self, tokens:list, 
            stateObj = None,
            all_logits = False
        ):
        with torch.no_grad():
            return self._forward(tokens, stateObj, all_logits)

    # Sampling logits
    def sample_logits(
            self, logits, 
            prv_tokens=[0], 
            temperature=1.0, top_p=0.9,
            token_ban: list = []
            ):
        # Copy to CPU first
        logits = logits.cpu()

        # Max negative float
        max_neg = -torch.finfo(torch.float).max

        # Apply token ban
        for x in token_ban:
            logits[x] = max_neg
        
        # Remove NaNs from logits
        for x in range(len(logits)):
            if torch.isnan(logits[x]):
                logits[x] = max_neg

        # Handle sampling with temperature
        if temperature > 0.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs = torch.sort(probs, descending=True)[0]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return out
        else: 
            # Since the tokenizer sample does not support temp==0
            # we handle this case ourself, by fining the top token
            return torch.argmax(logits, dim=-1).item()

    # Completion API
    def completion(self, 
            prompt, 
            max_tokens: int = 32,
            temperature: float = 1.0,
            top_p: float = 0.9,
            token_ban: list = [],
            start_state = None,
            stream_to_stdout: bool = False,
        ):
        # Encode the context, if its a string
        if isinstance(prompt, str):
            enc = self.encode(prompt)
        # Check if the prompt is a list of tokens
        elif isinstance(prompt, list):
            enc = prompt
        else:
            raise ValueError("Prompt must be a string or a list of tokens")

        # Keep track of the logits and state
        logits = None
        stateObj = start_state

        # For each token, process the state
        logits, stateObj = self.forward(enc, stateObj)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Generate each token
        out_tokens = []
        for i in range(max_tokens):
            ttt = self.sample_logits(
                logits, 
                # prv_tokens=full_tokens,
                temperature=temperature, top_p=top_p,
                token_ban=token_ban
            )
            
            # Append the token
            out_tokens.append(ttt)
            # full_tokens.append(ttt)
            if stream_to_stdout:
                print(self.decode([ttt]), end="", flush=True)

            # Perform the forward pass
            logits, stateObj = self.forward([ttt], stateObj)

        # Decode the tokens
        out_str = self.decode(out_tokens)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Return the output string, and state
        return out_str, stateObj
