########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math
from random import randint
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

RWKV_JIT_ON = True

if RWKV_JIT_ON:
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
else:
    MyModule = nn.Module
    MyFunction = lambda x: x


class TimeMixState:

    def __init__(self, token_shift_state: torch.Tensor,
                 wkv_state: torch.Tensor):
        self.token_shift_state = token_shift_state
        self.wkv_state = wkv_state


class ChannelMixState:

    def __init__(self, token_shift_state: torch.Tensor):
        self.token_shift_state = token_shift_state


class BlockState:

    def __init__(self, time_mix_state: torch.Tensor,
                 channel_mix_state: torch.Tensor):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


def init_block_state(B, C, device, dtype):
    wkv_state = torch.zeros((B, C, 3), device=device, dtype=torch.float)
    wkv_state[:, :, -1] = -1e38
    token_shift_state = torch.zeros((B, C), device=device, dtype=dtype)
    return BlockState(TimeMixState(token_shift_state, wkv_state),
                      ChannelMixState(token_shift_state))


from torch.utils.cpp_extension import load

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################


class RWKV_TimeMix(MyModule):

    def __init__(self, layer_id, n_layer, n_embd, dim_att):
        super().__init__()

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for h in range(dim_att):
                decay_speed[h] = -5 + 8 * (h /
                                           (dim_att - 1))**(0.7 +
                                                            1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1
                                   for i in range(dim_att)]) * 0.5
            self.time_first = nn.Parameter(
                torch.ones(dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(
                torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(
                torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, dim_att, bias=False)
        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)

    @MyFunction
    def forward(self, x, last_state: TimeMixState):
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = torch.concat(
            (last_state.token_shift_state.unsqueeze(1), x[:, :-1]), dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        sr = torch.sigmoid(r)

        y, new_wkv_state = torch.ops.rwkv.wkv(self.time_decay, self.time_first,
                                              k, v, last_state.wkv_state)
        return self.output(sr * y), TimeMixState(x[:, -1], new_wkv_state)


########################################################################################################


class RWKV_ChannelMix(MyModule):

    def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
        super().__init__()

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

    @MyFunction
    def forward(self, x, last_state: ChannelMixState):
        xx = torch.concat(
            (last_state.token_shift_state.unsqueeze(1), x[:, :-1]), dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return (torch.sigmoid(self.receptance(xr)) * kv,
                ChannelMixState(x[:, -1]))


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, dim_att, dim_ffn):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, dim_att)
        self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)

    def forward(self, x, last_state: BlockState):
        if self.layer_id == 0:
            x = self.ln0(x)
        att_out, att_state = self.att(
            self.ln1(x),
            last_state.time_mix_state,
        )
        x = x + att_out
        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state.channel_mix_state,
        )
        x = x + ffn_out
        return x, BlockState(att_state, ffn_state)


class L2Wrap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, y, token_amount):
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
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
        return (grad_output, gy, None)


class RWKV(L.LightningModule):

    def __init__(self,
                 ctx_len: int,
                 ctx_len_cutoffs: List[int],
                 ctx_len_warmup_steps: List[int],
                 n_embd: int,
                 n_layer: int,
                 vocab_size: int,
                 grad_cp: bool,
                 lr_init: float,
                 warmup_steps: int,
                 beta1: float,
                 beta2: float,
                 adam_eps: float,
                 layerwise_lr: bool = True,
                 dim_att: Optional[int] = None,
                 dim_ffn: Optional[int] = None,
                 load_model: Optional[str] = None):
        super().__init__()
        self.ctx_len = ctx_len
        self.ctx_len_cutoffs = ctx_len_cutoffs
        self.ctx_len_warmup_steps = ctx_len_warmup_steps
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layerwise_lr = layerwise_lr
        self.grad_cp = grad_cp
        self.lr_init = lr_init
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps

        dim_att = dim_att or n_embd
        dim_ffn = dim_ffn or n_embd * 4

        self.emb = nn.Embedding(vocab_size, n_embd)

        load(name=f"wkv_{self.ctx_len}_bf16",
             sources=["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"],
             verbose=True,
             extra_cflags=["-std=c++17", "-O3", f"-DTmax={self.ctx_len}"],
             extra_cuda_cflags=[
                 "-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60",
                 "--use_fast_math", "-O3", "-Xptxas -O3",
                 "--extra-device-vectorization", f"-DTmax={self.ctx_len}"
             ],
             is_python_module=False)

        self.blocks = nn.ModuleList([
            Block(i, n_layer, n_embd, dim_att, dim_ffn) for i in range(n_layer)
        ])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.load_state_dict(torch.load(load_model, map_location='cpu'))

    def configure_optimizers(self):
        if self.layerwise_lr:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    lr_1x.add(n)
                elif "time_decay" in n:
                    lr_2x.add(n)
                elif "time_first" in n:
                    lr_3x.add(n)
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
                    "lr": 1.0 * self.lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 2.0 * self.lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.0,
                    "lr": 3.0 * self.lr_init
                },
            ]
        else:
            optim_groups = [
                {
                    "params": [p for n, p in self.named_parameters()],
                    "weight_decay": 0.0
                },
            ]

        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=self.lr_init,
                                         betas=(self.beta1, self.beta2),
                                         eps=self.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=False,
                                         weight_decay=0,
                                         amsgrad=False)
        else:
            optimizer = FusedAdam(optim_groups,
                                  lr=self.lr_init,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.adam_eps,
                                  bias_correction=True,
                                  adam_w_mode=False,
                                  weight_decay=0,
                                  amsgrad=False)

        if self.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.lr_init,
                warmup_max_lr=self.lr_init,
                warmup_num_steps=self.warmup_steps,
                warmup_type='linear')

            return optimizer, lr_scheduler
        else:
            return optimizer

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return "offload_optimizer" in cfg or "offload_parameters" in cfg
        return False

    def forward(self, idx, last_states: List[BlockState]):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        new_states = []
        for block, last_state in zip(self.blocks, last_states):
            if self.grad_cp:
                x, new_state = deepspeed.checkpointing.checkpoint(
                    block, x, last_state)
            else:
                x, new_state = block(x, last_state)
            new_states.append(new_state)

        x = self.ln_out(x)

        x = self.head(x)

        return x, new_states

    def compute_loss(self, batch, batch_idx, do_cutoff: bool):
        seq = batch['input_ids']
        assert isinstance(seq, torch.Tensor) and seq.ndim == 2
        seq_mask = batch['attention_mask']

        # Check if attent mask is set, if not initialize it
        if seq_mask is None or seq_mask.ndim != 2:
            seq_mask = torch.ones_like(seq[:, 1:])
        
        if do_cutoff:
            prev_step = 0
            for step, len_cut in zip(self.ctx_len_warmup_steps,
                                    self.ctx_len_cutoffs):
                if prev_step <= self.global_step < step and len_cut < seq.shape[1] - 1:
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

        idx, targets = seq[:, :-1], seq[:, 1:]

        B, T = idx.shape
        C = self.n_embd

        def checkpointed_step(idx, targets, mask, prev_loss, last_states,
                              prev_steps):
            logits, new_states = self(idx, last_states)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), reduction="none")
            submask = mask.view(-1)[:loss.shape[0]]
            loss = torch.sum(loss * submask) / torch.max(torch.sum(submask), torch.tensor(1, dtype=submask.dtype))
            loss = L2Wrap.apply(loss, logits, B * T)
            new_steps = prev_steps + idx.shape[1]
            new_loss = prev_loss * (prev_steps / new_steps) + loss * (
                1 - prev_steps / new_steps)
            return new_loss, new_states, new_steps

        total_loss = torch.tensor(0, dtype=self.emb.weight.dtype)
        steps = 0
        states = [
            init_block_state(B, C, seq.device, self.emb.weight.dtype)
        ] * self.n_layer
        for i in range(math.ceil(T / self.ctx_len)):
            if i != math.ceil(T / self.ctx_len) - 1:
                total_loss, states, steps = deepspeed.checkpointing.checkpoint(
                    checkpointed_step,
                    idx[:, i * self.ctx_len:(i + 1) * self.ctx_len],
                    targets[:, i * self.ctx_len:(i + 1) * self.ctx_len],
                    seq_mask[:, i * self.ctx_len:(i + 1) * self.ctx_len],
                    total_loss,
                    states,
                    steps,
                )
            else:
                total_loss, states, steps = checkpointed_step(
                    idx[:, i * self.ctx_len:(i + 1) * self.ctx_len],
                    targets[:, i * self.ctx_len:(i + 1) * self.ctx_len],
                    seq_mask[:, i * self.ctx_len:(i + 1) * self.ctx_len],
                    total_loss,
                    states,
                    steps,
                )

        # @TODO : Figure out how to check if wandb is enabled, and skip the wandb log accordingly
        wandb.log({'substep': batch_idx, 'real_ctx_len': T, 'train/loss': total_loss, 'trainer/global_step':self.global_step})

        return total_loss

    def training_step(self, batch, batch_idx):
        total_loss = self.compute_loss(batch, batch_idx, True)
        self.log('train/loss', total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss = self.compute_loss(batch, batch_idx, False)
        self.log('validation/loss', total_loss, prog_bar=True, sync_dist=True)
        return total_loss
