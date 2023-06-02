########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
from typing import List, Optional
import numpy as np
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


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
    wkv_state[:, :, -1] = 1e-38
    token_shift_state = torch.zeros((B, C), device=device, dtype=dtype)
    return BlockState(TimeMixState(token_shift_state, wkv_state),
                      ChannelMixState(token_shift_state))


########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = int(os.environ["RWKV_T_MAX"])
assert os.environ[
    "RWKV_MY_TESTING"] == "", "a/b variants not supported in InfCtx"

from torch.utils.cpp_extension import load

assert os.environ[
    "RWKV_FLOAT_MODE"] == "bf16", "InfCtx currently only supports BF16"
load(name=f"wkv_{T_MAX}_bf16",
     sources=["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"],
     verbose=True,
     extra_cflags=["-std=c++17", "-O3", f"-DTmax={T_MAX}"],
     extra_cuda_cflags=[
         "-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60",
         "--use_fast_math", "-O3", "-Xptxas -O3",
         "--extra-device-vectorization", f"-DTmax={T_MAX}"
     ],
     is_python_module=False)


########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################



class RWKV_TimeMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for h in range(args.dim_att):
                decay_speed[h] = -5 + 8 * (h / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(args.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        if 'a' in os.environ["RWKV_MY_TESTING"]:
            self.register_buffer("att_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
            d_qkv = args.n_embd // 16
            self.qq = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.kk = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.vv = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.oo = nn.Linear(d_qkv, args.n_embd, bias=False)
            with torch.no_grad():
                self.time_mix_qq = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_kk = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_vv = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)

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
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x, last_state: ChannelMixState):
        xx = torch.concat(
            (last_state.token_shift_state.unsqueeze(1), x[:, :-1]), dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, ChannelMixState(x[:, -1])


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_TimeMix(args, layer_id)
        self.ffn = RWKV_ChannelMix(args, layer_id)

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


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = args.n_embd * 4

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def configure_optimizers(self):
        args = self.args
        if args.layerwise_lr > 0:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif "time_decay" in n:
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
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
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [
                {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0},
            ]

        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx, last_states: List[BlockState]):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        new_states = []
        for block, last_state in zip(self.blocks, last_states):
            if args.grad_cp == 1:
                x, new_state = deepspeed.checkpointing.checkpoint(
                    block, x, last_state)
            else:
                x, new_state = block(x, last_state)
            new_states.append(new_state)

        x = self.ln_out(x)

        x = self.head(x)

        return x, new_states

    def training_step(self, batch, batch_idx):
        args = self.args

        seq = batch['input_ids']
        assert isinstance(seq, torch.Tensor) and seq.ndim == 2
        idx, targets = seq[:, :-1], seq[:, 1:]

        B, T = idx.shape
        C = args.n_embd

        states = [init_block_state(B, C, seq.device, self.emb.weight.dtype)
                  ] * args.n_layer

        def checkpointed_step(idx, targets, prev_loss, last_states):
            logits, new_states = self(idx, last_states)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = L2Wrap.apply(loss, logits, B * T)
            return prev_loss + loss, new_states

        total_loss = torch.tensor(0, dtype=self.emb.weight.dtype)
        for i in range(math.ceil(T / args.ctx_len)):
            print(f'step {i}')
            if i != math.ceil(T / args.ctx_len) - 1:
                total_loss, states = deepspeed.checkpointing.checkpoint(
                    checkpointed_step,
                    idx[:, i * args.ctx_len:(i + 1) * args.ctx_len],
                    targets[:, i * args.ctx_len:(i + 1) * args.ctx_len],
                    total_loss,
                    states,
                )
            else:
                total_loss, states = checkpointed_step(
                    idx[:, i * args.ctx_len:(i + 1) * args.ctx_len],
                    targets[:, i * args.ctx_len:(i + 1) * args.ctx_len],
                    total_loss,
                    states,
                )
        return total_loss

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        args = self.args

        seq = batch['input_ids']
        assert isinstance(seq, torch.Tensor) and seq.ndim == 1
        #!FIXME: temporary workaround!!! Arbitrary length should be supported later
        if len(seq) > T_MAX:
            seq = seq[:T_MAX]

        T, = idx.shape
        C = args.n_embd

        state = [init_block_state(1, C, seq.device, seq.dtype)] * args.n_layer

        idx, target = seq[:-1], seq[1:]
        loss = np.array([], dtype=np.float32)
        for i in range(math.ceil(T / args.ctx_len)):
            logit, state = self(
                idx[i * args.ctx_len:(i + 1) * args.ctx_len].view(1, -1),
                state)
            piece_loss: np.ndarray = F.cross_entropy(
                logit,
                target[i * args.ctx_len:(i + 1) * args.ctx_len],
                reduction='none').float().cpu().numpy()
            loss = np.concatenate((loss, piece_loss))

        print("validation loss shape: ", loss.shape)
        exp_mean_loss = []
        for i in range(8, math.ceil(math.log2(loss.shape[0]))):
            exp_mean_loss.append([i, loss[:min(len(loss), 2**i)].mean()])

        print(exp_mean_loss)

        import wandb
        table = wandb.Table(data=exp_mean_loss,
                            columns=["length", "cross_entropy_loss"])
        wandb.log({
            f"validation/loss_curve/{self.real_epoch}/{batch_idx}":
            wandb.plot.line(table,
                            "length",
                            "cross_entropy_loss",
                            title="Loss Curve"),
        })

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    for kk in [".att.key.", ".att.receptance.", ".att.output.", ".att.key.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()
        torch.cuda.empty_cache()
        return m
