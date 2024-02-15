# Dependencies
from .CoreDependencies import *
from .OptimizedOps import modified_lerp
from .rwkv_inner import rwkv_inner
import os

from .rwkv_inner import rwkv_inner

# Current code file path
code_file_path = os.path.realpath(__file__)
code_dir = os.path.dirname(code_file_path)

### ---
# Special WKV6 CUDA kernel handling
### ---

# the cuda kernel (if its used)
global wkv6_cuda_kernel
wkv6_cuda_kernel = None

# WKV6_CUDA autograd module
class WKV6_CUDA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, 
                B:int, T:int, C:int, H:int, 
                r:torch.Tensor, k:torch.Tensor, 
                v:torch.Tensor, w:torch.Tensor, 
                u:torch.Tensor):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            #assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda_kernel.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda_kernel.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

@TCompileDisable 
@torch.jit.ignore
def RUN_WKV6_CUDA(
    B:int, T:int, C:int, H:int, 
    r:torch.Tensor, k:torch.Tensor, 
    v:torch.Tensor, w:torch.Tensor, 
    u:torch.Tensor):
    return WKV6_CUDA.apply(B, T, C, H, r, k, v, w, u)

# RWKV TimeMix module
class RWKV_TimeMix7_0(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, chunk_len:int = 128, precision:int = 64):
        super().__init__()
        
        self.dim_att = dim_att
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.layer_id = layer_id

        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8

        # V5-R4 changes
        # https://github.com/BlinkDL/RWKV-LM/commit/5aab658f945ba80745d36c2ab411fb43df3a74f9    
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            self.time_x_maa = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_r_maa = nn.Parameter(1 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_w_maa = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_k_maa = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_v_maa = nn.Parameter(1 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_g_maa = nn.Parameter(1 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            TIME_MIX_EXTRA_DIM = 32
            self.time_tm_w1 = nn.Parameter(torch.empty(n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(-1e-4, 1e-4))
            self.time_tm_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, n_embd))
            W_MIX_EXTRA_DIM = 64
            self.time_td_w1 = nn.Parameter(torch.empty(n_embd, W_MIX_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_td_w2 = nn.Parameter(torch.zeros(W_MIX_EXTRA_DIM, n_embd))
            D_GATE_LORA = 64
            self.time_gate_w1 = nn.Parameter(torch.empty(n_embd, D_GATE_LORA).uniform_(-1e-4, 1e-4))
            self.time_gate_w2 = nn.Parameter(torch.zeros(D_GATE_LORA, n_embd).uniform_(-1e-4, 1e-4))

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
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
        self.ln_x = nn.GroupNorm(n_head, dim_att, eps=(1e-5)*(self.head_size_divisor**2))
        
        # Preload the CUDA kernel if needed
        self.use_cuda = False
        self._preload_cuda()
        
        self.chunk_len = chunk_len
        self.precision = precision

    def _preload_cuda(self):
        global wkv6_cuda_kernel, RWKV_NO_CUDA

        # Skip preload if cuda is disabled
        if RWKV_NO_CUDA is True:
            self.use_cuda = False
            return

        # Load cuda if needed
        if wkv6_cuda_kernel is None:
            # Head sizing
            HEAD_SIZE = self.head_size

            # Log the compillation block
            print("---")
            print(f"[RWKV.TimeMix] Compiling CUDA kernel with HEAD_SIZE={HEAD_SIZE}")

            wkv6_cuda_kernel = torch.utils.cpp_extension.load(
                name="wkv6", 
                sources=[
                    os.path.join(code_dir, "cuda/wkv6_op.cpp"),
                    os.path.join(code_dir, "cuda/wkv6_cuda.cu"),
                ],
                verbose=True, 
                extra_cuda_cflags=[
                    "-res-usage", 
                    "--use_fast_math", 
                    "-O3", "-Xptxas -O3", 
                    "--extra-device-vectorization", 
                    f"-D_N_={HEAD_SIZE}", 
                    f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"
                ]
            )

            # Close log the compillation block
            print(f"[RWKV.TimeMix6_0] CUDA kernel compiled & loaded globally")
            print("---")
        
        # Initialize the cuda kernel
        self.use_cuda = True

    # forwarding time mix given the model weights and the input tokens and states.
    #
    # Given:
    # - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
    # - Last states containing of shape [
    #       [batch_size, state_size] ## Channel mix state,
    #       [batch_size, n_head, head_size, head_size] ## WKV state
    #   ]
    #
    # Returns a pair 
    # - of output embedding of shape [batch_size, seq_len, embedding_size]
    # - and the last output state of shape [
    #       [batch_size, state_size] ## Channel mix state,
    #       [batch_size, n_head, head_size, head_size] ## WKV state
    #   ]
    @JITModMethod
    def forward(self, x, last_state: tuple[torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:
        # Run with cuda
        if self.use_cuda is True:
           return self._forward_cuda(x, last_state)
        
        # Run without cuda (cpu mode, etc)
        return self._forward_nocuda_optimized(x, last_state)

    def _forward_cuda(self, x, last_state: tuple[torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:
        shift_state_out = x[:,-1]

        assert(x.size(-2) % self.chunk_len == 0)

        # Get the x sizing
        B, T, C = x.size()
        H = self.n_head
        K = self.head_size
        V = K

        dxprev = torch.concat((last_state[0].unsqueeze(1), x[:, :-1]), dim=1) - x
        xxx = x + dxprev * self.time_x_maa
        xxx = torch.tanh(xxx @ self.time_tm_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_tm_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

		# Get the xk, xv, xr, xg, xw, and rkvg
        xk = x + dxprev * (self.time_k_maa + mk)
        xv = x + dxprev * (self.time_v_maa + mv)
        xr = x + dxprev * (self.time_r_maa + mr)
        xg = x + dxprev * (self.time_g_maa + mg)
        xw = x + dxprev * (self.time_w_maa + mw)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(torch.tanh(xg @ self.time_gate_w1) @ self.time_gate_w2)

        ww = torch.tanh(xw @ self.time_td_w1) @ self.time_td_w2
        w = self.time_decay.view(1, 1, C) + ww
        u = self.time_faaaa

        # Logits and state
        wkv_state = last_state[1].to(r.dtype)

        # Perform the cuda forward pass
        x_logits = RUN_WKV6_CUDA(
            B, T, C, H, 
            #state, # FIXME - allow state to be passed in
            r, k, v, 
            w, 
            u
        )

        x_logits = x_logits.view(-1, C)
        x_logits = self.ln_x(x_logits / self.head_size_divisor).view(B, T, C)
        x_logits = self.output(x_logits * g)

        # Return the logits and the state

        # FIXME - state output is junk here
        return (x_logits, (shift_state_out,wkv_state))
    
    def _forward_nocuda_optimized(self, x, last_state: tuple[torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:
        shift_state_out = x[:,-1]

        assert(x.size(-2) % self.chunk_len == 0)

        # Get the x sizing
        B, T, C = x.size()
        H = self.n_head
        K = self.head_size
        V = K

        dxprev = torch.concat((last_state[0].unsqueeze(1), x[:, :-1]), dim=1) - x
        xxx = x + dxprev * self.time_x_maa
        xxx = torch.tanh(xxx @ self.time_tm_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_tm_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

		# Get the xk, xv, xr, xg, xw, and rkvg
        xk = x + dxprev * (self.time_k_maa + mk)
        xv = x + dxprev * (self.time_v_maa + mv)
        xr = x + dxprev * (self.time_r_maa + mr)
        xg = x + dxprev * (self.time_g_maa + mg)
        xw = x + dxprev * (self.time_w_maa + mw)

        r = self.receptance(xr).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(xk).view(B, T, H, K).transpose(1, 2)      # BHTK
        v = self.value(xv).view(B, T, H, V).transpose(1, 2)    # BHTV
        g = F.silu(torch.tanh(xg @ self.time_gate_w1) @ self.time_gate_w2)

        w = self.time_decay.float().view(1,H,1,K)
        w = w + (torch.tanh(xw @ self.time_td_w1) @ self.time_td_w2).view(B, T, H, K).transpose(1, 2) # BHTK
        w = torch.exp(-torch.exp(w))

        u = self.time_faaaa.view(1,H,1,K).to(r.dtype)

        # Logits and state
        wkv_state = last_state[1].to(r.dtype)

        x_logits, wkv_state = rwkv_inner(r, k, v, w, u, wkv_state, self.chunk_len, self.precision) 
        x_logits = x_logits.transpose(1,2).reshape(B,T,C)

        # Reshape and normalize the logits
        x_logits = x_logits.view(-1, C)
        x_logits = self.ln_x(x_logits / self.head_size_divisor).view(B, T, C)
        x_logits = self.output(x_logits * g)

        # Return the logits and the state
        return (x_logits, (shift_state_out,wkv_state))
    