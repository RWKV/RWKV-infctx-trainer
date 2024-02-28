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
class RWKV_TimeMix6_0x_Upgraded(JITModClass):

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

            # fancy time_mix
            self.time_mix_x = nn.Parameter(torch.ones_like(ddd)) #self.time_mix_x = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_w = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D = n_embd

            TIME_MIX_EXTRA_DIM = 32
            self.time_tm_w1 = nn.Parameter(torch.empty(n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(-1e-4, 1e-4))
            self.time_tm_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, n_embd))
            W_MIX_EXTRA_DIM = max(64, D // 16)
            self.time_td_w1 = nn.Parameter(torch.empty(D, W_MIX_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_td_w2 = nn.Parameter(torch.zeros(W_MIX_EXTRA_DIM, D))
            self.time_td_mult = nn.Parameter(torch.ones(D))
            # W_COMMON_EXTRA_DIM = max(32, D // 32)
            # self.time_tc_w1 = nn.Parameter(torch.empty(D, W_COMMON_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            # self.time_tc_w2 = nn.Parameter(torch.zeros(W_COMMON_EXTRA_DIM, D))
            # self.time_tc_mult = nn.Parameter(torch.ones(D))
            U_MIX_EXTRA_DIM = max(32, D // 32)
            self.time_tf_w1 = nn.Parameter(torch.empty(D, U_MIX_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_tf_w2 = nn.Parameter(torch.zeros(U_MIX_EXTRA_DIM, D))
            self.time_tf_mult = nn.Parameter(torch.ones(D))

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
        #if self.use_cuda is True:
        #   return self._forward_cuda(x, last_state)
        
        # Run without cuda (cpu mode, etc)
        return self._forward_nocuda_optimized(x, last_state)

    # def _forward_cuda(self, x, last_state: tuple[torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:
    #     shift_state_out = x[:,-1]

    #     assert(x.size(-2) % self.chunk_len == 0)

    #     # Get the x sizing
    #     B, T, C = x.size()
    #     H = self.n_head
    #     K = self.head_size
    #     V = K

    #     xprev = torch.concat((last_state[0].unsqueeze(1), x[:, :-1]), dim=1)

    #     dx = x - xprev

    #     xxx = xprev + dx * self.time_mix_x
    #     xxx = torch.tanh(xxx @ self.time_tm_w1).view(B*T, 5, -1).transpose(0, 1)
    #     xxx = torch.bmm(xxx, self.time_tm_w2).view(5, B, T, -1)
    #     mw, mk, mv, mr, mg = xxx.unbind(dim=0)

	# 	# Get the xk, xv, xr, xg, xw, and rkvg
    #     xk = xprev + dx * (self.time_mix_k + mk)
    #     xv = xprev + dx * (self.time_mix_v + mv)
    #     xr = xprev + dx * (self.time_mix_r + mr)
    #     xg = xprev + dx * (self.time_mix_g + mg)
    #     xw = xprev + dx * (self.time_mix_w + mw)

    #     r = self.receptance(xr)
    #     k = self.key(xk)
    #     v = self.value(xv)
    #     g = F.silu(self.gate(xg))

    #     ww = torch.tanh(xw @ self.time_td_w1) @ self.time_td_w2
    #     w = self.time_decay.view(1, 1, C) + ww
    #     u = self.time_faaaa

    #     # Logits and state
    #     wkv_state = last_state[1].to(r.dtype)

    #     # Perform the cuda forward pass
    #     x_logits = RUN_WKV6_CUDA(
    #         B, T, C, H, 
    #         #state, # FIXME - allow state to be passed in
    #         r, k, v, 
    #         w, 
    #         u
    #     )

    #     x_logits = x_logits.view(-1, C)
    #     x_logits = self.ln_x(x_logits).view(B, T, C)
    #     x_logits = self.output(x_logits * g)

    #     # Return the logits and the state

    #     # FIXME - state output is junk here
    #     return (x_logits, (shift_state_out,wkv_state))
            
    def _forward_nocuda_optimized(self, x, last_state: tuple[torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:
        shift_state_out = x[:,-1]

        assert(x.size(-2) % self.chunk_len == 0)

        # Get the x sizing
        B, T, C = x.size()
        H = self.n_head
        K = self.head_size
        V = K

        # Perform the tokenshift, and get the respective state
        xprev = torch.concat((last_state[0].unsqueeze(1), x[:, :-1]), dim=1)

        dx = x - xprev

        xxx = xprev + dx * self.time_mix_x
        xxx = torch.tanh(xxx @ self.time_tm_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_tm_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

		# Get the xk, xv, xr, xg, xw, and rkvg
        xk = xprev + dx * (self.time_mix_k + mk)
        xv = xprev + dx * (self.time_mix_v + mv)
        xr = xprev + dx * (self.time_mix_r + mr)
        xg = xprev + dx * (self.time_mix_g + mg)
        xuw = xprev + dx * (self.time_mix_w + mw)

        r = self.receptance(xr).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(xk).view(B, T, H, K).transpose(1, 2)      # BHTK
        v = self.value(xv).view(B, T, H, V).transpose(1, 2)    # BHTV
        g = F.silu(self.gate(xg))

        w = self.time_decay.float().view(1,H,1,K)
        w = w + self.time_td_mult.view(1, H, 1, K) * (torch.tanh(xuw @ self.time_td_w1) @ self.time_td_w2).view(B, T, H, K).transpose(1, 2) # BHTK
        w = torch.exp(-torch.exp(w))

        u_kaxis = self.time_faaaa.view(1,H,1,K).to(r.dtype)

        # data-dependent U_vaxis and Wcommon
        ulora = torch.tanh((xuw @ self.time_tf_w1) @ self.time_tf_w2 ).view(B, T, H, V).transpose(1, 2) # BHTK
        u_vaxis = self.time_tf_mult.view(1, H, 1, V) * ulora.pow(2)
        u_vaxis = u_vaxis.to(r.dtype)
        #anti_u_vaxis_lora = torch.tanh((xuw @ self.time_tc_w1) @ self.time_tc_w2).view(B, T, H, V).transpose(1, 2) # BHTK
        anti_u_vaxis = 1.0 # + self.time_tc_mult.view(1, H, 1, V) * anti_u_vaxis_lora.pow(2)

        # Logits and state
        wkv_state = last_state[1].to(r.dtype)

        x_logits, wkv_state = rwkv_inner(r, k, v, w, u_kaxis, wkv_state, self.chunk_len, self.precision) 

        x_logits = x_logits * anti_u_vaxis + v * u_vaxis

        x_logits = x_logits.transpose(1,2).reshape(B,T,C)

        # Reshape and normalize the logits
        x_logits = x_logits.view(-1, C)
        x_logits = self.ln_x(x_logits).view(B, T, C)
        x_logits = self.output(x_logits * g)

        # Return the logits and the state
        return (x_logits, (shift_state_out,wkv_state))
    