# Dependencies
from .CoreDependencies import *
from .OptimizedOps import modified_lerp
import os

# Current code file path
code_file_path = os.path.realpath(__file__)
code_dir = os.path.dirname(code_file_path)

### ---
# Special WKV5 CUDA kernel handling
### ---

# the cuda kernel (if its used)
global wkv5_cuda_kernel
wkv5_cuda_kernel = None

# WKV5_CUDA autograd module
class WKV5_CUDA(torch.autograd.Function):  

    # WKV5 forwarding process
    # NOTE: This will modify the state value as part of the forward process
    @staticmethod
    def forward(ctx, 
            B:int, T:int, C:int, H:int, 
            state:torch.Tensor, 
            r:torch.Tensor, k:torch.Tensor, 
            v:torch.Tensor, w:torch.Tensor, 
            u:torch.Tensor
        ):
        with torch.no_grad():
            # Save the sizing & dtype
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            dtype = r.dtype
            ctx.dtype = dtype

            # State and W is expected to be float32
            assert state.dtype == torch.float32
            assert w.dtype == torch.float32
            assert state.is_contiguous()
            assert w.is_contiguous()

            # Rest can be their respective types, but they are expected
            # to be consistent with each other
            assert dtype == k.dtype
            assert dtype == v.dtype
            assert dtype == u.dtype
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert u.is_contiguous()

            # Lets pre-compute the exp(-w) and exp(exp(-w))
            ew = (-torch.exp(w.float())).contiguous()
            eew = (torch.exp(ew)).contiguous()
            ctx.save_for_backward(r, k, v, eew, ew, u, state.clone())

            # Output logits
            y = torch.empty(B, T, C, device=r.device, dtype=dtype, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            
            # # Debugging y value is populated by cuda kernel
            # y = torch.zeros(B, T, C, device=r.device, dtype=dtype).contiguous() # .uniform_(-1, 1)
            # assert torch.sum(y) == 0, "Initial zero check"
            # assert not torch.isnan(y).any(), "Initial NaN check"

            # # Asserting non NaN valeus in inputs
            # assert not torch.isnan(state).any(), "Initial state NaN check"
            # assert not torch.isnan(r).any(), "Initial r NaN check"
            # assert not torch.isnan(k).any(), "Initial k NaN check"
            # assert not torch.isnan(v).any(), "Initial v NaN check"
            # assert not torch.isnan(w).any(), "Initial w NaN check"
            # assert not torch.isnan(u).any(), "Initial u NaN check"

            # Call the cuda kernel
            if dtype == torch.bfloat16:
                wkv5_cuda_kernel.forward_bf16(B, T, C, H, state, r, k, v, eew, u, y)
            elif dtype == torch.float16:
                wkv5_cuda_kernel.forward_fp16(B, T, C, H, state, r, k, v, eew, u, y)
            elif dtype == torch.float32:
                wkv5_cuda_kernel.forward_fp32(B, T, C, H, state, r, k, v, eew, u, y)
            else:
                raise ValueError(f"Unsupported dtype {dtype} for WKV5_CUDA")
            
            # # Assert output logits y is not zero, nor NaN
            # assert torch.sum(y) != 0, "Post kernel, non zero check"
            # assert not torch.isnan(y).any(), "Post kernel, NaN check"

            # Logits (without state)
            return y
    
    # WKV5 backward pass process
    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():

            # Get the sizing & dtype
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            dtype = ctx.dtype

            # GY dtype
            assert gy.dtype == dtype
            assert gy.is_contiguous()
            r, k, v, eew, ew, u, inState = ctx.saved_tensors

            # Initialize all the backward pass vars required
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=dtype, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=dtype, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=dtype, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=dtype, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=dtype, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            
            # Perform the backward pass
            if dtype == torch.bfloat16:
                wkv5_cuda_kernel.backward_bf16(B, T, C, H, inState, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
            elif dtype == torch.float16:
                wkv5_cuda_kernel.backward_fp16(B, T, C, H, inState, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
            elif dtype == torch.float32:
                wkv5_cuda_kernel.backward_fp32(B, T, C, H, inState, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
            else:
                raise ValueError(f"Unsupported dtype {dtype} for WKV5_CUDA")
            
            gw = torch.sum(gw, 0).view(H, C//H)
            gu = torch.sum(gu, 0).view(H, C//H)

            # # Log the shapes of gr-gu (debugging)
            # print(f"[WKV5_CUDA] gr.shape={gr.shape}, gk.shape={gk.shape}, gv.shape={gv.shape}, gw.shape={gw.shape}, gu.shape={gu.shape}")

            # Backprop values
            return (
                # B, T, C, H,
                None, None, None, None,
                # GState,
                None,
                # Gr, Gk, Gv, Gw, Gu
                gr, gk, gv, gw, gu
            )
       
@TCompileDisable 
def RUN_WKV5_CUDA(
    B:int, T:int, C:int, H:int, 
    state:torch.Tensor, 
    r:torch.Tensor, k:torch.Tensor, 
    v:torch.Tensor, w:torch.Tensor, 
    u:torch.Tensor
):
    return WKV5_CUDA.apply(B, T, C, H, state, r, k, v, w, u)

### ---
# TimeMix block class handling
### ---

# RWKV TimeMix module
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

        # Preload the CUDA kernel if needed
        self.use_cuda = False
        self._preload_cuda()

    def _preload_cuda(self):
        global wkv5_cuda_kernel, RWKV_NO_CUDA

        # Skip preload if cuda is disabled
        if RWKV_NO_CUDA is True:
            self.use_cuda = False
            return
        
        # Load cuda if needed
        if wkv5_cuda_kernel is None:
            # Head sizing
            HEAD_SIZE = self.head_size

            # Log the compillation block
            print("---")
            print(f"[RWKV.TimeMix] Compiling CUDA kernel with HEAD_SIZE={HEAD_SIZE}")

            # The cuda kernel
            wkv5_cuda_kernel = torch.utils.cpp_extension.load(
                name="wkv5",
                sources=[
                    os.path.join(code_dir, "cuda/wkv5_op.cpp"),
                    os.path.join(code_dir, "cuda/wkv5_cuda.cu"),
                ],
                verbose=True,
                extra_cuda_cflags=[
                    "-res-usage", 
                    "--use_fast_math", 
                    "-O3", "-Xptxas -O3", 
                    "--extra-device-vectorization", 
                    f"-D_N_={HEAD_SIZE}"
                ]
            )

            # Close log the compillation block
            print(f"[RWKV.TimeMix] CUDA kernel compiled & loaded globally")
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
    def forward(self, x, last_state: tuple[torch.Tensor,torch.Tensor]):
        # Run with cuda
        if self.use_cuda is True:
            return self._forward_cuda(x, last_state)
        
        # Run without cuda (cpu mode, etc)
        return self._forward_nocuda(x, last_state)
     
    def _forward_cuda(self, x, last_state: tuple[torch.Tensor,torch.Tensor]):
        # Get the x sizing
        B, TT, C = x.size()
        H = self.n_head

        # Perform the tokenshift, and get the respective state
        xx = torch.concat((last_state[0].unsqueeze(1), x[:, :-1]), dim=1)
    
        # Get the xk, xv, xr, xg, and rkvg
        xk = modified_lerp(x, self.time_mix_k, xx)
        xv = modified_lerp(x, self.time_mix_v, xx)
        xr = modified_lerp(x, self.time_mix_r, xx)
        xg = modified_lerp(x, self.time_mix_g, xx)

        r = self.receptance(xr)#.view(B, TT, self.n_head, 1, -1)
        k = self.key(xk)#.view(B, TT, self.n_head, -1, 1)
        v = self.value(xv)#.view(B, TT, self.n_head, 1, -1)
        g = F.silu(self.gate(xg))

        # Logits and state
        state = last_state[1].clone().to(torch.float32).contiguous()

        # Perform the cuda forward pass
        x_logits = RUN_WKV5_CUDA(
            B, TT, C, H, 
            state, 
            r, k, v, 
            self.time_decay.float(), 
            self.time_faaaa.to(r.dtype)
        )

        # Reshape and normalize the logits
        x_logits = self._x_logits_gate(x_logits, g)

        # Return the logits and the state
        return (x_logits, (x[:,-1],state))

    # Doing the forward pass withotu CUDA, this is currently rather slow
    # and is not recommended - but it works (awaiting future optimization)
    #
    # We intentionally split the forward pass into smaller chunks of 32
    # to ensure it processes at a resonable rate / vram usage
    @TCompileMax 
    @JITModMethod  
    def _forward_nocuda(self, x, last_state: tuple[torch.Tensor,torch.Tensor]):
        # Get the x sizing
        B, TT, C = x.size()

        # Logits to return
        x_logits = torch.zeros(B, TT, C, device=x.device, dtype=x.dtype)

        # Process in chunks
        chunk_len = 32
        for i in range(0, TT, chunk_len):
            # Get the chunk
            chunk = x[:, i:i+chunk_len]

            # Process the chunk
            chunk_logits, last_state = self._forward_nocuda_noChunking(chunk, last_state)

            # Store the chunk logits
            x_logits[:, i:i+chunk_len] = chunk_logits
        
        # Return the logits and the state
        return (x_logits, last_state)

    # The no chunking varient of forwarding without cuda
    @JITModMethod  
    def _forward_nocuda_noChunking(self, x, last_state: tuple[torch.Tensor,torch.Tensor]):
        # Get the x sizing
        B, TT, C = x.size()

        # Perform the tokenshift, and get the respective state
        xx = torch.concat((last_state[0].unsqueeze(1), x[:, :-1]), dim=1)
    
        # Get the xk, xv, xr, xg
        xk = modified_lerp(x, self.time_mix_k, xx)
        xv = modified_lerp(x, self.time_mix_v, xx)
        xr = modified_lerp(x, self.time_mix_r, xx)
        xg = modified_lerp(x, self.time_mix_g, xx)

        r = self.receptance(xr).view(B, TT, self.n_head, 1, -1)
        k = self.key(xk).view(B, TT, self.n_head, -1, 1)
        v = self.value(xv).view(B, TT, self.n_head, 1, -1)
        g = F.silu(self.gate(xg))

        # The WKV state to update
        if last_state[1] is None:
            wkv_state = torch.zeros((B, self.n_head, self.head_size, self.head_size)).to(r.dtype)
        else:
            wkv_state = last_state[1].clone().to(r.dtype)
        
        # # Compute attent and the initial output tensor
        # at = k @ v
        # u = self.time_faaaa.view(1,1,self.n_head, 1, -1)

        # # Slightly inefficent, but it works, lets compute all the tokens
        # w = self.time_decay.exp().neg().exp().reshape(1, self.n_head,-1,1)

        # out = (u * r) @ at
        # for t in range(TT):
        #     out[:,t] += r[:,t] @ wkv_state
            
        #     # We make a clone copy, so the previous object backprop state is tracked seperately
        #     wkv_state = wkv_state.clone()
        #     wkv_state *= w
        #     wkv_state += at[:,t]

        wkv_state, out = compute_wkv_state(
            k, v, r,
            self.time_faaaa,
            self.time_decay,
            wkv_state, 
            self.n_head, self.head_size,
            B, TT
        )

        # Compute the final x output
        x_logits = out.view(-1, C)
        x_logits = self.ln_x(x_logits / self.head_size_divisor).view(B, TT, C)
        x_logits = self.output(x_logits * g)

        # Return the logits and the state
        return (x_logits, (x[:,-1],wkv_state))
        # return x_logits, (x[:,-1],ms[-1])
    
    @TCompileMax 
    @JITModMethod  
    def _x_logits_gate(self, x_logits, gate):
        B, T, C = x_logits.size()
        x_logits = x_logits.view(-1, C)
        x_logits = self.ln_x(x_logits / self.head_size_divisor).view(B, T, C)
        return self.output(x_logits * gate)

def compute_wkv_state(
        k, v, r,
        time_faaaa: torch.nn.Parameter,
        time_decay: torch.nn.Parameter,
        wkv_state, 
        n_head:int, head_size:int,
        B:int, TT:int
    ):
    # Compute attent and the initial output tensor
    at = k @ v
    u = time_faaaa.view(1,1,n_head, 1, -1)

    # Slightly inefficent, but it works, lets compute all the tokens
    w = time_decay.exp().neg().exp().reshape(1, n_head,-1,1)

    out = (u * r) @ at
    for t in range(TT):
        out[:,t] += r[:,t] @ wkv_state
        
        # We make a clone copy, so the previous object backprop state is tracked seperately
        wkv_state = wkv_state.clone()
        wkv_state *= w
        wkv_state += at[:,t]

    return wkv_state, out


# @TCompileMax
# @JITFunction
# def x_logits_output_parsing(out_emb, head_size_divisor, B, TT, C, self_ln_x, self_output, g):
#     x_logits = out_emb.view(-1, C)
#     x_logits = self_ln_x(x_logits / head_size_divisor).view(B, TT, C)
#     return self_output(x_logits * g)
