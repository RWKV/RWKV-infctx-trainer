from .JitModClass import JITModClass, torch, nn, JITModMethod, F, TCompileMax
from .States import TimeMixState
# RWKV5 attention

wkv5_cuda = None
class RWKV_TimeMix(JITModClass):
    global wkv5_cuda

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att):
        global wkv5_cuda
        super().__init__()
        
        self.dim_att = dim_att
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.layer_id = layer_id

        self.n_head = n_head
        self.head_size = head_size

        # Optimized chunk length is fixed for now
        self.chunk_len = 512
        # assert ctx_len % self.chunk_len == 0
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
        self.gate = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)
        self.ln_x = nn.GroupNorm(n_head, dim_att)

    # this is based on jit_func(self,x)
        if wkv5_cuda is None:
            from torch.utils.cpp_extension import load
            loc = __file__[:__file__.rindex('/')] + '/'
            # wkv5_cuda = load(name="wkv5", sources=[loc+"cuda/wkv.cpp", loc + f"cuda/wkv.cu"],
            #                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={head_size}"])
                
        class WKV_5(torch.autograd.Function):
        
            def forward(ctx, B, T, C, H, r, k, v, w, u):
                with torch.no_grad():
                    assert r.dtype == torch.bfloat16
                    assert k.dtype == torch.bfloat16
                    assert v.dtype == torch.bfloat16
                    assert w.dtype == torch.bfloat16
                    assert u.dtype == torch.bfloat16
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
                    eew = (torch.exp(ew)).contiguous()
                    ctx.save_for_backward(r, k, v, eew, ew, u)
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
                    return y


            def backward(ctx, gy):
                with torch.no_grad():
                    assert gy.dtype == torch.bfloat16
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, eew, ew, u = ctx.saved_tensors
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                    wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
                    gw = torch.sum(gw, 0).view(H, C//H)
                    gu = torch.sum(gu, 0).view(H, C//H)
                    return (None, None, None, None, gr, gk, gv, gw, gu)

        self.WKV_5 = WKV_5
        class RWKV_5(torch.autograd.Function):
                   
            def forward(ctx, B, T, C, H, state, r, k, v, w, u):
                with torch.no_grad():
                    assert head_size == C // H
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert state.dtype == torch.float32
                    assert w.dtype == torch.float32
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()                            
                    assert u.is_contiguous()                            
                    assert state.is_contiguous()

                y = torch.empty((B, T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)
                if r.dtype == torch.bfloat16:
                    wkv5_cuda.forward_bf16(B, T, C, H, state, r, k, v, w, u, y)
                elif r.dtype == torch.float16:
                    wkv5_cuda.forward_fp16(B, T, C, H, state, r, k, v, w, u, y)
                elif r.dtype == torch.float32:
                    wkv5_cuda.forward_fp32(B, T, C, H, state, r, k, v, w, u, y)
                return y, state
                    
        self.RWKV_5 = RWKV_5


    def jit_func(self, x, state):
        B, T, C = x.size()

        xx = torch.cat((state.unsqueeze(1).clone(),x[:,:-1]),1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g, x[:,-1].clone()

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x / 8).view(B, T, C)
        x = self.output(x * g)
        return x
    
    # @torch.compile
    def cumsumwdecay(self,x,a,d,dim=1):
        x = torch.cat((a.unsqueeze(dim),x),dim)
        out = torch.empty_like(x)
        last = torch.zeros_like(x[:,0])
        for i in torch.arange(x.shape[dim]):
            last = x[:,i] + last*d
            out[:,i] = last

        return x
    
    def torchwise(self, B, T, C, H, s, r, k, v, w, u):
        out = torch.empty((B, T, H, C//H), dtype=r.dtype, device=r.device)
        at = k@v
        
        ss = self.cumsumwdecay(at,s,w)

        att = at*u + ss[:,:-1]

        out = r.float() @ att
            
        out = out.reshape(B, T, C)  
        return out, ss[:,-1]

    def forward(self, x, last_state:TimeMixState):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, xx = self.jit_func(x, last_state.shift_state)

        # if self.training:
        #     state = last_state.wkv_state
            
        #     x = self.WKV_5.apply(B, T, C, H, r, k, v, self.time_decay, self.time_faaaa)
        #     at = k.reshape(B,T,H,-1,1)@v.reshape(B,T,H,1,-1)
        #     p0 = torch.arange(T-1, -1, -1, device=r.device, dtype=torch.float32)
        #     p1 = self.time_decay.float().exp().neg().exp().reshape(1,1, H, C//H, 1).repeat(1, T, 1, 1, 1).pow(p0.reshape(1, T, 1, 1, 1))    
        #     # state at final step
        #     wkvstate = (at*p1).sum(1) + state*(self.time_decay.time_decay.float().exp().neg().exp().reshape(1,H, C//H,-1).pow(T))
            
        # else:
        state = last_state.wkv_state.to(x.device, torch.float32)
        x, wkvstate = self.torchwise(B, T, C, H, state, r.view(B, T, H, 1, C//H), k.view(B, T, H, C//H, 1), v.view(B, T, H, 1, C//H), self.time_decay.double().exp().neg().exp().reshape(1,self.n_head,-1,1).to(x.dtype), self.time_faaaa.reshape(1,1,self.n_head, -1, 1).to(x.dtype))

        x = x.reshape(B, T, C)
        out = self.jit_func_2(x.to(g.dtype), g)
        return out, TimeMixState(xx, wkvstate)
