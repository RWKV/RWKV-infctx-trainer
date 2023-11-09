# Dependencies
from .CoreDependencies import *
from .OptimizedOps import modified_lerp

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
    @TCompileMax
    @JITModMethod
    def forward(self, x, last_state: tuple[torch.Tensor,torch.Tensor]):
        # Get the x sizing, and the chunking length
        B, TT, C = x.size()
        chunk_len = 32

        # Logits to return
        x_logits = torch.zeros(B, TT, C, device=x.device, dtype=x.dtype)

        # Process in chunks
        for i in range(0, TT, chunk_len):
            # Get the chunk
            chunk = x[:, i:i+chunk_len]

            # Process the chunk
            chunk_logits, last_state = self._forward_noBatching(chunk, last_state)

            # Store the chunk logits
            x_logits[:, i:i+chunk_len] = chunk_logits
        
        # Return the logits and the state
        return (x_logits, last_state) 

    def _forward_noBatching(self, x, last_state: tuple[torch.Tensor,torch.Tensor]):
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