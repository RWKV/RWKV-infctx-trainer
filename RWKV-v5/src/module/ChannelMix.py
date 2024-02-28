# Dependencies
from .CoreDependencies import *
from .OptimizedOps import modified_lerp

class RWKV_Expert(nn.Module):
    def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
        super().__init__()
        self.ffn_key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.ffn_receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.ffn_value = nn.Linear(dim_ffn, n_embd, bias=False)

    def forward(self, x):
        kv = self.ffn_value( torch.relu( self.ffn_key(x) ) ** 2 )
        # NOTE - we use the same x for receptance here, unlike in normal chanmix
        return torch.sigmoid(self.ffn_receptance(x)) * kv

class DDLorExp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.w1 = nn.Parameter(torch.empty(in_dim, hidden_dim).uniform_(-0.01, 0.01))
        self.w2 = nn.Parameter(torch.zeros(hidden_dim, out_dim))
        self.gmult = nn.Parameter(torch.ones(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        glora = (torch.tanh(x @ self.w1) @ self.w2).exp()
        gate = self.gmult.to(glora) * glora
        return gate


class RWKV_Expert6_0x(nn.Module):
    def __init__(self, layer_id, n_layer, D, F):
        super().__init__()

        self.w_hidden = nn.Linear(D, F, bias=False)
        self.w_out = nn.Linear(F, D, bias=False)
        GATE_EXP_DIM = max(64, D // 16)
        self.w_gate = DDLorExp(in_dim=D, hidden_dim=GATE_EXP_DIM, out_dim=D)

    def forward(self, x: torch.Tensor):
        hidden = self.w_hidden(x)
        hidden = torch.relu(hidden).pow(2)
        gate = self.w_gate(x)
        return gate * self.w_out(hidden)

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

    # forwarding channel mix given the model weights and the input tokens and states.
    #
    # Given:
    # - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
    # - Last shift states of the various batches [batch_size, state_size]
    #
    # Returns a pair 
    # - of output embedding of shape [batch_size, seq_len, embedding_size]
    # - and the last output state of shape [batch_size, state_size]
    @JITModMethod
    def forward(self, x : torch.Tensor, xx : torch.Tensor):
        # out_emb, out_state = channelMix_batchForward(
        #     self.time_mix_k,self.time_mix_r,
        #     self.key.bi
        #     self.key.weight, self.receptance.weight, self.value.weight,
        #     x, last_state.shift_state
        # )
        # return (out_emb, (out_state))
    
        #xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]),
        #                  dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        kv = self.value( torch.relu( self.key(xk) ) ** 2 )
        return torch.sigmoid(self.receptance(xr)) * kv

# Pure lambda implementation, of forwarding channel mix given the model weights
# and the input tokens and states.
#
# Returns a pair 
# - of output embedding of shape [batch_size, seq_len, embedding_size]
# - and the output state of shape [batch_size, seq_len, state_size]
#
# @ TCompileMax
# @ JITFunction
def channelMix_batchForward(
    # Various weights from the channel mix layer
    time_mix_k,
    time_mix_r,
    w_key: torch.nn.Module, 
    receptance: torch.nn.Module, 
    value: torch.nn.Module,
    # Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
    x_embedding,
    # Last shift states of the various batches [batch_size, state_size]
    last_shift_state
):
    # Compute accordingly the full state shift
    # [batch_size, seq_len, state_size]
    full_state_shift = torch.concat((last_shift_state.unsqueeze(1), x_embedding[:, :-1]), dim=1)

    # Computing the channel mix components
    xk = modified_lerp(x_embedding, time_mix_k, full_state_shift)
    xr = modified_lerp(x_embedding, time_mix_r, full_state_shift)
    k  = w_key(xk)
    kv = value( torch.relu(k) ** 2 )

    # Compute the output embeddings, and the last_shift_state
    return (torch.sigmoid(receptance(xr)) * kv, x_embedding[:, -1])