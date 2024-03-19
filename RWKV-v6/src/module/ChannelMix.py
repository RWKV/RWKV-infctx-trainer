# Dependencies
from .CoreDependencies import *

class RWKV_ChannelMix6_0(JITModClass):    
    def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
        super().__init__()

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

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
    def forward(self, x : torch.Tensor, last_state: torch.Tensor):
        xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]),
                          dim=1)
        dxx = xx - x
        xk = x + dxx * self.time_maa_k
        xr = x + dxx * self.time_maa_r
        kv = self.value( torch.relu( self.key(xk) ) ** 2 )
        return torch.sigmoid(self.receptance(xr)) * kv, x[:, -1]
