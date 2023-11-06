# Dependencies
from .CoreDependencies import *
from .OptimizedOps import modified_lerp

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