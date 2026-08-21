import math
import torch

def gated_mla(hidden_states, query_projection, latent_down_projection, key_up_projection, value_up_projection, output_gate_projection, output_projection, num_heads, causal=True):
    """
    Returns: gated attention outputs and the latent key-value cache.
    """
    X = hidden_states
    B, S, D = X.shape
    H = num_heads
    Dh = D // H
    Q = X @ query_projection.T # (B,S,D)
    C = X @ latent_down_projection.T # (B,S,L)
    K, V = C @ key_up_projection.T, C @ value_up_projection.T # (B,S,D)
    Q = Q.reshape(B,S,H,Dh)
    K = K.reshape(B,S,H,Dh)
    V = V.reshape(B,S,H,Dh)

    # Put heads before sequence so the last two dims are
    # the actual matrices involved in attention.
    Q = Q.transpose(1, 2)              # (B,H,S,Dh)
    K = K.transpose(1, 2)              # (B,H,S,Dh)
    V = V.transpose(1, 2)              # (B,H,S,Dh)

    # For each head H, (S,Dh) @ (Dh,S) -> (S,S)
    scores = Q @ K.transpose(-1,-2)
    scores = scores / math.sqrt(Dh)
    if causal:
        # True above diagonal = future tokens that must be blocked
        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=scores.device),
            diagonal=1
        )  # (S, S)

        scores = scores.masked_fill(mask, float("-inf"))

    # Scores: B,H,S,S

    # Attention output (head context)
    A = torch.softmax(scores, dim=-1)@V # (B,H,S,S) @ (B,H,S,Dh) = B,H,S,Dh

    # Concatenate head contexts (attn scores) B,H,S,Dh into B,S,D
    A = A.transpose(1,2) # B,S,H,Dh
    O = A.reshape(B,S,D)

    Y = (torch.sigmoid(X@output_gate_projection.T) * O) @ output_projection.T
    return Y, C
    