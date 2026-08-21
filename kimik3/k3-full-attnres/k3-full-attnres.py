import torch

def full_attention_residual(embedding, previous_outputs, pseudo_query, eps=1e-6):
    """
    Returns:
        retrieved: (B, S, D)
        weights:   (L, B, S)
    """

    # embedding:        (B, S, D)
    # previous_outputs: (P, B, S, D)
    # pseudo_query:     (D,)
    #
    # P = number of previous layer outputs
    # L = total number of depth sources = P + embedding

    B, S, D = embedding.shape

    # Add embedding as depth source 0.
    # (B,S,D) -> (1,B,S,D)
    embedding_source = embedding.unsqueeze(0)     # (1, B, S, D)

    # Stack embedding + previous layer outputs along depth.
    # (1,B,S,D) + (P,B,S,D) -> (L,B,S,D)
    sources = torch.cat(
        [embedding_source, previous_outputs],
        dim=0
    )                                             # (L, B, S, D)

    # RMS-normalize each source across feature dimension D.
    # mean over D -> (L,B,S,1)
    rms = torch.sqrt(
        torch.mean(sources ** 2, dim=-1, keepdim=True) + eps
    )                                             # (L, B, S, 1)

    normalized_keys = sources / rms               # (L, B, S, D)

    # Score each depth source with pseudo-query.
    #
    # (L,B,S,D) @ (D,) -> (L,B,S)
    scores = normalized_keys @ pseudo_query       # (L, B, S)

    # Normalize across DEPTH L.
    # dim=0 because depth is the first dimension.
    weights = torch.softmax(scores, dim=0)        # (L, B, S)

    # Weight the ORIGINAL, unnormalized source representations.
    #
    # weights.unsqueeze(-1): (L,B,S,1)
    # sources:               (L,B,S,D)
    # product:               (L,B,S,D)
    weighted_sources = (
        weights.unsqueeze(-1) * sources
    )                                             # (L, B, S, D)

    # Sum over all depth sources.
    # (L,B,S,D) -> (B,S,D)
    retrieved = weighted_sources.sum(dim=0)       # (B, S, D)

    return retrieved, weights