import torch

def _read_depth_sources(sources, pseudo_query, eps):
    normalized = sources / torch.sqrt(
        sources.square().mean(dim=-1, keepdim=True) + eps
    )
    logits = (normalized * pseudo_query).sum(dim=-1)
    weights = torch.softmax(logits, dim=0)
    retrieved = (weights.unsqueeze(-1) * sources).sum(dim=0)
    return retrieved, weights


def block_attention_residual(
    embedding,
    previous_outputs,
    pseudo_query,
    block_size,
    eps=1e-6
):
    """
    Returns: retrieved values, depth weights, and block-level sources.
    """

    # embedding:        (B,S,D)
    # previous_outputs: (P,B,S,D)

    P = previous_outputs.shape[0]

    # Keep each depth source as one (B,S,D) tensor.
    # Source 0 is always the embedding.
    sources = [embedding]

    # Number of full groups of block_size previous layers.
    num_complete_blocks = P // block_size

    for j in range(num_complete_blocks):
        start = j * block_size
        end = start + block_size

        # (block_size,B,S,D) -> sum layers -> (B,S,D)
        block = previous_outputs[start:end].sum(dim=0)

        # This whole block becomes ONE depth source.
        sources.append(block)

    # Number of layers left after all complete blocks.
    remainder = P % block_size

    if remainder > 0:
        # Only the ungrouped trailing layers.
        # (remainder,B,S,D) -> (B,S,D)
        partial_block = previous_outputs[-remainder:].sum(dim=0)

        # Partial block also becomes ONE depth source.
        sources.append(partial_block)

    # List of (B,S,D) tensors
    # -> (num_sources,B,S,D)
    block_sources = torch.stack(sources, dim=0)

    # Run the same depth-attention read as Full AttnRes.
    # retrieved: (B,S,D)
    # weights:   (num_sources,B,S)
    retrieved, weights = _read_depth_sources(
        block_sources,
        pseudo_query,
        eps
    )

    return retrieved, weights, block_sources