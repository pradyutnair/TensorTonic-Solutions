import torch

def quantile_balancing(router_scores, current_bias, selected_count):
    """
    Returns: selected experts, mixture weights, loads, and the next centered bias.
    """

    # router_scores: (m, n)
    # m = tokens, n = experts
    m, n = router_scores.shape

    # Bias affects WHICH experts get routed to.
    biased_scores = router_scores + current_bias          # (m, n)

    # Pick top-k experts for every token.
    selected_values, selected_indices = torch.topk(
        biased_scores,
        k=selected_count,
        dim=-1
    )                                                     # both (m, k)

    # Mixture weights use RAW router scores, not biased scores.
    selected_raw_scores = torch.gather(
        router_scores,
        dim=-1,
        index=selected_indices
    )                                                     # (m, k)

    # Normalize selected expert scores separately for each token.
    mixture_weights = (
        selected_raw_scores
        / selected_raw_scores.sum(dim=-1, keepdim=True)
    )                                                     # (m, k)

    # Count how many tokens were assigned to each expert.
    loads = torch.bincount(
        selected_indices.reshape(-1),
        minlength=n
    )                                                     # (n,)

    # For each token, the (k+1)-th best biased expert is the cutoff.
    top_k_plus_one = torch.topk(
        biased_scores,
        k=selected_count + 1,
        dim=-1
    ).values                                              # (m, k+1)

    alpha = top_k_plus_one[:, -1]                         # (m,)

    # Desired assignments per expert.
    q = (m * selected_count) // n

    # How far each RAW expert score is above/below that token's cutoff.
    margins = router_scores - alpha.unsqueeze(-1)         # (m, n)

    # For every expert, look across all m tokens and take
    # its (q+1)-th largest margin.
    top_margins = torch.topk(
        margins,
        k=q + 1,
        dim=0                 # dim 0 = look DOWN tokens for each expert
    ).values                                              # (q+1, n)

    quantile_margin = top_margins[-1]                     # (n,)

    # Negating the margin gives the bias needed to move that expert's
    # routing boundary toward the desired load.
    next_bias_raw = -quantile_margin                      # (n,)

    # Bias is only meaningful relatively, so force mean bias to zero.
    next_bias = next_bias_raw - next_bias_raw.mean()      # (n,)

    return selected_indices, mixture_weights, loads, next_bias