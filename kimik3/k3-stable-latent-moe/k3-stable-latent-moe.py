import torch

def stable_latent_moe(
    tokens,
    latent_down_projection,
    latent_up_projection,
    router_projection,
    current_bias,
    routed_gate_weights,
    routed_up_weights,
    routed_down_weights,
    shared_gate_weights,
    shared_up_weights,
    shared_down_weights,
    selected_count,
    eps=1e-6,
    gate_cap=4.0,
    up_cap=25.0
):
    """
    Returns:
        final_output
        selected_indices
        mixture_weights
        latent_routed_aggregate
    """

    # tokens: (T, D)
    # T = number of token rows
    # D = model width
    #
    # routed expert weights:
    # (E, ..., ...)
    # E = number of routed experts
    #
    # latent width = L
    T,D = tokens.shape

    # TODO 1: Compute RAW router scores from the original full-width tokens.
    #
    # tokens:            (T,D)
    # router_projection: (E,D)
    #
    # raw_scores = sigmoid(W_r x)
    # raw_scores: (T,E)
    raw_scores = tokens @ router_projection.T
    raw_scores = torch.sigmoid(raw_scores)


    # TODO 2: Add current expert bias ONLY for deciding which experts win.
    #
    # raw_scores:   (T,E)
    # current_bias: (E,) -> broadcast across T
    #
    # biased_scores: (T,E)
    biased_scores = raw_scores + current_bias

    # TODO 3: Pick top-k routed experts independently for every token.
    #
    # selected_indices: (T,k)
    selected_values, selected_indices = torch.topk(
        biased_scores,
        k=selected_count,
        dim=1
    )

    # TODO 4: Gather the RAW scores of those selected experts.
    #
    # Important:
    # selection uses biased_scores,
    # mixture weights use raw_scores.
    #
    # selected_raw_scores: (T,k)
    selected_raw_scores = torch.gather(
        raw_scores,
        dim=-1,
        index=selected_indices
    )

    # TODO 5: Normalize selected raw scores per token.
    #
    # each row should sum to 1
    #
    # mixture_weights: (T,k)
    mixture_weights = (
        selected_raw_scores
        / selected_raw_scores.sum(dim=-1, keepdim=True)
    )                                                     # (m, k)


    # TODO 6: Down-project every token into latent width.
    #
    # tokens: (T,D)
    # latent_down_projection: (L,D)
    #
    # z: (T,L)
    z = tokens @ latent_down_projection.T 

    # TODO 7: Run the TWO shared experts directly on full-width tokens.
    #
    # Each shared expert is a SiTU-GLU FFN:
    #
    # gate = W_gate x
    # up   = W_up x
    # activated = bounded_SiTU_gate(gate) * bounded_up(up)
    # output = activated @ W_down^T
    #
    # shared_1: (T,D)
    # shared_2: (T,D)
    g = tokens @ shared_gate_weights[0].T
    u = tokens @ shared_up_weights[0].T
    
    h = (
        gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g)
    ) * (
        up_cap * torch.tanh(u / up_cap)
    )
    
    shared_1 = h @ shared_down_weights[0].T

    g = tokens @ shared_gate_weights[1].T
    u = tokens @ shared_up_weights[1].T
    
    h = (
        gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g)
    ) * (
        up_cap * torch.tanh(u / up_cap)
    )
    shared_2 = h @ shared_down_weights[1].T
    


    # TODO 8: Run selected routed experts in latent width.
    routed_outputs = []
    
    for t in range(T):
        token_outputs = []
    
        for j in range(selected_count):
            expert_id = selected_indices[t, j]
    
            x = z[t]   # (L,)
    
            g = x @ routed_gate_weights[expert_id].T
            u = x @ routed_up_weights[expert_id].T
    
            h = (
                gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g)
            ) * (
                up_cap * torch.tanh(u / up_cap)
            )
    
            expert_output = h @ routed_down_weights[expert_id].T   # (L,)
    
            token_outputs.append(expert_output)
    
        token_outputs = torch.stack(token_outputs, dim=0)   # (k,L)
        routed_outputs.append(token_outputs)
    
    routed_outputs = torch.stack(routed_outputs, dim=0)     # (T,k,L)
        


    # TODO 9: Weight selected routed expert outputs using mixture_weights.
    # mixture_weights: (T,k) -> (T,k,1)
    # routed_outputs:  (T,k,L)
    # weighted sum over k -> (T,L)
    u = (
        mixture_weights.unsqueeze(-1) * routed_outputs
    ).sum(dim=1)  # (T,L)
    
    
    # TODO 10: RMS-normalize the routed latent aggregate over L.
    # u:   (T,L)
    # rms: (T,1)
    rms = torch.sqrt(
        torch.mean(u ** 2, dim=-1, keepdim=True) + eps
    )  # (T,1)
    
    norm_u = u / rms  # (T,L)
    
    
    # TODO 11: Up-project routed latent result back to model width.
    # (T,L) @ (L,D) -> (T,D)
    routed_full = norm_u @ latent_up_projection.T  # (T,D)
    
    
    # TODO 12: Merge the two shared full-width experts
    # with the routed contribution.
    final_output = shared_1 + shared_2 + routed_full  # (T,D)
    
    
    # TODO 13: Return required outputs.
    return final_output, selected_indices, mixture_weights, u