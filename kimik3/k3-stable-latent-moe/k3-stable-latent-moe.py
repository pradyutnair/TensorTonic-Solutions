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
    tokens: (T, D)

    Returns:
        final_output:     (T, D)
        selected_indices: (T, k)
        mixture_weights:  (T, k)
        routed_aggregate: (T, L)
    """

    T, D = tokens.shape


    # ============================================================
    # 1. ROUTING: decide which routed experts each token will use
    # ============================================================

    # Each token gets one raw score for every routed expert.
    # (T,D) @ (D,E) -> (T,E)
    raw_scores = torch.sigmoid(
        tokens @ router_projection.T
    )  # (T,E)

    # Bias affects WHICH experts are selected, not their mixture weights.
    biased_scores = raw_scores + current_bias  # (T,E)

    # Pick top-k experts independently for each token.
    _, selected_indices = torch.topk(
        biased_scores,
        k=selected_count,
        dim=-1
    )  # (T,k)

    # Get the RAW scores of only those selected experts.
    selected_raw_scores = torch.gather(
        raw_scores,
        dim=-1,
        index=selected_indices
    )  # (T,k)

    # Convert selected raw scores into mixture weights.
    # Each token's k weights sum to 1.
    mixture_weights = (
        selected_raw_scores
        / selected_raw_scores.sum(dim=-1, keepdim=True)
    )  # (T,k)


    # ============================================================
    # 2. SHARED PATH: both shared experts process EVERY token at D
    # ============================================================

    # ----- Shared expert 0 -----

    shared_gate_0 = tokens @ shared_gate_weights[0].T
    shared_up_0 = tokens @ shared_up_weights[0].T

    shared_hidden_0 = (
        gate_cap
        * torch.tanh(shared_gate_0 / gate_cap)
        * torch.sigmoid(shared_gate_0)
    ) * (
        up_cap
        * torch.tanh(shared_up_0 / up_cap)
    )

    shared_output_0 = (
        shared_hidden_0 @ shared_down_weights[0].T
    )  # (T,D)


    # ----- Shared expert 1 -----

    shared_gate_1 = tokens @ shared_gate_weights[1].T
    shared_up_1 = tokens @ shared_up_weights[1].T

    shared_hidden_1 = (
        gate_cap
        * torch.tanh(shared_gate_1 / gate_cap)
        * torch.sigmoid(shared_gate_1)
    ) * (
        up_cap
        * torch.tanh(shared_up_1 / up_cap)
    )

    shared_output_1 = (
        shared_hidden_1 @ shared_down_weights[1].T
    )  # (T,D)


    # ============================================================
    # 3. ROUTED PATH: compress tokens D -> L
    # ============================================================

    latent_tokens = (
        tokens @ latent_down_projection.T
    )  # (T,L)


    # ============================================================
    # 4. Run ONLY the selected routed experts on each latent token
    # ============================================================

    all_token_expert_outputs = []

    for t in range(T):

        # Outputs of this token's k selected experts.
        token_expert_outputs = []

        for j in range(selected_count):

            expert_id = selected_indices[t, j]

            # This expert receives the token in LATENT width.
            latent_token = latent_tokens[t]  # (L,)

            # Same SiTU-GLU FFN, but all computation happens in L.
            expert_gate = (
                latent_token @ routed_gate_weights[expert_id].T
            )

            expert_up = (
                latent_token @ routed_up_weights[expert_id].T
            )

            expert_hidden = (
                gate_cap
                * torch.tanh(expert_gate / gate_cap)
                * torch.sigmoid(expert_gate)
            ) * (
                up_cap
                * torch.tanh(expert_up / up_cap)
            )

            expert_output = (
                expert_hidden
                @ routed_down_weights[expert_id].T
            )  # (L,)

            token_expert_outputs.append(expert_output)

        # k expert outputs for this token:
        # k × (L,) -> (k,L)
        token_expert_outputs = torch.stack(
            token_expert_outputs,
            dim=0
        )  # (k,L)

        all_token_expert_outputs.append(token_expert_outputs)

    # T × (k,L) -> (T,k,L)
    routed_outputs = torch.stack(
        all_token_expert_outputs,
        dim=0
    )  # (T,k,L)


    # ============================================================
    # 5. Combine the k routed experts for each token
    # ============================================================

    # mixture_weights: (T,k)   -> (T,k,1)
    # routed_outputs:  (T,k,L)
    #
    # Multiply each expert output by its routing weight,
    # then sum across the k selected experts.
    routed_aggregate = (
        mixture_weights.unsqueeze(-1)
        * routed_outputs
    ).sum(dim=1)  # (T,L)


    # ============================================================
    # 6. Normalize routed result and bring L -> D
    # ============================================================

    rms = torch.sqrt(
        routed_aggregate.square().mean(
            dim=-1,
            keepdim=True
        ) + eps
    )  # (T,1)

    normalized_routed = routed_aggregate / rms  # (T,L)

    routed_full_width = (
        normalized_routed @ latent_up_projection.T
    )  # (T,D)


    # ============================================================
    # 7. Merge all three paths
    # ============================================================

    final_output = (
        shared_output_0
        + shared_output_1
        + routed_full_width
    )  # (T,D)

    return (
        final_output,
        selected_indices,
        mixture_weights,
        routed_aggregate
    )