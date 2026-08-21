import torch

def kda_recurrence(
    query,
    key,
    value,
    decay_logits,
    write_strength,
    output_gate_logits,
    output_projection,
    initial_state,
    g_min=-5.0,
    eps=1e-6
):
    """
    Returns: sequence outputs and the final recurrent state.
    """

    # query, key:          (B, S, H, Dk)
    # value, gate logits:  (B, S, H, Dv)
    # decay_logits:        (B, S, H, Dk)
    # initial_state:       (B, H, Dk, Dv)

    B, S, H, Dk = query.shape
    Dv = value.shape[-1]

    state = initial_state                  # (B, H, Dk, Dv)
    outputs = []

    for t in range(S):

        q_t = query[:, t]                  # (B, H, Dk)
        k_t = key[:, t]                    # (B, H, Dk)
        v_t = value[:, t]                  # (B, H, Dv)
        z_t = decay_logits[:, t]            # (B, H, Dk)
        beta_t = write_strength[:, t]       # (B,H) or (B,H,1)
        gate_t = output_gate_logits[:, t]   # (B, H, Dv)

        # 1. Retention
        alpha_t = torch.exp(
            g_min * torch.sigmoid(z_t)
        ).unsqueeze(-1)                    # (B, H, Dk, 1)

        # 2. Decay
        decayed_state = alpha_t * state    # (B, H, Dk, Dv)

        # 3. Find what is currently stored at key k_t
        memory_at_key = torch.einsum(
            "bhd,bhdv->bhv",
            k_t,
            decayed_state
        )                                  # (B, H, Dv)

        # k_t (k_t^T S)
        erase_term = (
            k_t.unsqueeze(-1)              # (B, H, Dk, 1)
            * memory_at_key.unsqueeze(-2)  # (B, H, 1, Dv)
        )                                  # (B, H, Dk, Dv)

        # beta -> (B, H, 1, 1)
        if beta_t.ndim == 2:
            beta_t = beta_t.unsqueeze(-1)  # (B, H, 1)

        beta_t = beta_t.unsqueeze(-1)       # (B, H, 1, 1)

        # 4. Erase
        erased_state = (
            decayed_state
            - beta_t * erase_term
        )                                  # (B, H, Dk, Dv)

        # 5. Write beta * k_t * v_t^T
        write_term = beta_t * (
            k_t.unsqueeze(-1)              # (B, H, Dk, 1)
            * v_t.unsqueeze(-2)            # (B, H, 1, Dv)
        )                                  # (B, H, Dk, Dv)

        state = erased_state + write_term  # (B, H, Dk, Dv)

        # 6. Read from UPDATED state: S_t^T q_t
        state_T = state.transpose(-1, -2)  # (B, H, Dv, Dk)
        q_col = q_t.unsqueeze(-1)          # (B, H, Dk, 1)

        read = (state_T @ q_col).squeeze(-1)
        # (B, H, Dv)

        # 7. RMS normalization over Dv
        rms = torch.sqrt(
            torch.mean(read ** 2, dim=-1, keepdim=True) + eps
        )                                  # (B, H, 1)

        normalized_read = read / rms       # (B, H, Dv)

        # 8. Output gate
        gate = torch.sigmoid(gate_t)       # (B, H, Dv)
        gated_read = normalized_read * gate
        # (B, H, Dv)

        # 9. Concatenate heads
        concatenated = gated_read.reshape(B, H * Dv)
        # (B, H * Dv)

        # 10. Output projection
        # projection is (model_width, H*Dv)
        token_output = concatenated @ output_projection.T
        # (B, model_width)

        # 11. Save token output
        outputs.append(token_output)

    # 12. Restore sequence dimension
    outputs = torch.stack(outputs, dim=1)
    # (B, S, model_width)

    # 13. Return outputs + final recurrent memory
    return outputs, state