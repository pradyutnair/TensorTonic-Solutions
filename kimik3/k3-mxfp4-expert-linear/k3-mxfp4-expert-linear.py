import torch

_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5,
    2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5,
    -2.0, -3.0, -4.0, -6.0
]


def mxfp4_expert_linear(
    latent_tokens,
    packed_weights,
    scale_bytes,
    selected_experts,
    mixture_weights,
    shared_output
):
    # latent_tokens:    (T,I)
    # packed_weights:   (E,O,G,16)
    # scale_bytes:      (E,O,G)
    # selected_experts: (T,K)
    # mixture_weights:  (T,K)
    # shared_output:    (T,O)
    #
    # I = G * 32

    T, I = latent_tokens.shape
    E, O, G, _ = packed_weights.shape
    K = selected_experts.shape[1]

    # Maps compressed codes 0..15 -> actual FP4 numbers.
    lookup = torch.tensor(
        _E2M1_VALUES,
        dtype=latent_tokens.dtype,
        device=latent_tokens.device
    )  # (16,)

    # Shared experts have already been computed.
    # Start with them and add routed expert contributions.
    output = shared_output.clone()  # (T,O)

    for t in range(T):

        x = latent_tokens[t]  # (I,)

        for j in range(K):

            # --------------------------------------------------
            # 1. Which expert did this token select?
            # --------------------------------------------------

            expert_id = selected_experts[t, j]
            mixture_weight = mixture_weights[t, j]

            # Only decode THIS selected expert.
            expert_packed = packed_weights[expert_id]  # (O,G,16)
            expert_scales = scale_bytes[expert_id]     # (O,G)


            # --------------------------------------------------
            # 2. Each byte stores TWO 4-bit codes.
            #
            # Example byte:
            #
            # 0011 0010
            # ---- ----
            #  3    2
            #
            # low code  = 2
            # high code = 3
            # --------------------------------------------------

            low_codes = expert_packed & 0x0F   # (O,G,16)
            high_codes = expert_packed >> 4    # (O,G,16)


            # --------------------------------------------------
            # 3. Turn codes into actual numbers.
            #
            # code 2 -> 1.0
            # code 3 -> 1.5
            # --------------------------------------------------

            low_values = lookup[low_codes.long()]    # (O,G,16)
            high_values = lookup[high_codes.long()]  # (O,G,16)


            # --------------------------------------------------
            # 4. Restore original order.
            #
            # byte0 -> low0, high0
            # byte1 -> low1, high1
            # ...
            #
            # 16 bytes -> 32 weights
            # --------------------------------------------------

            decoded_groups = torch.stack(
                [low_values, high_values],
                dim=-1
            )  # (O,G,16,2)

            decoded_groups = decoded_groups.reshape(
                O, G, 32
            )  # (O,G,32)


            # --------------------------------------------------
            # 5. Each group of 32 weights has one scale.
            #
            # scale_byte = 127 -> 2^(127-127) = 1
            # scale_byte = 129 -> 2^(129-127) = 4
            # --------------------------------------------------

            scales = torch.pow(
                2.0,
                expert_scales.to(latent_tokens.dtype) - 127.0
            ).unsqueeze(-1)  # (O,G,1)

            scaled_groups = (
                decoded_groups * scales
            )  # (O,G,32)


            # --------------------------------------------------
            # 6. Join G groups back into the full expert matrix.
            #
            # (O,G,32) -> (O,G*32) = (O,I)
            # --------------------------------------------------

            expert_matrix = scaled_groups.reshape(
                O, I
            )  # (O,I)


            # --------------------------------------------------
            # 7. Actually run the expert.
            #
            # W: (O,I)
            # x: (I,)
            #
            # -> (O,)
            # --------------------------------------------------

            expert_output = expert_matrix @ x  # (O,)


            # --------------------------------------------------
            # 8. Add this expert according to router weight.
            # --------------------------------------------------

            output[t] = (
                output[t]
                + mixture_weight * expert_output
            )

    return output