import torch

def per_head_muon(
    parameter,
    gradient,
    previous_momentum,
    num_heads,
    momentum_coefficient,
    learning_rate
):
    """
    Returns: updated parameter, momentum, and per-head orthogonalized update.
    """

    # parameter / gradient / previous_momentum: (R, C)
    # R = output rows
    # C = input columns
    # Rows are split evenly across attention heads.

    R, C = parameter.shape

    # TODO 1: Update momentum.
    #
    # M_t = mu * M_{t-1} + G_t
    #
    # momentum: (R,C)
    momentum = momentum_coefficient * previous_momentum + gradient


    # TODO 2: Work out how many ROWS belong to each attention head.
    #
    # Example:
    # R = 12, H = 3
    # -> each head owns 4 rows
    rows_per_head = R // num_heads


    # TODO 3: Create somewhere to collect each head's
    # independently orthogonalized update.
    head_updates = []


    # TODO 4: Loop over heads.
    for h in range(num_heads):

        # Find this head's row range.
        #
        # head 0 -> rows [0 : rows_per_head]
        # head 1 -> next rows
        # etc.
        start = h * rows_per_head
        end = (h + 1) * rows_per_head

        # Slice ONLY this head's momentum rows.
        #
        # momentum:      (R,C)
        # head_momentum: (rows_per_head,C)
        head_momentum = momentum[start:end]


        # TODO 5: Compact SVD of this head block.
        #
        # head_momentum = U @ diag(S) @ Vh
        #
        # U:  (rows_per_head, r)
        # Vh: (r, C)
        #
        # r = min(rows_per_head, C)
        U, S, Vh = torch.linalg.svd(
            head_momentum,
            full_matrices=False
        )


        # TODO 6: Compute this head's polar factor.
        #
        # O_h = U @ Vh
        #
        # shape should return to:
        # (rows_per_head,C)
        head_update = U @ Vh


        head_updates.append(head_update)


    # TODO 7: Put all head updates back in their ORIGINAL row order.
    #
    # H blocks of (rows_per_head,C)
    # -> (R,C)
    orthogonal_update = torch.cat(
        head_updates,
        dim=0
    )


    # TODO 8: Apply optimizer step.
    #
    # theta_{t+1} = theta_t - eta * O
    #
    # updated_parameter: (R,C)
    updated_parameter = parameter - learning_rate*orthogonal_update


    return updated_parameter, momentum, orthogonal_update