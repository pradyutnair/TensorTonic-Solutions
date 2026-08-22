import torch

def kda_context_parallel(
    transition_matrices,
    local_contributions,
    initial_state
):
    """
    Returns: incoming states, outgoing states, and final global state.
    """

    # transition_matrices: (N, W, W)
    # local_contributions: (N, W, V)
    # initial_state:       (W, V)
    #
    # N = number of sequence segments
    # W = state width
    # V = value width

    N = transition_matrices.shape[0]

    # TODO 1: Start from the global initial state.
    #
    # current_state is the state entering the next segment.
    # current_state: (W,V)
    current_state = initial_state.clone()

    # TODO 2: Store each segment's incoming and outgoing state.
    incoming_states = []
    outgoing_states = []

    # TODO 3: Process segments in SEQUENCE ORDER.
    for i in range(N):

        # State BEFORE segment i runs.
        # incoming: (W,V)
        incoming = current_state 
        incoming_states.append(incoming)

        # Segment i's affine-map pieces.
        #
        # M_i: (W,W)
        # U_i: (W,V)
        M_i = transition_matrices[i]
        U_i = local_contributions[i]

        # TODO 4: Apply this segment:
        #
        # outgoing = M_i @ incoming + U_i
        #
        # (W,W) @ (W,V) -> (W,V)
        outgoing = M_i @ incoming + U_i

        # TODO 5: Save incoming/outgoing states.
        outgoing_states.append(outgoing)

        # TODO 6: The next segment receives this outgoing state.
        current_state = outgoing

        

    # TODO 7:
    # N tensors of (W,V) -> (N,W,V), use stack for new dim
    incoming_states = torch.stack(
        incoming_states,
        dim=0
    )
    outgoing_states = torch.stack(
        outgoing_states,
        dim=0
    )

    # After the last segment, current_state is the final global state.
    final_state = current_state  # (W,V)

    return incoming_states, outgoing_states, final_state