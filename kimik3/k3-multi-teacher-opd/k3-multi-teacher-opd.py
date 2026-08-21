import torch

def multi_teacher_opd_reward(
    student_logits,
    teacher_logits,
    domain_indices,
    effort_indices,
    sampled_tokens,
    clip_threshold
):
    """
    Returns:
        clipped_reward:          (B, S)
        teacher_token_log_probs: (B, S)
    """

    # student_logits:  (B, S, V)
    # teacher_logits:  (3, 3, B, S, V)
    B, S, V = student_logits.shape

    # Each batch example may use a different teacher.
    # For batch b, select teacher[domain[b], effort[b], b].
    # B separate (S,V) selections -> (B,S,V).
    batch_indices = torch.arange(B, device=teacher_logits.device)

    selected_teacher_logits = teacher_logits[
        domain_indices,
        effort_indices,
        batch_indices
    ]  # (B, S, V)

    # Convert logits into log-probabilities over the vocabulary.
    # Each token position gets its own distribution over V.
    student_log_probs = torch.log_softmax(
        student_logits, dim=-1
    )  # (B, S, V)

    teacher_log_probs = torch.log_softmax(
        selected_teacher_logits, dim=-1
    )  # (B, S, V)

    # We don't care about all V token probabilities.
    # At each (batch, sequence) position, pick the probability
    # of the token that was actually sampled.
    student_token_log_probs = torch.gather(
        student_log_probs,
        dim=-1,
        index=sampled_tokens.unsqueeze(-1)
    ).squeeze(-1)  # (B, S)

    teacher_token_log_probs = torch.gather(
        teacher_log_probs,
        dim=-1,
        index=sampled_tokens.unsqueeze(-1)
    ).squeeze(-1)  # (B, S)

    # Positive reward = teacher likes sampled token more than student.
    # Negative reward = student likes it more than teacher.
    raw_reward = (
        teacher_token_log_probs
        - student_token_log_probs
    )  # (B, S)

    # Reward should act as a fixed score during training,
    # not as another differentiable path into either policy.
    detached_reward = raw_reward.detach()  # (B, S)

    # Prevent unusually large teacher/student gaps from dominating.
    clipped_reward = torch.clamp(
        detached_reward,
        -clip_threshold,
        clip_threshold
    )  # (B, S)

    return clipped_reward, teacher_token_log_probs