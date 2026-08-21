import torch

def situ_glu(input_tensor, gate_projection, up_projection, gate_cap=4.0, up_cap=25.0):
    """
    Returns: the bounded element-wise gated activation.
    """
    g = input_tensor @ gate_projection.T   # (..., F)
    u = input_tensor @ up_projection.T     # (..., F)
    sig_g = torch.sigmoid(g)
    situ_glu = (gate_cap * torch.tanh(g / gate_cap) * sig_g) * (up_cap * torch.tanh(u / up_cap))
    return situ_glu