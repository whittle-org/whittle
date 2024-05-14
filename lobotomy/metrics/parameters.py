def compute_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
