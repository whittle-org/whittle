import torch


def compute_mag(model, mask):
    magnitude = 0
    for params in model.input_layer.parameters():
        magnitude += torch.sum(torch.abs(params))

    binary_mask = torch.tensor(mask, dtype=torch.bool)
    for params in model.input_layer.parameters():
        if len(params.shape) == 1:
            p = params[binary_mask]
        else:
            p = params[binary_mask, :]
        magnitude += torch.sum(torch.abs(p))
    for params in model.output_layer.parameters():
        magnitude += torch.sum(torch.abs(params))

    return float(magnitude)
