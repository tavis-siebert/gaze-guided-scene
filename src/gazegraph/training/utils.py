import torch


def get_optimizer(optimizer_name, model, lr):
    if optimizer_name.lower() == "adam":
        # TODO add support for further parameters via kwargs such as beta, decay, etc
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif optimizer_name.lower() == "sgd":
        # TODO add support for further parameters via kwargs such as momentum, decay, etc
        optimizer = torch.optim.SGD(model.parameters(), lr)
    else:
        raise ValueError(f"{optimizer_name} is not a valid optimizer")
    return optimizer
