from torch.optim import SGD, Adam, AdamW

def get_optimizer(optimizer_name, model, lr, **kwargs):
    """
    General method for abstracting optimizers
    NOTE: please refer to torch.optim documentation to 
          define additional parameters properly (e.g. beta, weight_decay)
    """
    if optimizer_name.lower() == "adam":
        optimizer = Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"{optimizer_name} is not a valid optimizer")
    return optimizer
