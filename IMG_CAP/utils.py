import torch


def save_checkpoint(state, filename="IMG_CAP.pth.tar"):
    print("__Saving checkpoint__")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("__Loading checkpoint__")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])