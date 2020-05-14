import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch

def imshow(inp, title=None):
    """Imshow for Tensor."""
    D = torch.from_numpy(inp)

    # normalize to show
    mins, _ = D.min(dim=0)
    maxs, _ = D.max(dim=0)
    D = (D-mins) / (maxs - mins)
    D = torch.transpose(D, 0, 1)
    D = D.reshape(-1, 1, 28, 28).repeat(1, 3, 1, 1)
    out = make_grid(D)

    inp = out.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    plt.title('Dictionary of K-SVD (28*28, 64)')
    plt.show()  # pause a bit so that plots are updated



