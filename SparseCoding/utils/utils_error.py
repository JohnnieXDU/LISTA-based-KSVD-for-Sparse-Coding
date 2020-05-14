import numpy as np
import torch
from math import log10, sqrt


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def l2_error(y1, y2):
    """

    :param y1: size [sample len, sample num], np.array
    :param y2: size [sample len, sample num], np.array
    :return:
    """

    diff = y1 - y2
    error = np.sqrt(np.sum(np.power(diff, 2))) / y1.shape[1]
    return error


def l2_error_torch(y1, y2):
    """

    :param y1:  size [sample len, sample num], torch.tensor
    :param y2:  size [sample len, sample num], torch.tensor
    :return:
    """

    y1 = y1.cpu().detach().numpy()
    y2 = y2.cpu().detach().numpy()

    return l2_error(y1, y2)