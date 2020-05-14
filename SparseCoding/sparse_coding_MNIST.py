import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import dft

from ksvd.ksvd import ksvd
from ksvd.sparseRep import random_sensor, sense, reconstruct


def generate_MNIST(train_num=60000, valid_num=10000):
    torch.manual_seed(1)  # reproducible

    DOWNLOAD_MNIST = False

    if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        DOWNLOAD_MNIST = True

    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST,
    )

    test_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST,
    )

    # convert list data to numpy array

    data_train = np.asarray(train_data.data).reshape(60000, -1)
    data_valid = np.asarray(test_data.data).reshape(10000, -1)

    label_train = np.asarray(train_data.targets).reshape(60000, -1)
    label_valid = np.asarray(test_data.targets).reshape(10000, -1)

    data_train = data_train[:train_num]
    data_valid = data_valid[:valid_num]

    label_train = label_train[:train_num]
    label_valid = label_valid[:valid_num]

    print("[INFO] Train samples {}, Valid samples {}".format(train_num, valid_num))

    return [data_train, label_train], [data_valid, label_valid]

