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

from .ksvd import ksvd
from .sparseRep import random_sensor, sense, reconstruct


def generate_MNIST():
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

    data_train = data_train[:2000]
    data_valid = data_valid[:500]

    num_train = data_train.shape[0]
    num_test = data_valid.shape[0]

    print("[INFO] => Train data {}".format(num_train))
    print("[INFO] => Test  data {}".format(num_test))

    return [data_train, label_train], [data_valid, label_valid]


if __name__ == "__main__":
    # Generate train data: N samples of white noise signals
    [data_train, label_train], [data_test, label_test] = generate_MNIST()

    N = data_train.shape[0]
    signal_length = data_train.shape[1]
    sense_size = 25

    # K-SVD algorithm to learn sparse representation of data
    D_size = 64
    max_sparsity = int(0.25*D_size)
    max_iter = 1000
    D, X, e = ksvd(data_train, D_size, max_sparsity, maxiter=max_iter, debug=True)

    np.save('./D', D)
    np.save('./X', X)
    np.save('./e', e)

    # compressive sensing
    sensor = random_sensor((signal_length, sense_size))
    representation = sense(data_test, sensor)

    # sparse reconstruction
    reconstruction = reconstruct(representation, sensor, D, max_sparsity)

    print (D.shape, sensor.shape)

    plt.plot(data_test[0, :], '--', label="true signal")
    plt.plot(reconstruction, '.-', label="reconstruction")
    plt.title("True vs. reconstructed signal")
    plt.legend()
    plt.show()