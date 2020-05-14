import numpy as np
import torch
from compressed_sensing_MNIST import reconstruction_MNIST
from sparse_coding_MNIST import generate_MNIST
from ksvd.ksvd import ksvd
from sparse_coding_1d_sample import run_1D_sparse_coding

import matplotlib.pyplot as plt


def sparse_coding_demo():
    print('[INFO] ========== Sparse Coding Demo ========== ')

    run_1D_sparse_coding(m=64, n=256, s=0.05, n_sample=1000)


def dictionary_learning_demo():
    """
    Decomposing data Y into D and X via K-SVD.
    Formulation:
                    Y = D * X
        Y: target image
        D: dictionary
        X: coefficients (need to be sparse)

    :return:
        D: dictionary
        X: coefficients
        e: error
    """

    print('[INFO] ========== Dictionary Learning Demo ========== ')

    # Generate train data: N samples of white noise signals
    [data_train, label_train], [data_test, label_test] = generate_MNIST(2000, 500)

    N = data_train.shape[0]

    # K-SVD algorithm to learn sparse representation of data
    D_size = 64
    max_sparsity = 0.25
    max_iter = 1000
    D, X, e = ksvd(data_train.T, D_size, max_sparsity, maxiter=max_iter, debug=True)

    np.save('./ksvd/D', D)
    np.save('./ksvd/X', X)
    np.save('./ksvd/e', e)


def reconstruction_demo():
    """
    Reconstruct images.
    Formulation:
                    Y = S * D * Coef
        Y: is measurement. (given)
        S: is sensing matrix. (given)
        D: is dictionary. (given)
        Coef: is coefficients. (unknown, need to be sparse)

    Goal:
        Reconstruct image X from Y, S, D by computing Coef (X = D * Coef.)

    :return:
        Reconstructed image.
    """

    print('[INFO] ========== CS Reconstruction Demo ========== ')

    # settings
    [data_train, label_train], [data_test, label_test] = generate_MNIST(2000, 5)
    D = np.load('./ksvd/D.npy')  # dictionary
    s = 0.25  # sparse level
    sparsity = s * D.shape[1]
    sense_size = 500  # measurements number

    # reconstruction
    img, img_recon = reconstruction_MNIST(data_test, D, sparsity, sense_size)

    for idx in range(img.shape[0]):
        plt.figure()
        plt.subplot(211)
        plt.imshow(img[idx], 'gray')
        plt.subplot(212)
        plt.imshow(img_recon[idx], 'gray')
        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    sparse_coding_demo()
    # dictionary_learning_demo()
    # reconstruction_demo()