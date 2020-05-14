import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import orth
import time

from models.ISTA_model import ista
from models.LISTA_model import train_lista


def generate_data(m=256, n=1024, s=0.1, N=5000):
    """

    :param m: signal length
    :param n: coefficient length
    :param s: sparsity level
    :param N: sample number
    :param load: if load current data
    :return:
    """
    np.random.seed(0)

    print('[INFO] Dictionary size [{}, {}], Sparsity {:.2f}, training samples {}'.format(m, n, s, N))

    # dimensions of the sparse signal, measurement and sparsity
    k = int(n * s)

    # generate dictionary
    Phi = np.random.randn(m, n)
    W_d = np.transpose(orth(np.transpose(Phi)))

    # generate sparse signal Z and measurement X
    X = np.zeros((N, n))
    Y = np.zeros((N, m))
    for i in range(N):
        index_k = np.random.choice(a=n, size=k, replace=False, p=None)
        X[i, index_k] = 5 * np.random.randn(k, 1).reshape([-1, ])
        Y[i] = np.dot(W_d, X[i, :])

    # show signal samples
    plt.figure()
    plt.subplot(211)
    plt.plot(range(Y[0].shape[0]), Y[0])
    plt.title('dense signal')
    plt.subplot(212)
    plt.plot(range(X[0].shape[0]), X[0])
    plt.title('sparse coefficients')
    plt.show()

    return Y, W_d, X


def run_1D_sparse_coding(m, n, s, n_sample):
    print('[INFO] Loading data ')
    Y, Wd, X = generate_data(m, n, s, n_sample)
    L = 2

    # ISTA
    Y_recon_ista, error_ista = run_ISTA(Y, Wd, s, L, maxiter=200)

    # LISTA
    Y_recon_lista, error_lista = run_LISTA(Y, Wd, s, L, epochs=200)

    plt.figure()
    plt.subplot(211)
    plt.plot(Y[0])
    plt.title('dense signal')
    plt.subplot(212)
    plt.plot(X[0], label="Signal")
    # plt.scatter(range(Y_recon_ista[0].shape[0]), Y_recon_ista[0], c='r', s=8, label="ISTA")
    # plt.scatter(range(Y_recon_lista[0].shape[0]), Y_recon_lista[0], c='g', marker='v', s=8, label="LISTA")
    plt.plot(Y_recon_ista[0], label="ISTA")
    plt.plot(Y_recon_lista[0], label="LISTA")
    plt.title('sparse coefficients')
    plt.legend()
    plt.show()


def run_ISTA(Y, Wd, s, L, maxiter, eps=0):
    print('[INFO] Ready to run ISTA')

    coef_recon, recon_errors, train_time = ista(np.mat(Y).T, np.mat(Wd), s, L, maxiter, eps)

    print('[INFO] ISTA finished. Error: {:.3f}. Time: {:.3f} s\n'.format(recon_errors, train_time))
    return coef_recon.T, recon_errors


def run_LISTA(Y, Wd, s, L, epochs=100):
    print('[INFO] Ready to train LISTA')
    model, coef_recon, recon_error, train_time = train_lista(Y, Wd, s, L, epochs)

    print('[INFO] LISTA finished. Error: {:.3f} Time: {:.3f} s'.format(recon_error, train_time))
    return coef_recon, recon_error

