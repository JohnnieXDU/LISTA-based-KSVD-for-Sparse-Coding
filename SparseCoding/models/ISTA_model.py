import numpy as np
from tqdm import tqdm
import time

from utils.utils_error import l2_error


def shrinkage(x, theta):
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - theta, 0))


def ista(Y, W_d, a, L, max_iter, eps):
    eig, eig_vector = np.linalg.eig(W_d.T * W_d)
    assert L > np.max(eig)
    del eig, eig_vector

    W_e = W_d.T / L

    recon_error = 0.0
    Z_old = np.zeros((W_d.shape[1], 1))

    end = time.time()
    for i in tqdm(range(max_iter)):
        temp = W_d * Z_old - Y
        Z_new = shrinkage(Z_old - W_e * temp, a / L)
        if np.sum(np.abs(Z_new - Z_old)) <= eps:
            break
        Z_old = Z_new

        # compute error
        recon_error = l2_error(Y, W_d*Z_new)
        # print(recon_error)

    training_time = time.time() - end
    return np.array(Z_new), recon_error, training_time
