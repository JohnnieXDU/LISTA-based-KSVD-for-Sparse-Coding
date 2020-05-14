import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.linear_model import OrthogonalMatchingPursuit

from ksvd.ksvd import ksvd
from tqdm import tqdm

from dataset.generate_denoising_data import generate_denoising_data
from utils.utils_patches import patch2im_for_denoising
from utils.utils_dict import get_init_dict


def denoising_omp(dicsize, sparsity, blksize, overlap, maxiter):

    np.random.seed(0)

    img_ori, masked_img, mask, patchvec, maskvec = generate_denoising_data(blksize, overlap)

    # settings
    m = patchvec.shape[0]
    n = dicsize

    print('[INFO] Dictionary size ({}, {}).'.format(m, n))
    print('[INFO] Sparsity {}.'.format(sparsity))
    print('[INFO] De-noising max iter {}.'.format(maxiter))

    load_results = False
    if load_results:
        print('[INFO] Load OMP results.')
        dict = np.load('./dict.npy')
        coef = np.load('./coef.npy')

        # reconstruct image
        print('[INFO] Reconstruction ...')
        recovered_img = patch2im_for_denoising(image=img_ori, patch_vecs=np.matmul(dict, coef), imgsize=img_ori.shape,
                                               blksize=blksize, overlap=overlap)

        return img_ori, masked_img, recovered_img

    else:
        # OMP for de-noising
        print('[INFO] Run OMP for denoising')

        recovered_img = None
        for iter in range(maxiter):
            D = get_init_dict(dicsize=dicsize, blksize=blksize)
            dict, coef = denoise_omp_once(patchvec, D, sparsity)

            np.save('./dict', dict)
            np.save('./coef', coef)

            # reconstruct image
            print('[INFO] Reconstruction ...')
            recovered_img = patch2im_for_denoising(image=img_ori, patch_vecs=np.matmul(dict, coef), imgsize=img_ori.shape,
                                                   blksize=blksize, overlap=overlap)

        return img_ori, masked_img, recovered_img


def denoise_omp_once(Y, D, sparsity, maxiter):
    """
    Run OMP denoising.

    :param Y: data
    :param D: dictionary
    :param sparsity: sparsity (between 0-1) used for OMP
    :param maxiter: max iteration

    :return:
        - coef: sparse coefficients

    """
    coef = np.zeros((D.shape[1], Y.shape[1]))

    # fill the codes ...

    return coef