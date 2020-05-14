import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.linear_model import OrthogonalMatchingPursuit
from dataset.generate_inpainting_data import generate_inpainting_data, load_inpainting_image
from ksvd.ksvd import ksvd
from tqdm import tqdm
from utils.utils_patches import patch2im_for_inpainting
from utils.utils_dict import get_init_dict, get_init_imagedict
import time
from utils.utils_error import l2_error


def shrinkage(x, theta):
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - theta, 0))


def ISTA_Inpainting(D, Y, n_non_zeros, max_iter, eps=0.1):
    eig, eig_vector = np.linalg.eig(D.T.dot(D))

    L = np.max(eig)  #  Lipschitz constant
    s = n_non_zeros / D.shape[1]

    W_e = D.T / L

    coef_old = np.zeros((D.shape[1], 1))
    coef_new = np.zeros((D.shape[1], 1))

    for i in range(max_iter):
        temp = D.dot(coef_old) - Y
        coef_new = shrinkage(coef_old - W_e.dot(temp), s / L)

        err = np.sum(np.abs(coef_new - coef_old))
        # print(err)
        if err <= eps:
            break
        coef_old = coef_new


    return coef_new

from scipy import linalg
def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)
def ISTA_Inpainting2(A, b, n_non_zeros, max_iter):
    l = n_non_zeros / A.shape[1]
    x = np.zeros(A.shape[1])
    pobj = []
    L = linalg.norm(A) ** 2  # Lipschitz constant
    time0 = time.time()
    for _ in range(max_iter):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, l / L)
        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times


def inpainting_ista(missing_pixels, dicsize, n_nonzero_coefs, blksize, overlap, maxiter, grayimg=False):

    img_ori, masked_img, patchvec, maskvec = generate_inpainting_data(missing_pixels=missing_pixels,
                                                                      blksize=blksize, overlap=overlap, grayimg=grayimg)
    # img_ori, masked_img, patchvec, maskvec = load_inpainting_image(blksize=blksize, overlap=overlap)

    # settings
    h, w = img_ori.shape[0], img_ori.shape[1]
    m = patchvec.shape[0]
    n = dicsize

    print('[INFO] Dictionary size ({}, {}).'.format(m, n))

    print('[INFO] Sparsity {}.'.format(n_nonzero_coefs))
    print('[INFO] OMP max iter {}.'.format(maxiter))

    load_results = False
    if load_results:
        print('[INFO] Load OMP results.')
        if grayimg:
            dict = np.load('./D.npy')
            coef = np.load('./coef.npy')
            recovered_img = patch2im_for_inpainting(patch_vecs=np.matmul(dict, coef), mask_vecs=maskvec,
                                                    imgsize=(h, w), blksize=blksize, overlap=overlap)
        else:
            dictR = np.load('./DR.npy')
            dictG = np.load('./DG.npy')
            dictB = np.load('./DB.npy')
            coefR = np.load('./coefR.npy')
            coefG = np.load('./coefG.npy')
            coefB = np.load('./coefB.npy')

            recovered_imgR = patch2im_for_inpainting(patch_vecs=np.matmul(dictR, coefR), mask_vecs=maskvec,
                                                     imgsize=(h, w), blksize=blksize, overlap=overlap)
            recovered_imgG = patch2im_for_inpainting(patch_vecs=np.matmul(dictG, coefG), mask_vecs=maskvec,
                                                     imgsize=(h, w), blksize=blksize, overlap=overlap)
            recovered_imgB = patch2im_for_inpainting(patch_vecs=np.matmul(dictB, coefB), mask_vecs=maskvec,
                                                     imgsize=(h, w), blksize=blksize, overlap=overlap)
            recovered_img = np.stack((recovered_imgR, recovered_imgG, recovered_imgB), axis=2)

        return img_ori, masked_img, recovered_img

    else:
        # OMP for inpainting
        print('[INFO] Run OMP for image inpainting')

        if grayimg:
            D = get_init_dict(dicsize=dicsize, blksize=blksize)
            Y = patchvec
            coef = inpainting_ista_once(Y, D, maskvec, n_nonzero_coefs=50)

            print('[INFO] Saving dictionary ...')
            np.save('./D', D)

            print('[INFO] Saving coefficients ...')
            np.save('./coef', coef)

            # recover data
            print('[INFO] Reconstruction ...')

            recovered_img = patch2im_for_inpainting(patch_vecs=np.matmul(D, coef), mask_vecs=maskvec, imgsize=img_ori.shape,
                                                    blksize=blksize, overlap=overlap)
        else:
            D = get_init_dict(dicsize=dicsize, blksize=blksize)
            RGBname = ['R', 'G', 'B']
            recovered_img = np.zeros_like(img_ori)
            # do R-G-B
            for channel in range(3):
                Y = patchvec[:, :, channel]
                coef = inpainting_ista_once(Y, D, maskvec, n_nonzero_coefs=50)

                print('[INFO] Saving dictionary ...')
                np.save('./D{}'.format(RGBname[channel]), D)

                print('[INFO] Saving coefficients ...')
                np.save('./coef{}'.format(RGBname[channel]), coef)

                # recover data
                print('[INFO] Reconstruction ...')

                recovered_img_c = patch2im_for_inpainting(patch_vecs=np.matmul(D, coef), mask_vecs=maskvec, imgsize=img_ori.shape,
                                                          blksize=blksize, overlap=overlap)
                recovered_img[:, :, channel] = recovered_img_c

    return img_ori, masked_img, recovered_img

import pylops
def inpainting_ista_once(Y, D, maskvec, n_nonzero_coefs, sigma=0.001, rc_min=0.01):
    coef = np.zeros((D.shape[1], Y.shape[1]))
    for k in tqdm(range(Y.shape[1])):
        nonzero_pos = np.nonzero(maskvec[:, k])[0]
        D_sub = D[nonzero_pos, :]
        Y_sub = Y[:, k:k + 1][nonzero_pos]

        # ISTA
        # coef[:, k] = ISTA_Inpainting2(D_sub, Y_sub, n_nonzero_coefs, max_iter=1000)[:, 0]
        coef[:, k] = pylops.optimization.sparsity.ISTA(pylops.MatrixMult(D_sub), Y_sub[:, 0], 5000, eps=1e-2, tol=1e-3, returninfo=True)[0]

    return coef
