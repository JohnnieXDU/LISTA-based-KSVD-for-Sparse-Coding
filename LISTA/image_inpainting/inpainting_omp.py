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


def OrthogonalMatchingPursuit_Inpainting(D, Y, n_non_zeros, sigma=0.001, rc_min=0.01):
    """
    This OMP is more strict than official OMP.

    :param D:
    :param Y:
    :param n_non_zeros:
    :param sigma:
    :param rc_min:
    :return:
    """
    def norm_residual(x):
        return np.sqrt((x**2).sum())

    W = 1 / np.sqrt(np.diag(np.matmul(D.transpose(), D)))
    D = np.matmul(D, np.diag(W))

    residual = Y
    indx = []
    alpha = np.zeros(1)
    DD = np.zeros_like(D)
    j = 0
    rc_max = np.inf
    threshold = sigma * norm_residual(Y)

    while norm_residual(residual) > threshold and rc_max > rc_min and np.count_nonzero(alpha) < n_non_zeros:
        proj = np.matmul(D.transpose(), residual)
        rc_max, pos = np.abs(proj).max(), np.abs(proj).argmax()
        DD[:, j] =D[:, pos]
        indx.append(pos)
        alpha = np.matmul(np.linalg.pinv(DD[:, :j+1]), Y)
        residual = Y - np.matmul(DD[:, :j+1], alpha)
        j += 1

    coef = np.zeros((D.shape[1], 1))
    if len(indx) > 0:
        coef[indx] = alpha
        coef = W.reshape(W.shape[0], 1) * coef

        return coef

    return


def inpainting_omp(missing_pixels, dicsize, n_nonzero_coefs, blksize, overlap, maxiter, grayimg=False):
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

    load_results = True
    if load_results:
        print('[INFO] Load OMP results.')
        if grayimg:
            dict = np.load('./save_omp/D.npy')
            coef = np.load('./save_omp/coef.npy')
            recovered_img = patch2im_for_inpainting(patch_vecs=np.matmul(dict, coef), mask_vecs=maskvec,
                                                    imgsize=(h, w), blksize=blksize, overlap=overlap)
        else:
            dictR = np.load('./save_omp/DR.npy')
            dictG = np.load('./save_omp/DG.npy')
            dictB = np.load('./save_omp/DB.npy')
            coefR = np.load('./save_omp/coefR.npy')
            coefG = np.load('./save_omp/coefG.npy')
            coefB = np.load('./save_omp/coefB.npy')

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
            coef = inpainting_omp_once(Y, D, maskvec, n_nonzero_coefs=50)

            print('[INFO] Saving dictionary ...')
            np.save('./save_omp/D', D)

            print('[INFO] Saving coefficients ...')
            np.save('./save_omp/coef', coef)

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
                coef = inpainting_omp_once(Y, D, maskvec, n_nonzero_coefs=50)

                print('[INFO] Saving dictionary ...')
                np.save('./save_omp/D{}'.format(RGBname[channel]), D)

                print('[INFO] Saving coefficients ...')
                np.save('./save_omp/coef{}'.format(RGBname[channel]), coef)

                # recover data
                print('[INFO] Reconstruction ...')

                recovered_img_c = patch2im_for_inpainting(patch_vecs=np.matmul(D, coef), mask_vecs=maskvec, imgsize=img_ori.shape,
                                                          blksize=blksize, overlap=overlap)
                recovered_img[:, :, channel] = recovered_img_c

    return img_ori, masked_img, recovered_img


def inpainting_omp_once(Y, D, maskvec, n_nonzero_coefs, sigma=0.001, rc_min=0.01):
    coef = np.zeros((D.shape[1], Y.shape[1]))
    for k in tqdm(range(Y.shape[1])):
        nonzero_pos = np.nonzero(maskvec[:, k])[0]
        D_sub = D[nonzero_pos, :]
        Y_sub = Y[:, k:k + 1][nonzero_pos]

        # official OMP
        # omp = OrthogonalMatchingPursuit(n_nonzero_coefs)
        # omp.fit(D_sub, Y_sub)
        # coef[:, k] = omp.coef_

        # more strict version OMP
        coef[:, k] = OrthogonalMatchingPursuit_Inpainting(D_sub, Y_sub, n_nonzero_coefs, sigma=sigma, rc_min=rc_min)[:, 0]

    return coef
