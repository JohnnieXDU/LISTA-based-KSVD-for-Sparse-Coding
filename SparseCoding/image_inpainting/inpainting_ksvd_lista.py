import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.linear_model import OrthogonalMatchingPursuit
from dataset.generate_inpainting_data import generate_inpainting_data, load_inpainting_image
from ksvd.ksvd import ksvd
from tqdm import tqdm
from utils.utils_patches import patch2im_for_inpainting
from utils.utils_dict import get_init_dict, get_init_imagedict
from image_inpainting.inpainting_omp import OrthogonalMatchingPursuit_Inpainting, inpainting_omp_once
from image_inpainting.inpainting_lista import train_lista
import torch


def KSVD_LISTA_Inpainting(D, Y, maskvec, n_non_zeros, img_ori, blksize, overlap, maxiter=10, etol=1e-10, approx=False):
    """

        Returns:
            D:               learned dictionary
            X:               sparse coding of input data
            error_norms:     array of training errors for each iteration
        Task: find best dictionary D to represent Data Y;
              minimize squared norm of Y - DX, constraining
              X to sparse codings.
    """

    # X = np.zeros([D.shape[1], Y.shape[1]])  # coefficients

    error_norms = []

    print('[INFO] Start KSVD ...')
    for iteration in range(1, maxiter + 1):
        end = time.time()
        print('  -> update coefficient: LISTA')

        model = train_lista(D, Y, maskvec, n_non_zeros / D.shape[1], img_ori, blksize, overlap)
        with torch.no_grad():
            Y_pt = torch.from_numpy(Y.T / 255.0).cuda()
            coef_pt = model(Y_pt)
            X = coef_pt.cpu().detach().numpy().T * 255

        # X = np.load('debug_X.npy')
        # X = inpainting_omp_once(Y, D, maskvec, n_non_zeros, sigma=0.001, rc_min=0.01)

        print('  -> update dictionary: SVD')
        for j in tqdm(range(D.shape[1])):
            # index set of nonzero components
            index_set = np.nonzero(X[j, :])[0]
            if len(index_set) == 0:
                # for now, replace with some white noise
                if not approx:
                    D[:, j] = np.random.randn(*D[:, j].shape)
                    D[:, j] = D[:, j] / np.linalg.norm(D[:, j])
                continue
            # approximate K-SVD update
            if approx:
                coef = X.copy()
                coef[j, :] = 0
                E = Y[:, index_set] - D.dot(coef[:, index_set])
                E = E * maskvec[:, index_set]

                dd = E.dot(X[j:j + 1, index_set].transpose())
                dd_w = maskvec[:, index_set].dot((X[j:j + 1, index_set] ** 2).T)

                dd = dd / dd_w
                dd[np.where(dd_w == 0)] = 0

                D[:, j:j + 1] = dd / np.linalg.norm(dd)

                w = maskvec[:, index_set] * (dd.dot(np.ones((1, len(index_set)))))
                X[j:j + 1, index_set] = dd.T.dot(E) / (w ** 2).sum(axis=0, keepdims=True)

                # D[:, j] = E.dot(X[j, index_set])  # update D
                # D[:, j] /= np.linalg.norm(D[:, j])
                # X[j, index_set] = (E.T).dot(D[:, j])  # update X
            else:
                # error matrix E on unmissing pixels

                E_idx = np.delete(range(D.shape[1]), j, 0)

                row_set = E_idx
                col_set = index_set

                E = Y[:, col_set] - np.dot(D[:, E_idx], X[E_idx, :][:, col_set])
                E = E * maskvec[:, col_set]

                U, S, VT = np.linalg.svd(E[:, col_set])
                # update jth column of D
                D[:, j] = U[:, 0]
                # update sparse elements in jth row of X
                X[j, :] = np.array([
                    S[0] * VT[0, np.argwhere(index_set == n)[0][0]]
                    if n in index_set else 0
                    for n in range(X.shape[1])])
        # stopping condition: check error
        err = np.linalg.norm(Y - D.dot(X), 'fro')
        error_norms.append(err)
        if err < etol:
            break

        print('[INFO] KSVD finished. Total iter={}, time={}'.format(iteration, time.time() - end))

    return D, X, np.array(error_norms)


def inpainting_ksvd_lista(missing_pixels, dicsize, n_nonzero_coefs, blksize, overlap, maxiter, grayimg=False):
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
            newD, coef = inpainting_ksvd_lista_once(Y, D, maskvec, img_ori, blksize, overlap, n_nonzero_coefs=50,
                                                    maxiter=maxiter)

            print('[INFO] Saving dictionary ...')
            np.save('./D', newD)

            print('[INFO] Saving coefficients ...')
            np.save('./coef', coef)

            # recover data
            print('[INFO] Reconstruction ...')

            recovered_img = patch2im_for_inpainting(patch_vecs=np.matmul(newD, coef), mask_vecs=maskvec,
                                                    imgsize=img_ori.shape,
                                                    blksize=blksize, overlap=overlap)
        else:
            D = get_init_dict(dicsize=dicsize, blksize=blksize)
            RGBname = ['R', 'G', 'B']
            recovered_img = np.zeros_like(img_ori)
            # do R-G-B
            for channel in range(3):
                Y = patchvec[:, :, channel]
                newD, coef = inpainting_ksvd_lista_once(Y, D, maskvec, img_ori, blksize, overlap, n_nonzero_coefs=50,
                                                        maxiter=maxiter)

                print('[INFO] Saving dictionary ...')
                np.save('./D{}'.format(RGBname[channel]), newD)

                print('[INFO] Saving coefficients ...')
                np.save('./coef{}'.format(RGBname[channel]), coef)

                # recover data
                print('[INFO] Reconstruction ...')

                recovered_img_c = patch2im_for_inpainting(patch_vecs=np.matmul(newD, coef), mask_vecs=maskvec,
                                                          imgsize=img_ori.shape,
                                                          blksize=blksize, overlap=overlap)
                recovered_img[:, :, channel] = recovered_img_c

    return img_ori, masked_img, recovered_img


def inpainting_ksvd_lista_once(Y, D, maskvec, img_ori, blksize, overlap, n_nonzero_coefs, maxiter):
    # KSVD to update dict and coef
    new_D, new_coef, error = KSVD_LISTA_Inpainting(D, Y, maskvec, n_nonzero_coefs,
                                                   img_ori, blksize, overlap,
                                                   maxiter=maxiter, etol=1e-10, approx=True)

    # coef = np.zeros((D.shape[1], Y.shape[1]))
    # for k in tqdm(range(Y.shape[1])):
    #     nonzero_pos = np.nonzero(maskvec[:, k])[0]
    #     D_sub = D[nonzero_pos, :]
    #     Y_sub = Y[:, k:k + 1][nonzero_pos]
    #
    #
    #     # update
    #     D[nonzero_pos, :] = new_D_sub
    #     coef[:, k] = new_coef[:, 0]

    return new_D, new_coef
