import torch
import numpy as np
from dataset.generate_inpainting_data import generate_inpainting_data, load_inpainting_image
from utils.utils_dict import get_init_dict, get_init_imagedict
from utils.utils_patches import patch2im_for_inpainting
from models.LISTA_model import LISTA

import torch.nn as nn
import os


def inpainting_lista(missing_pixels, dicsize, n_nonzero_coefs, blksize, overlap, maxiter, grayimg):


    img_ori, masked_img, patchvec, maskvec = generate_inpainting_data(missing_pixels=missing_pixels,
                                                                      blksize=blksize, overlap=overlap, grayimg=grayimg)

    # settings
    h, w = img_ori.shape[0], img_ori.shape[1]
    m = patchvec.shape[0]
    n = dicsize
    s = n_nonzero_coefs / n

    print('[INFO] Dictionary size ({}, {}).'.format(m, n))

    print('[INFO] Sparsity {}.'.format(n_nonzero_coefs))
    print('[INFO] OMP max iter {}.'.format(maxiter))

    D = get_init_dict(dicsize=dicsize, blksize=blksize)
    Y = patchvec

    # training
    model = train_lista(D, Y, maskvec, s)

    # testing
    with torch.no_grad():
        Y_pt = torch.from_numpy(Y.T/255.0).cuda()
        coef_pt = model(Y_pt)
        coef = coef_pt.cpu().detach().numpy().T

    # testing
    recovered_img = patch2im_for_inpainting(patch_vecs=255.0*np.matmul(D, coef), mask_vecs=maskvec, imgsize=img_ori.shape,
                                            blksize=blksize, overlap=overlap)

    # save results
    if not os.path.exists('./save_lista'):
        os.mkdir('./save_lista')

    np.save('./save_lista/img_ori', img_ori)
    np.save('./save_lista/masked_img', masked_img)
    np.save('./save_lista/recovered_img', recovered_img)

    return img_ori, masked_img, recovered_img


def train_lista(D, Y, maskvec, s, epochs=50):
    """
    Trianing LISTA network.

    :param D: size [m, n], e.g [64, 256]
    :param Y: size [64, K], e.g [64, 10000]
    :param s:
    :param epochs:
    :return:
    """

    # pre-settings
    bs = 128
    learning_rate = 1e-2
    K = Y.shape[1]
    iter_epoch = K // bs

    eig, eig_vector = np.linalg.eig(D.T.dot(D))
    L = np.max(eig)  # Lipschitz constant

    theta = s/L
    m, n = D.shape

    # normalize image
    Y = (Y / 255.0) * maskvec

    # convert the data into tensors
    Y = Y.T  # batch should be the 0-dim in pytorch
    Y = torch.from_numpy(Y).float()
    D = torch.from_numpy(D).float().cuda()
    maskvec = torch.from_numpy(maskvec.T).cuda()

    # network initialization
    net = LISTA(m, n, D, L, theta, max_iter=30)
    net.weights_init()
    net = net.cuda()

    # build the optimizer and criterion
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    all_zeros = torch.zeros(bs, n).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for i_epoch in range(epochs):
        indexs = np.arange(K)
        np.random.shuffle(indexs)

        epoch_loss = 0
        epoch_loss1 = 0
        epoch_loss2 = 0
        for i_iter in range(iter_epoch):
            Y_batch = Y[indexs[i_iter * bs:(i_iter + 1) * bs]].cuda()
            mask = maskvec[indexs[i_iter * bs:(i_iter + 1) * bs]]

            # get the outputs
            X = net(Y_batch)
            Y_recons = torch.mm(X, D.T)

            # compute the losss
            loss1 = criterion1(Y_batch*mask, Y_recons*mask)
            loss2 = (1-s) * criterion2(X, all_zeros)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            epoch_loss += loss.item()

        print('[INFO] Epoch: {}/{} loss: {:.3f} ({:.3f}, {:.3f})'.format(i_epoch, epochs,
                                                                         epoch_loss/iter_epoch,
                                                                         epoch_loss1/iter_epoch, epoch_loss2/iter_epoch))


    return net