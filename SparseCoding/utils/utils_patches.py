import numpy as np


def im2patch(img, blksize, overlap, ifprint=False):
    h, w = img.shape

    # determine the max size when cropping patches
    hh = int(np.ceil((h-blksize)/overlap)*overlap+blksize)
    ww = int(np.ceil((w-blksize)/overlap)*overlap+blksize)

    img_new = np.zeros((hh, ww))
    img_new[0:h, 0:w] = img

    # create patches
    patch_num_h = int((hh - blksize) / overlap)+1
    patch_num_w = int((ww - blksize) / overlap)+1

    patch_num = int(patch_num_h * patch_num_w)
    patches_vecs = np.zeros((blksize*blksize, patch_num), dtype=np.float32)
    idx = 0
    for pi in range(patch_num_h):
        for pj in range(patch_num_w):
            i = pi * overlap
            j = pj * overlap
            patches_vecs[:, idx] = img_new[i:i+blksize, j:j+blksize].reshape(-1)
            idx += 1

    if ifprint:
        print('[INFO] Patch size ({}, {}).'.format(blksize, blksize))
        print('[INFO] Patch num {}.'.format(idx))

    return patches_vecs


def patch2im_for_inpainting(patch_vecs, mask_vecs, imgsize, blksize, overlap):

    h, w = imgsize[0], imgsize[1]

    hh = int(np.ceil((h-blksize)/overlap)*overlap+blksize)
    ww = int(np.ceil((w-blksize)/overlap)*overlap+blksize)

    # create patches
    patch_num_h = int((hh - blksize) / overlap)+1
    patch_num_w = int((ww - blksize) / overlap)+1

    wmap = np.zeros((hh, ww))
    imgr = np.zeros((hh, ww))

    idx = 0
    for pi in range(patch_num_h):
        for pj in range(patch_num_w):
            i = pi * overlap
            j = pj * overlap

            imgr[i:i+blksize, j:j+blksize] = imgr[i:i+blksize, j:j+blksize] + patch_vecs[:, idx].reshape(blksize, blksize)
            wmap[i:i+blksize, j:j+blksize] += 1
            idx += 1

    wmap[wmap == 0] = 1
    imgr /= wmap

    return imgr[:h, :w]


def patch2im_for_denoising(image, patch_vecs, imgsize, blksize, overlap):
    """

    This function is dangerous!!!
    i haven't check it for denoising. Ruizhi should check it carefully.

    """

    h, w = imgsize[0], imgsize[1]

    hh = int(np.ceil((h-blksize)/overlap)*overlap+blksize)
    ww = int(np.ceil((w-blksize)/overlap)*overlap+blksize)

    # create patches
    patch_num_h = int((hh - blksize) / overlap)+1
    patch_num_w = int((ww - blksize) / overlap)+1

    wmap = np.zeros((hh, ww))
    imgr = image.copy()

    idx = 0
    for pi in range(patch_num_h):
        for pj in range(patch_num_w):
            i = pi * overlap
            j = pj * overlap

            imgr[i:i+blksize, j:j+blksize] += patch_vecs[:, idx].reshape(blksize, blksize)
            wmap[i:i+blksize, j:j+blksize] += 1
            idx += 1

    wmap[wmap == 0] = 1
    imgr /= wmap

    return imgr[:h, :w]