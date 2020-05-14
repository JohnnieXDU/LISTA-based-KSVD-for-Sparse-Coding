import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.utils_patches import im2patch
from skimage.restoration import inpaint


def generate_inpainting_data(missing_pixels=0.5, blksize=8, overlap=3, grayimg=False):
    img = Image.open('../dataset/got.png')
    img = img.resize((256, 256))
    img = np.array(img, dtype=np.float32)
    h, w, _ = img.shape

    mask = np.ones((h, w))
    mask_idx = np.random.randint(low=0, high=h*w, size=int(missing_pixels * h*w))
    mask.reshape(-1)[mask_idx] = 0
    blkmask_vecs = im2patch(mask, blksize=blksize, overlap=overlap)

    if grayimg:
        img = img.mean(axis=2)
        mask_img = mask * img

        patches_vecs = im2patch(mask_img, blksize=blksize, overlap=overlap, ifprint=True) * blkmask_vecs

        return img, mask_img, patches_vecs, blkmask_vecs
    else:
        mask_img = img * np.expand_dims(mask, axis=2).repeat(3, axis=2)
        mask_imgr = mask_img[:, :, 0]
        mask_imgg = mask_img[:, :, 1]
        mask_imgb = mask_img[:, :, 2]

        patchesR_vecs = im2patch(mask_imgr, blksize=blksize, overlap=overlap, ifprint=True)
        patchesG_vecs = im2patch(mask_imgg, blksize=blksize, overlap=overlap, ifprint=False)
        patchesB_vecs = im2patch(mask_imgb, blksize=blksize, overlap=overlap, ifprint=False)

        patches_vecs = np.zeros((patchesR_vecs.shape[0], patchesG_vecs.shape[1], 3), dtype=np.float32)
        patches_vecs[:, :, 0] = patchesR_vecs * blkmask_vecs
        patches_vecs[:, :, 1] = patchesG_vecs * blkmask_vecs
        patches_vecs[:, :, 2] = patchesB_vecs * blkmask_vecs

        return img, mask_img, patches_vecs, blkmask_vecs


def load_inpainting_image(blksize=8, overlap=3):
    img = Image.open('../dataset/Test_Fig2_Missing.png')

    img = np.array(img, dtype=np.float32)
    img = img.mean(axis=2)
    h, w = img.shape

    mask = img.copy()
    mask[mask>0] = 1

    mask_img = mask * img

    patches_vecs = im2patch(mask_img, blksize=blksize, overlap=overlap, ifprint=True)
    blkmask_vecs = im2patch(mask, blksize=blksize, overlap=overlap)

    # plt.figure()
    # plt.imshow(img, 'gray')
    # plt.figure()
    # plt.imshow(mask_img, 'gray')
    # plt.show()

    return img, mask_img, patches_vecs, blkmask_vecs