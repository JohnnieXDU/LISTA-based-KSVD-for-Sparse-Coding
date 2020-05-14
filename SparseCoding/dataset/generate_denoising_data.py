import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.utils_patches import im2patch


def generate_noise():
    """
    TODO
    """
    print('[Need to to ...] generate noise for image.')
    noise = None

    # fill the code ...

    return noise


def generate_denoising_data(blksize=8, overlap=3):
    img = Image.open('dataset/got.png')

    img = np.array(img, dtype=np.float32)
    img = img.mean(axis=2)  # only use gray image for simplicity
    h, w = img.shape

    # generate noise
    noise = generate_noise()
    if noise is None:
        raise Exception('Fill the code: generate noise for image.')

    noisy_img = noise * img

    patches_vecs = im2patch(noisy_img, blksize=blksize, overlap=overlap, ifprint=True)

    # plt.figure()
    # plt.imshow(img, 'gray')
    # plt.figure()
    # plt.imshow(mask_img, 'gray')
    # plt.show()

    return img, noisy_img, patches_vecs