import numpy as np


def get_init_dict(dicsize, blksize):
    """
    Generate DCT dictionary.

    :param dicsize: size of dictionary.
    :param blksize: size of image patch.
    :return:
    """

    DCT = np.zeros((blksize, int(np.sqrt(dicsize))), dtype=np.float32)
    for k in range(int(np.sqrt(dicsize))):
        x = np.arange(blksize) * k * np.pi / np.sqrt(dicsize)
        v = np.cos(x)
        if k > 0:
            v = v - v.mean()

        DCT[:, k] = v / np.linalg.norm(v, 2)

    D = np.kron(DCT, DCT)

    return D


def get_init_imagedict(dicsize, blksize, image):
    """
    Generate dictionary from image.

    """
    h, w = image.shape[0], image.shape[1]
    D = np.zeros((blksize**2, dicsize), dtype=np.float32)

    pi = np.random.randint(low=0, high=h - blksize, size=dicsize)
    pj = np.random.randint(low=0, high=w - blksize, size=dicsize)

    for idx in range(dicsize):
        pii = pi[idx]
        pjj = pj[idx]
        D[:, idx:idx+1] = image[pii:pii+blksize, pjj:pjj+blksize].reshape((blksize**2, 1))

    # normalize
    # D = D / np.linalg.norm(D, axis=0)

    return D