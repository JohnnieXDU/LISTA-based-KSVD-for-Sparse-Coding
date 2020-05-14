import numpy as np
import matplotlib.pyplot as plt
from image_inpainting import inpainting_omp, inpainting_ksvd_omp, inpainting_ista, inpainting_lista, inpainting_ksvd_lista, inpainting_lstm
from utils.utils_error import PSNR
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    grayimg = True
    missing_pixels = 0.5
    dicsize = 256  # dictionary size (number of atoms)
    n_nonzero_coefs = 50
    blksize = 8  # image patch size
    overlap = 3  # image patch overlap stride
    maxiter = 5

    method = 'LISTA_LSTM'

    if method == 'OMP':
        img_ori, masked_img, recovered_img = inpainting_omp.inpainting_omp(missing_pixels, dicsize, n_nonzero_coefs,
                                                                           blksize, overlap, maxiter, grayimg)

    if method == 'KSVD_OMP':
        img_ori, masked_img, recovered_img = inpainting_ksvd_omp.inpainting_ksvd(missing_pixels, dicsize, n_nonzero_coefs,
                                                                                 blksize, overlap, maxiter, grayimg)

    if method == 'ISTA':
        img_ori, masked_img, recovered_img = inpainting_ista.inpainting_ista(missing_pixels, dicsize, n_nonzero_coefs,
                                                                             blksize, overlap, maxiter, grayimg)
    if method == 'LISTA':
        img_ori, masked_img, recovered_img = inpainting_lista.inpainting_lista(missing_pixels, dicsize, n_nonzero_coefs,
                                                                               blksize, overlap, maxiter, grayimg)

    if method == 'KSVD_LISTA':
        img_ori, masked_img, recovered_img = inpainting_ksvd_lista.inpainting_ksvd_lista(missing_pixels, dicsize, n_nonzero_coefs,
                                                                                 blksize, overlap, maxiter, grayimg)

    if method == 'LISTA_LSTM':
        img_ori, masked_img, recovered_img = inpainting_lstm.inpainting_lstm(missing_pixels, dicsize, n_nonzero_coefs,
                                                                                 blksize, overlap, maxiter, grayimg)

    # PSNR
    psnr = PSNR(img_ori, recovered_img)
    print('PSNR={:.2f}'.format(psnr))

    if grayimg:
        plt.figure()
        plt.imshow(img_ori, 'gray')
        plt.title('original image')
        plt.figure()
        plt.imshow(masked_img, 'gray')
        plt.title('missing pixels {:.2f}%'.format(missing_pixels*100))
        plt.figure()
        plt.imshow(recovered_img, 'gray')
        plt.title('image inpainting PSNR={:.2f}'.format(psnr))
        plt.show()
    else:
        plt.figure()
        plt.imshow(img_ori.astype(np.uint8))
        plt.title('original image')
        plt.figure()
        plt.imshow(masked_img.astype(np.uint8))
        plt.title('missing pixels {:.2f}%'.format(missing_pixels*100))
        plt.figure()
        plt.imshow(recovered_img.astype(np.uint8))
        plt.title('image inpainting PSNR={:.2f}'.format(psnr))
        plt.show()
