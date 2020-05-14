import numpy as np
import matplotlib.pyplot as plt
from image_denoising import denoise_omp


if __name__ == '__main__':
    dicsize = 256  # dictionary size (number of atoms)
    sparsity = 0.2
    blksize = 8  # image patch size
    overlap = 3  # image patch overlap stride
    maxiter = 10
    img_ori, masked_img, recovered_img = denoise_omp.denoising_omp(dicsize, sparsity, blksize, overlap, maxiter)