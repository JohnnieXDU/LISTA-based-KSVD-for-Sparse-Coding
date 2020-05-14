import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


img_ori = np.load('img_ori.npy')
masked_img = np.load('masked_img.npy')
recovered_img = np.load('recovered_img.npy')

psnr = PSNR(img_ori, recovered_img)

plt.figure()
plt.imshow(img_ori, 'gray')
plt.title('original image')
plt.figure()
plt.imshow(masked_img, 'gray')
plt.title('missing pixels')
plt.figure()
plt.imshow(recovered_img, 'gray')
plt.title('image inpainting PSNR={:.2f}'.format(psnr))
plt.show()