import numpy as np
from sklearn.linear_model import orthogonal_mp_gram
from tqdm import tqdm

"""
    # sparse representations and compressed sensing
    #
    # raw data -> sensing matrix -> representation
    #
    # representation -> dictionary -> reconstructed data
"""

# generate random sampling matrix. Default binary.
def random_sensor(size, low=0, high=2):
    return np.random.randint(low, high, size=size)

# acquisition using sensing matrix
def sense(data, sensor):
    return np.dot(data,sensor)

# reconstruct
def reconstruct(measurement, sensor, dictionary, max_sparsity):
    """

    :param measurement: [image_nums, sensing_times]
    :param sensor:
    :param dictionary:
    :param max_sparsity: int
    :return:
    """
    measurement = measurement.T
        
    # reconstruction matrix, gram matrix
    SD = np.dot(sensor.T, dictionary)
    SD_norm = SD / np.linalg.norm(SD, axis=0)
    gram = np.dot(SD_norm.T, SD_norm)

    reconstruction = np.zeros([dictionary.shape[0], measurement.shape[1]])
    for col in range(measurement.shape[1]):
        print('  -> Reconstructing # {} image'.format(col))
        w = orthogonal_mp_gram(gram, np.dot(SD_norm.T, measurement[:,col]),
                               n_nonzero_coefs=int(max_sparsity))

        # img = dic * coef
        imgs_vec = np.matmul(dictionary, w)
        reconstruction[:, col] = imgs_vec

    return np.transpose(reconstruction)
