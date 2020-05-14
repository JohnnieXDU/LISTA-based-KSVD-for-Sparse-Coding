from ksvd.sparseRep import random_sensor, sense, reconstruct


def reconstruction_MNIST(data, D, sparsity, sense_size):
    signal_length = data.shape[1]
    # compressive sensing
    sensor = random_sensor((signal_length, sense_size))
    measurement = sense(data, sensor)

    # sparse reconstruction
    imgs_recon = reconstruct(measurement, sensor, D, sparsity)

    return data.reshape([data.shape[0], 28, 28]), imgs_recon.reshape([imgs_recon.shape[0], 28, 28])
