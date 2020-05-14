import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from utils.utils_error import l2_error_torch


class LISTA(nn.Module):
    def __init__(self, m, n, W_e, L, theta, max_iter):
        """
        # Arguments
            m: int, dimensions of the measurement
            n: int, dimensions of the sparse signal
            W_e: array, dictionary
            max_iter:int, max number of internal iteration
            L: Lipschitz const
            theta: Thresholding
        """

        super(LISTA, self).__init__()
        self._W = nn.Linear(in_features=m, out_features=n, bias=False)
        self._S = nn.Linear(in_features=n, out_features=n,
                            bias=False)
        self.shrinkage = nn.Softshrink(theta)
        self.theta = theta
        self.max_iter = max_iter
        self.A = W_e
        self.L = L

    # weights initialization based on the dictionary
    def weights_init(self):
        A = self.A.cpu().numpy()
        L = self.L
        S = torch.from_numpy(np.eye(A.shape[1]) - (1 / L) * np.matmul(A.T, A))
        S = S.float().cuda()
        W = torch.from_numpy((1 / L) * A.T)
        W = W.float().cuda()

        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(W)

    def forward(self, y):
        """

        :param y: [bs, m]
        :return:
        """
        x = self.shrinkage(self._W(y))

        if self.max_iter == 1:
            return x

        for iter in range(self.max_iter):
            x = self.shrinkage(self._W(y) + self._S(x))

        return x




def train_lista(Y, dictionary, a, L, epochs=100):
    m, n = dictionary.shape
    n_samples = Y.shape[0]
    batch_size = n_samples
    steps_per_epoch = n_samples // batch_size

    # convert the data into tensors
    Y = torch.from_numpy(Y).float().cuda()
    W_d = torch.from_numpy(dictionary).float().cuda()
    theta = a/L
    net = LISTA(m, n, W_d, L, theta, max_iter=30)
    net.weights_init()
    net = net.cuda()

    # build the optimizer and criterion
    learning_rate = 1e-2
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    all_zeros = torch.zeros(batch_size, n).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    end = time.time()
    for epoch in tqdm(range(epochs)):
        index_samples = np.random.choice(a=n_samples, size=n_samples, replace=False, p=None)
        Y_shuffle = Y[index_samples]
        for step in range(steps_per_epoch):
            Y_batch = Y_shuffle[step * batch_size:(step + 1) * batch_size]
            optimizer.zero_grad()

            # get the outputs
            X_h = net(Y_batch)
            Y_h = torch.mm(X_h, W_d.T)

            # compute the losss
            loss1 = criterion1(Y_batch.float(), Y_h.float())
            loss2 = a * criterion2(X_h.float(), all_zeros.float())
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

    training_time = time.time() - end

    # reconstruction error
    coef = net(Y)
    Y_recon = torch.mm(W_d, torch.transpose(coef, 1, 0))
    recon_error = l2_error_torch(Y.transpose(0, 1), Y_recon)

    return net, coef.cpu().detach().numpy(), recon_error, training_time
