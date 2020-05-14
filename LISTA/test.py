import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')
np.random.seed(0)

N, M = 15, 20
A = np.random.randn(N, M)
A = A / np.linalg.norm(A, axis=0)
Aop = pylops.MatrixMult(A)

x = np.random.rand(M)
x[x < 0.9] = 0
y = Aop*x

# ISTA
eps = 1e-2
maxit = 500
x_ista = pylops.optimization.sparsity.ISTA(Aop, y, maxit, eps=eps, tol=1e-3, returninfo=True)[0]

print('x')