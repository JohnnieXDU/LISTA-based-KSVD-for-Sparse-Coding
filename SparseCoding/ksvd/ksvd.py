import numpy as np
from sklearn.linear_model import orthogonal_mp_gram
from tqdm import tqdm
import time

def ksvd(Data, num_atoms, sparsity, initial_D=None,
    maxiter=10, etol=1e-10, approx=False, maskvec=None):
    """
        K-SVD for Overcomplete Dictionary Learning
        Author: Alan Yang - Fall 2017

        See:
            M. Aharon, M. Elad and A. Bruckstein, "K-SVD: An 
            Algorithm for Designing Overcomplete Dictionaries 
            for Sparse Representation," in IEEE Transactions
            on Signal Processing, vol. 54, no. 11, pp. 4311-4322, 
            Nov. 2006.
            
            Rubinstein, R., Zibulevsky, M. and Elad, M., 
            "Efficient Implementation of the K-SVD Algorithm 
            using Batch Orthogonal Matching Pursuit Technical 
            Report" - CS Technion, April 2008.
                
        Data:       rows hold training data for dictionary fitting
        num_atoms:  number of dictionary atoms
        sparsity:   max sparsity of signals. Reduces to K-means
                    when sparsity=1
        initial_D:  if given, an initial dictionary. Otherwise, random
                    rows of data are chosen for initial dictionary
        maxiter:    maximum number of iterations
        err_thresh: stopping criteria; minimum residual
        approx:     True if using approximate KSVD update method.
                    Code runs faster if True, but results generally
                    in higher training error.
        
        Returns:
            D:               learned dictionary
            X:               sparse coding of input data
            error_norms:     array of training errors for each iteration
        Task: find best dictionary D to represent Data Y;
              minimize squared norm of Y - DX, constraining
              X to sparse codings.
    """

    assert Data.shape[1] > num_atoms # enforce this for now

    # intialization
    if initial_D is not None: 
        # D = initial_D / np.linalg.norm(initial_D, axis=0)
        D = initial_D
        Y = Data
        X = np.zeros([num_atoms, Data.shape[1]])
    else:
        # randomly select initial dictionary from data
        idx_set = range(Data.shape[1])
        idxs = np.random.choice(idx_set, num_atoms, replace=False)    
        # Y = Data[:,np.delete(idx_set, idxs)]
        Y = Data
        X = np.zeros([num_atoms, Data.shape[1] - num_atoms])
        D = Data[:,idxs] / np.linalg.norm(Data[:,idxs], axis=0)

    # repeat until convergence or stopping criteria
    error_norms = []

    print('[INFO] Start KSVD ...')
    for iteration in range(1, maxiter+1):
        end = time.time()
        # sparse coding stage: estimate columns of X
        print('  -> update coefficient: OMP')
        if maskvec is None:
            gram = (D.T).dot(D)
            Dy = (D.T).dot(Y)
            X = orthogonal_mp_gram(gram, Dy, n_nonzero_coefs=int(num_atoms*sparsity))
        else:
            X = np.zeros((D.shape[1], Y.shape[1]))
            for k in tqdm(range(Y.shape[1])):
                nonzero_pos = np.nonzero(maskvec[:, k])[0]
                nonzero_num = nonzero_pos.shape[0]
                D_sub = D[nonzero_pos, :]
                Y_sub = Y[:, k:k+1][nonzero_pos]

                gram_sub = (D_sub.T).dot(D_sub)
                Dy_sub = (D_sub.T).dot(Y_sub)
                X_sub = orthogonal_mp_gram(gram_sub, Dy_sub, n_nonzero_coefs=int(nonzero_num*sparsity))
                X[:, k] = X_sub

        print('  -> update dictionary: SVD')
        # codebook update stage
        for j in tqdm(range(D.shape[1])):
            # index set of nonzero components
            index_set = np.nonzero(X[j, :])[0]
            if len(index_set) == 0:
                # for now, replace with some white noise
                if not approx:
                    D[:,j] = np.random.randn(*D[:,j].shape)
                    D[:,j] = D[:,j] / np.linalg.norm(D[:,j])
                continue
            # approximate K-SVD update
            if approx:
                E = Y[:,index_set] - D.dot(X[:,index_set])
                D[:,j] = E.dot(X[j,index_set])     # update D
                D[:,j] /= np.linalg.norm(D[:,j])
                X[j,index_set] = (E.T).dot(D[:,j]) # update X
            else:
                # error matrix E
                E_idx = np.delete(range(D.shape[1]), j, 0)
                E = Y - np.dot(D[:,E_idx], X[E_idx,:])
                U,S,VT = np.linalg.svd(E[:,index_set])
                # update jth column of D
                D[:,j] = U[:,0]
                # update sparse elements in jth row of X    
                X[j,:] = np.array([
                    S[0]*VT[ 0,np.argwhere(index_set==n)[0][0] ]
                    if n in index_set else 0
                    for n in range(X.shape[1])])
        # stopping condition: check error        
        err = np.linalg.norm(Y-D.dot(X),'fro')
        error_norms.append(err)
        if err < etol:
            break

        print('[INFO] KSVD finished. Total iter={}, time={}'.format(iteration, time.time()-end))

    return D, X, np.array(error_norms)
