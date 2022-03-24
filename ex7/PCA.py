import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# data = loadmat('data/ex7data1.mat')
# X = data['X']
X = np.array([[1,25.5,24.1,17.8,0,0],
              [2,28.5,28.5,18.0,0,0],
              [5,34.0,34.0,34.0,0,0],
              [10,44.5,35.0,35.0,0,1],
              [10,38.0,38.0,38.0,1,1],
              [10,38.0,38.0,38.0,0,1],
              [10,44.5,35.0,35.0,0,0]])

def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()
    # compute the covariance matrix
    X = np.mat(X)
    cov = (X.T * X) / X.shape[0]
    # perform SVD
    U, S, V = np.linalg.svd(cov)
    return U, S, V


def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)


def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)


U, S, V = pca(X)
Z = project_data(X, U, 1)
X_recovered = recover_data(Z, U, 1)
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
# plt.show()
