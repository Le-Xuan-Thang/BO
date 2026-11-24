from sys import displayhook

import numpy as np

# Step 1: RBF Kernel function


def rbf_kernel(x1, x2, length_scale=1.0, signal_variance=1.0):

    # Step 1: calculate squared Eculidean distance
    dist_sq = np.sum((x1 - x2) ** 2)

    # Step 2: covariance
    covariance = signal_variance * np.exp(-dist_sq / (2 * (length_scale**2)))
    return covariance


def kernel_matrix(X1, X2, length_scale=1.0, signal_variance=1.0):

    # get the dimension of the solution
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # initial the K
    K = np.zeros((n1, n2))

    # evaluate all pairs
    for i in range(n1):
        for j in range(n2):
            K[i, j] = rbf_kernel(
                X1[i, :], X2[j, :], length_scale=1.0, signal_variance=1.0
            )

    return K


if __name__ == "__main__":

    X = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])

    K = kernel_matrix(X, X)

    print("Covariance matrix:")
    print(K)

    assert K.shape == (3, 3)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)
