import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    M = np.array(A)
    n = M.shape[0]
    m = M.shape[1]

    res = np.zeros((m, n))

    for index in np.ndindex(n, m):
        i, j = index
        res[j, i] = M[i, j]
    return res
