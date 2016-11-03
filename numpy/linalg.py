from numpy import array
from numpy import diag
from numpy.linalg import inv
from numpy.linalg import svd

def lstsq(a, b):

    # Make a numpy array
    a = array(a)
    b = array(b)

    # Solve Ax = b
    # x = (aT * a)^-1 * aT * b
    x = inv((a.T).dot(a)).dot(a.T).dot(b)

    # Compute sum of squared differences between Ax and b ||Ax-b||^2
    residual = 0
    for i in range(a.shape[0]):
        rowSum = 0
        for j in range(a[i].shape[0]):
            rowSum += a[i][j] * x[j]
        residual += (rowSum - b[i]) ** 2

    # Compute SVD
    u, s, v = svd(a)

    #rank = 0
    #for i in range(len(s)):
    #    if s[i] > 0:
    #        rank += 1

    rank = sum(s > 0)

    return x, residual, rank, s
