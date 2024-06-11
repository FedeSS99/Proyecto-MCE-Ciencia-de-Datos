from numpy import ndarray, zeros, float64, inf
from numba import njit, prange

@njit(fastmath=True)
def local_dist(x, y):
    return abs(x - y)

@njit
def DTW(Xi:ndarray, Xj:ndarray, q:float = 2.0):
    length_Xi = Xi.size
    length_Xj = Xj.size
    R = zeros((length_Xi, length_Xj), dtype = float64)
    R[0,0] = local_dist(Xi[0], Xj[0]) ** q
    
    for i in range(1, length_Xi):
        R[i, 0] = R[i-1, 0] + local_dist(Xi[i], Xj[0]) ** q
    for j in range(1, length_Xj):
        R[0, j] = R[0, j-1] + local_dist(Xi[0], Xj[j]) ** q
    
    for i in range(1, length_Xi):
        for j in range(1, length_Xj):
            cost = local_dist(Xi[i], Xj[j]) ** q
            R[i, j] = cost + min(R[i-1, j], R[i, j-1], R[i-1, j-1])

    return R[length_Xi - 1, length_Xj - 1] ** (1.0/q)

@njit(parallel = True)
def MatrixDTW(X:ndarray, q:float = 2.0):
    length_X:int = len(X)
    matrix_dtw:ndarray = zeros((length_X, length_X), dtype = float64)

    for n in prange(length_X-1):
        for m in range(n+1, length_X):
            matrix_dtw[n, m] = DTW(X[n], X[m], q)
            matrix_dtw[m, n] = matrix_dtw[n, m]
    
    return matrix_dtw