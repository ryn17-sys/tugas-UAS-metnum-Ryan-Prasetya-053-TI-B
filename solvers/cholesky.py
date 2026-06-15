import numpy as np

def cholesky_decomposition(A):
    """
    Dekomposisi Cholesky untuk matriks simetris positif definit (A = L * L^T)
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(max(A[i][i] - s, 0.0))
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def solve_cholesky(A, b):
    L = cholesky_decomposition(A)
    n = len(b)
    
    # Ly = b (Forward Substitution)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
        
    # L^T x = y (Backward Substitution)
    x = np.zeros(n)
    LT = L.T
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(LT[i][j] * x[j] for j in range(i + 1, n))) / LT[i][i]
    return L, x