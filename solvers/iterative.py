import numpy as np

def gauss_seidel(A, b, lmbda=1.0, es=5.0, max_it=200):
    """
    Iterasi Gauss-Seidel dengan dukungan Relaksasi (SOR/Underrelaxation)
    """
    n = len(b)
    x = np.zeros(n)
    for it in range(max_it):
        x_old = np.copy(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new = (b[i] - s) / A[i][i]
            x[i] = lmbda * x_new + (1.0 - lmbda) * x_old[i]
        
        err = np.zeros(n)
        for i in range(n):
            if x[i] != 0:
                err[i] = abs((x[i] - x_old[i]) / x[i]) * 100
        if max(err) < es and it > 0:
            return x, it + 1
    return x, max_it

def jacobi_iteration(A, b, es=5.0, max_it=200):
    """
    Iterasi Jacobi Standar
    """
    n = len(b)
    x = np.zeros(n)
    for it in range(max_it):
        x_old = np.copy(x)
        for i in range(n):
            s = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]
            
        err = np.zeros(n)
        for i in range(n):
            if x[i] != 0:
                err[i] = abs((x[i] - x_old[i]) / x[i]) * 100
        if max(err) < es and it > 0:
            return x, it + 1
    return x, max_it