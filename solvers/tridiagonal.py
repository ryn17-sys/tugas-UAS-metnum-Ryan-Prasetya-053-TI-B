import numpy as np

def thomas_algorithm(e, f, g, r):
    """
    e = subdiagonal bawah (e[0]=0)
    f = diagonal utama
    g = superdiagonal atas (g[-1]=0)
    r = vektor kolom konstanta kanan
    """
    n = len(f)
    e_c = np.array(e, dtype=float)
    f_c = np.array(f, dtype=float)
    g_c = np.array(g, dtype=float)
    r_c = np.array(r, dtype=float)
    
    # Forward Elimination
    for k in range(1, n):
        factor = e_c[k] / f_c[k-1]
        f_c[k] -= factor * g_c[k-1]
        r_c[k] -= factor * r_c[k-1]
        
    # Back Substitution
    x = np.zeros(n)
    x[-1] = r_c[-1] / f_c[-1]
    for k in range(n-2, -1, -1):
        x[k] = (r_c[k] - g_c[k] * x[k+1]) / f_c[k]
    return x

def solve_pentadiagonal(A, b):
    """
    Menyelesaikan sistem linier pentadiagonal (bandwidth = 5) tanpa pivoting
    """
    n = len(b)
    A_c = np.array(A, dtype=float)
    b_c = np.array(b, dtype=float)
    
    for i in range(n):
        for j in range(i + 1, min(i + 3, n)):
            factor = A_c[j, i] / A_c[i, i]
            A_c[j, i:] -= factor * A_c[i, i:]
            b_c[j] -= factor * b_c[i]
            
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = sum(A_c[i, j] * x[j] for j in range(i + 1, min(i + 3, n)))
        x[i] = (b_c[i] - s) / A_c[i, i]
    return x