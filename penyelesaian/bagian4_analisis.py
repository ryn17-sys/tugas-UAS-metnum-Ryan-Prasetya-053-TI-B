import numpy as np
from scipy.linalg import norm, hilbert

def jalankan():
    print("\n" + "="*40)
    print(" BAGIAN 4: BILANGAN KONDISI & MATRIKS BURUK")
    print("="*40)
    
    # Soal 11.16
    print("[Soal 11.16]")
    for n_dim, A, b in [
        (3, np.array([[1, 4, 9], [4, 9, 16], [9, 16, 25]], dtype=float), [14, 29, 50]),
        (4, np.array([[1, 4, 9, 16], [4, 9, 16, 25], [9, 16, 25, 36], [16, 25, 36, 49]], dtype=float), [30, 54, 86, 126])
    ]:
        cond = norm(A, np.inf) * norm(np.linalg.inv(A), np.inf)
        print(f" Matriks {n_dim}xD -> Condition Number (Row-Sum Norm): {cond:.2f}")
        
    # Soal 11.19
    print("\n[Soal 11.19]")
    H10 = hilbert(10)
    cond_H10 = np.linalg.cond(H10, p=2)
    print(f" Hilbert 10D Cond-Number: {cond_H10:.2e}")
    print(f" Perkiraan digit presisi yang hilang: {int(np.ceil(np.log10(cond_H10)))} digit")
    
    # Soal 11.20 
    print("\n[Soal 11.20]")
    x_vander = np.array([4, 2, 7, 10, 3, 5], dtype=float)
    n_v = len(x_vander)
    
    # Membuat matriks Vandermonde manual (V_ij = x_i^(j-1)) sesuai standar interpolasi Bab 10
    V6 = np.zeros((n_v, n_v))
    for i in range(n_v):
        for j in range(n_v):
            V6[i, j] = x_vander[i]**j
            
    cond_V6 = np.linalg.cond(V6, p=2)
    print(" Matriks Vandermonde 6D (Rujukan Prob 10.17):\n", V6)
    print(f" Vandermonde 6D Cond-Number: {cond_V6:.2e}")
    print(f" Perkiraan digit presisi yang hilang: {int(np.ceil(np.log10(cond_V6)))} digit")
    
    # Soal 11.21 & 11.22
    print("\n[Soal 11.21 & 11.22]")
    A_22 = np.array([[0, -7, 5], [4, 0, 7], [-4, 3, -7]], dtype=float)
    b_22 = [50, -30, 40]
    x_22 = np.linalg.solve(A_22, b_22)
    print(f" Solusi Unkown [x1, x2, x3]: {x_22}")
    print(" Transpose Matriks A:\n", A_22.T)
    print(" Invers Matriks A:\n", np.linalg.inv(A_22))