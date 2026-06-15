import numpy as np
from solvers.cholesky import cholesky_decomposition, solve_cholesky

def jalankan():
    print("\n" + "="*40)
    print(" BAGIAN 2: DEKOMPOSISI CHOLESKY")
    print("="*40)
    
    # Soal 11.4
    print("[Soal 11.4]")
    A_ex2 = np.array([[6, 15, 55], [15, 55, 225], [55, 225, 979]], dtype=float)
    L_ex2 = cholesky_decomposition(A_ex2)
    print(" Matriks L Rekonstruksi:\n", L_ex2)
    print(" Validasi L*L^T == A?", np.allclose(A_ex2, np.dot(L_ex2, L_ex2.T)))
    
    # Soal 11.5
    print("\n[Soal 11.5]")
    b_5 = [152.6, 585.6, 2488.8]
    _, x_5 = solve_cholesky(A_ex2, b_5)
    print(f" Nilai koefisien [a0, a1, a2]: {x_5}")
    
    # Soal 11.6
    print("\n[Soal 11.6]")
    A_6 = np.array([[8, 20, 15], [20, 80, 50], [15, 50, 60]], dtype=float)
    b_6 = [50, 250, 100]  # Sesuai gambar halaman 2 buku
    L_6, x_6 = solve_cholesky(A_6, b_6)
    print(" Matriks L Bawah:\n", L_6)
    print(f" Hasil penyelesaian x: {x_6}")
    
    # Soal 11.7
    print("\n[Soal 11.7]")
    A_7 = np.array([[9, 0, 0], [0, 25, 0], [0, 0, 4]], dtype=float)
    L_7 = cholesky_decomposition(A_7)
    print(" Matriks L dari Diagonal:\n", L_7)