import numpy as np
from solvers.tridiagonal import thomas_algorithm
from solvers.cholesky import solve_cholesky
from solvers.iterative import gauss_seidel

def jalankan():
    print("\n" + "="*40)
    print(" BAGIAN 6: IMPLEMENTASI PROGRAM (SOAL 11.24-11.26)")
    print("="*40)
    
    # 11.24: Thomas Algorithm untuk Tridiagonal
    print("\n[Soal 11.24: Thomas Algorithm]")
    f = [0.8, 0.8, 0.8]
    e = [0.0, -0.4, -0.4]
    g = [-0.4, -0.4, 0.0]
    r = [41.0, 25.0, 105.0]
    hasil_11_24 = thomas_algorithm(e, f, g, r)
    print(f" Hasil Sistem: {hasil_11_24}")
    
    # 11.25: Cholesky Decomposition
    print("\n[Soal 11.25: Cholesky Decomposition]")
    A_11_25 = np.array([[6, 15, 55], [15, 55, 225], [55, 225, 979]], dtype=float)
    b_11_25 = [152.6, 585.6, 2488.8]
    _, hasil_11_25 = solve_cholesky(A_11_25, b_11_25)
    print(f" Hasil Sistem: {hasil_11_25}")
    
    # 11.26: Gauss-Seidel
    print("\n[Soal 11.26: Gauss-Seidel]")
    A_11_26 = np.array([[0.8, -0.4, 0], [-0.4, 0.8, -0.4], [0, -0.4, 0.8]])
    b_11_26 = [41, 25, 105]
    hasil_11_26, _ = gauss_seidel(A_11_26, b_11_26, es=5.0)
    print(f" Hasil Sistem: {hasil_11_26}")