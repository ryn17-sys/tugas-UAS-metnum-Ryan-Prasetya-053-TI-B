import numpy as np
from solvers.iterative import gauss_seidel, jacobi_iteration

def jalankan():
    print("\n" + "="*40)
    print(" BAGIAN 3: ITERASI JACOBI & GAUSS-SEIDEL")
    print("="*40)
    
    # Soal 11.8
    print("[Soal 11.8]")
    A_1 = np.array([[0.8, -0.4, 0], [-0.4, 0.8, -0.4], [0, -0.4, 0.8]])
    b_1 = [41, 25, 105]
    x_8, it_8 = gauss_seidel(A_1, b_1, lmbda=1.2, es=5.0)
    print(f" SOR (lambda=1.2): {x_8} ({it_8} iterasi)")
    
    # Soal 11.9 & 11.10
    print("\n[Soal 11.9 & 11.10]")
    A_react = np.array([[15, -3, -1], [-3, 18, -6], [-4, -1, 12]], dtype=float)
    b_react = [3800, 1200, 2350]
    x_9, it_9 = gauss_seidel(A_react, b_react, es=5.0)
    x_10, it_10 = jacobi_iteration(A_react, b_react, es=5.0)
    print(f" Gauss-Seidel C: {x_9} ({it_9} iterasi)")
    print(f" Jacobi C: {x_10} ({it_10} iterasi)")
    
    # Soal 11.11
    print("\n[Soal 11.11]")
    A_11 = np.array([[10, 2, -1], [-3, -6, 2], [1, 1, 5]], dtype=float)
    b_11 = [27, -61.5, -21.5]
    x_11, it_11 = gauss_seidel(A_11, b_11, es=5.0)
    print(f" Hasil Gauss-Seidel: {x_11} ({it_11} iterasi)")
    
    # Soal 11.12 
    print("\n[Soal 11.12]")
    A_12 = np.array([[6, -1, -1], [6, 9, 1], [-3, 1, 12]], dtype=float)
    b_12 = [3, 40, 50]
    x_12_a, _ = gauss_seidel(A_12, b_12, lmbda=1.0, es=5.0)
    x_12_b, _ = gauss_seidel(A_12, b_12, lmbda=0.95, es=5.0)
    print(f" (a) Tanpa relaxation: {x_12_a}")
    print(f" (b) Berelaksasi (lambda=0.95): {x_12_b}")
    
    # Soal 11.13 - 
    print("\n[Soal 11.13]")
    # Diurutkan: baris 3, baris 1, baris 2
    A_13 = np.array([[-8, 1, -2], [2, -6, -1], [-3, -1, 7]], dtype=float)
    b_13 = [-20, -38, -34]
    x_13_a, _ = gauss_seidel(A_13, b_13, lmbda=1.0, es=5.0)
    x_13_b, _ = gauss_seidel(A_13, b_13, lmbda=1.2, es=5.0)
    print(f" (a) Tanpa relaxation: {x_13_a}")
    print(f" (b) Overrelaxation (lambda=1.2): {x_13_b}")
    
    # Soal 11.14 & 11.15
    print("\n[Soal 11.14 & 11.15 - Analisis Kualitatif]")
    print(" Sistem Set 2 & Set 3 pada Soal 11.15 tidak dominan secara diagonal dan akan divergen.")