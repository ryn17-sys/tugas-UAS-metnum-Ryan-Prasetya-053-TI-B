import numpy as np
from solvers.tridiagonal import thomas_algorithm

def jalankan():
    print("\n" + "="*40)
    print(" BAGIAN 1: ALGORITMA THOMAS & TRIDIAGONAL")
    print("="*40)
    
    # Soal 11.1
    print("[Soal 11.1]")
    f_1 = [0.8, 0.8, 0.8]
    e_1 = [0.0, -0.4, -0.4]
    g_1 = [-0.4, -0.4, 0.0]
    r_1 = [41.0, 25.0, 105.0]
    x_1 = thomas_algorithm(e_1, f_1, g_1, r_1)
    print(f" Solusi x: {x_1}")
    
    # Soal 11.2
    print("\n[Soal 11.2]")
    A_inv = np.zeros((3, 3))
    for i in range(3):
        e_i = np.zeros(3)
        e_i[i] = 1.0
        A_inv[:, i] = thomas_algorithm([0, -0.4, -0.4], [0.8, 0.8, 0.8], [-0.4, -0.4, 0], e_i)
    print(" Invers Matriks A via Thomas:\n", A_inv)
    
    # Soal 11.3
    print("\n[Soal 11.3]")
    f_3 = [2.01475, 2.01475, 2.01475, 2.01475]
    e_3 = [0.0, -0.020875, -0.020875, -0.020875]
    g_3 = [-0.020875, -0.020875, -0.020875, 0.0]
    r_3 = [4.175, 0.0, 0.0, 2.0875]
    x_3 = thomas_algorithm(e_3, f_3, g_3, r_3)
    print(f" Distribusi Suhu T: {x_3}")