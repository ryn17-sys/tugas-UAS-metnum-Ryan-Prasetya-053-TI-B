import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from solvers.tridiagonal import solve_pentadiagonal

def jalankan():
    print("\n" + "="*40)
    print(" BAGIAN 5: APLIKASI TEKNIK & PLOT GRAFIK")
    print("="*40)
    
    # Soal 11.17
    print("[Soal 11.17 - Non-linear]")
    sys_func = lambda v: [4 - v[1] - 2*v[0]**2, 8 - v[1]**2 - 4*v[0]]
    for guess in [[-2, -2], [2, 2]]:
        sol = root(sys_func, guess)
        print(f" Tebakan {guess} -> Hasil Konvergen: {sol.x}")
        
    # Soal 11.18
    print("\n[Soal 11.18 - Alokasi Produksi]")
    A_p = np.array([[4, 3, 2], [1, 3, 1], [2, 1, 3]], dtype=float)
    b_p = [960, 510, 610]
    x_p = np.linalg.solve(A_p, b_p)
    print(f" Transistor: {x_p[0]:.0f}, Resistor: {x_p[1]:.0f}, Chip: {x_p[2]:.0f}")
    
    # Soal 11.23 - REVISI (Plot detail FLOPs berdasarkan Sec. 9.2.1 vs Fig 11.2)
    print("\n[Soal 11.23] Menampilkan grafik perbandingan FLOPs (Sec. 9.2.1 vs Fig. 11.2)...")
    n_range = np.arange(2, 21) # n dari 2 sampai 20
    
    # Rumus FLOPs total dari Sec. 9.2.1 (Eliminasi Gauss Tanpa Pivoting)
    flops_gauss = (2/3) * n_range**3 + (3/2) * n_range**2 - (7/6) * n_range
    # Rumus FLOPs total Algoritma Thomas (Fig. 11.2)
    flops_thomas = 8 * n_range - 7
    
    plt.figure(figsize=(6, 4))
    plt.plot(n_range, flops_gauss, 'r-o', linewidth=1.5, label='Gauss Elimination (Sec. 9.2.1)')
    plt.plot(n_range, flops_thomas, 'b-s', linewidth=1.5, label='Thomas Algorithm (Fig. 11.2)')
    plt.xlabel('Ukuran Matriks (n)')
    plt.ylabel('Jumlah Operasi (FLOPs)')
    plt.title('Perbandingan Kompleksitas Komputasi')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show(block=False)
    
    # Soal 11.27
    print("\n[Soal 11.27] Menampilkan grafik Finite Difference Kanal...")
    D, U, k, dx = 2.0, 1.0, 0.2, 2.0
    x_nodes = np.arange(0, 12, dx)
    n_n = len(x_nodes)
    A_c, b_c = np.zeros((n_n, n_n)), np.zeros(n_n)
    A_c[0, 0] = 1.0; b_c[0] = 80.0
    A_c[-1, -1] = 1.0; b_c[-1] = 20.0
    for i in range(1, n_n - 1):
        A_c[i, i-1] = (D / dx**2) + (U / (2*dx))
        A_c[i, i] = -((2*D / dx**2) + k)
        A_c[i, i+1] = (D / dx**2) - (U / (2*dx))
    c_sol = np.linalg.solve(A_c, b_c)
    plt.figure(figsize=(5, 3))
    plt.plot(x_nodes, c_sol, 'm-o')
    plt.title("Profil Kanal"); plt.grid(True)
    plt.show(block=False)
    
    # Soal 11.28
    print("\n[Soal 11.28 - Pentadiagonal]")
    A_28 = np.array([
        [8, -2, -1,  0,  0], [-2, 9, -4, -1,  0], [-1, -3, 7, -1, -2],
        [0, -4, -2, 12, -5], [0,  0, -7, -3, 15]
    ], dtype=float)
    b_28 = [5, 2, 1, 1, 5]
    print(" Hasil Pentadiagonal x:", solve_pentadiagonal(A_28, b_28))