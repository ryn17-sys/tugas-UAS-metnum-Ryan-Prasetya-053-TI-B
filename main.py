from penyelesaian import (
    bagian1_tridiagonal,
    bagian2_cholesky,
    bagian3_iterasi,
    bagian4_analisis,
    bagian5_aplikasi,
    bagian6_program_user
)
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("="*60)
    print("   MULAI EKSEKUSI SELURUH JAWABAN METNUM (BAB 11)   ")
    print("="*60)
    
    # Menjalankan modul berurutan
    bagian1_tridiagonal.jalankan()
    bagian2_cholesky.jalankan()
    bagian3_iterasi.jalankan()
    bagian4_analisis.jalankan()
    bagian5_aplikasi.jalankan()
    bagian6_program_user.jalankan()
    
    print("\n" + "="*60)
    print(" Proses Selesai. Menampilkan semua grafik jendela...")
    print("="*60)
    plt.show()  # Menahan grafik agar tidak langsung menutup