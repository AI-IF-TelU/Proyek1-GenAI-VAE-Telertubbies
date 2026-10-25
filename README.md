# VAE MNIST

Implementasi Variational Autoencoder berbasis MLP di `main.ipynb`.

## Struktur project

- `main.ipynb`: preprocessing MNIST, arsitektur/loss VAE, training baseline,
  grid search, dan pemeriksaan rekonstruksi checkpoint.
- `data/mnist.npz`: dataset MNIST.
- `results/baseline/`: checkpoint baseline, konfigurasi, dan riwayat metrik.
- `results/grid_search/`: hasil tiap kombinasi, ringkasan CSV, dan
  `best_combination.json` yang menunjuk checkpoint terbaik.
- `requirements.txt`: dependensi lingkungan notebook. PyTorch juga harus
  tersedia di kernel yang digunakan; notebook memilih CUDA jika tersedia.

## Menjalankan notebook

Buka `main.ipynb` dengan working directory di root project, lalu jalankan cell
secara berurutan dari kernel baru. Baseline dan tuning diperiksa secara terpisah:

- Jika `results/baseline/vae_baseline.pth` tersedia, pengujian training satu epoch
  dan training baseline 10 epoch dilewati. Model dimuat sesuai konfigurasi dalam
  checkpoint dan riwayat metrik dimuat jika tersedia.
- Jika checkpoint yang ditunjuk `best_combination.json` tersedia, grid search
  dilewati dan checkpoint tersebut dipakai untuk rekonstruksi. Jika metadata
  belum tersedia, notebook mencari hasil terbaik dari ringkasan CSV.
- Jika checkpoint belum tersedia, proses terkait berjalan seperti biasa.
  Grid search melanjutkan kombinasi yang belum selesai atau checkpoint-nya hilang.

Artefak baseline dan tuning yang digunakan kembali tidak ditulis ulang. Untuk
eksperimen ulang, arsipkan hasil terkait ke lokasi lain terlebih dahulu.

## Verifikasi

Dengan lingkungan Python yang memiliki dependensi notebook, jalankan
`python -m unittest discover -s tests -v`. Pengujian memakai direktori sementara,
memeriksa pemuatan checkpoint asli serta jalur checkpoint yang belum tersedia,
dan tidak menjalankan training penuh atau mengubah hasil eksperimen project.
