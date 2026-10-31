# Pembagian Kontribusi Proyek VAE MNIST

Pembagian ini merupakan rencana tanggung jawab untuk melanjutkan proyek. Sasaran beban kerja adalah sekitar seperempat per anggota, dengan mempertimbangkan pekerjaan yang sudah dilakukan. Realisasi kontribusi dicatat berdasarkan pekerjaan nyata, commit, dan hasil yang diserahkan.

## Arah Proyek

VAE merupakan metode yang diberikan dosen. Proyek dilanjutkan dengan implementasi VAE berbasis MLP pada MNIST yang sudah tersedia di `main.ipynb`.

Fokus proyek adalah eksperimen yang dapat dijelaskan dan direproduksi: pipeline data, implementasi model, perbandingan baseline dengan hasil tuning, evaluasi pada test set, serta analisis hasil rekonstruksi dan generasi. Kesamaan metode dan dataset dengan Assignment 5 menjadi landasan materi yang diperdalam melalui eksperimen kelompok.

## Penanggung Jawab

Nama anggota kedua menggunakan akun Git yang terlihat pada riwayat repository. Nama anggota keempat dapat dilengkapi oleh kelompok.

| Anggota    | Fokus | Tanggung jawab | Hasil yang diserahkan |
|------------| --- | --- | --- |
| 1 — Darel  | Arsitektur model dan integrasi | Menjelaskan dan memeriksa encoder, reparameterization, decoder, serta loss BCE + KL; mengintegrasikan modul; membuat demonstrasi generasi acak dan interpolasi latent vector; memeriksa alur notebook dari awal sampai akhir. | Implementasi model dan loss yang terverifikasi, demonstrasi generasi/interpolasi, serta catatan pemeriksaan integrasi. |
| 2 — Debby  | Pipeline data, grafik eksperimen, dan evaluasi akhir | Memelihara preprocessing serta pembagian data; mengerjakan dua tugas berturut-turut: (1) grafik dan analisis training/tuning dari riwayat tersimpan, (2) evaluasi test dan perbandingan rekonstruksi baseline dengan GS_036. | Grafik loss/BCE/KL, analisis hyperparameter, tabel metrik test, dan visualisasi rekonstruksi beserta pembahasannya. |
| 3 — Dianne | Training, tuning, dan verifikasi checkpoint | Memelihara training/evaluation loop, konfigurasi eksperimen, dan artefak 54 kombinasi; menyerahkan checkpoint yang sudah diverifikasi beserta konfigurasi dan epoch bobotnya; menjelaskan strategi training serta membantu pemeriksaan teknis akhir. | Riwayat eksperimen, konfigurasi, checkpoint terpilih, dan bukti verifikasi pemuatan model. |
| 4 — Yasser     | Pencatatan interaksi dan penyusunan laporan | Mengompilasi riwayat chat LLM pada tahap akhir sesuai rencana kelompok; menyusun kronologi prompt, debugging, dan optimasi dari percakapan asli; menggabungkan grafik dan analisis yang diserahkan anggota teknis; menyusun laporan, materi presentasi, dan rekap kontribusi. | Log interaksi beserta tautan/ekspor asli, dokumentasi penggunaan, laporan, slide presentasi, dan rekap kontribusi. |

## Batas Tanggung Jawab Bersama

- Setiap anggota menguasai alur VAE secara keseluruhan dan menjelaskan alasan keputusan di bagiannya saat presentasi.
- Setiap anggota menyimpan percakapan LLM dan pesan error asli selama bekerja agar anggota dokumentasi dapat mengompilasinya setelah pengerjaan selesai.
- Penanggung jawab teknis menyerahkan kode, konfigurasi, metrik, gambar, dan penjelasan hasil kepada anggota dokumentasi. Penulisan laporan tidak memindahkan kewajiban analisis teknis dari pemilik modul.
- Debby membuat grafik serta analisis training/tuning dan menyerahkannya kepada anggota dokumentasi. Dianne menjadi rujukan untuk konfigurasi, prosedur training, dan metadata checkpoint.
- Pemilihan hyperparameter tetap memakai validation set. Test set digunakan setelah konfigurasi ditetapkan, dengan konfigurasi dan cara pengukuran yang dicatat.
- Pembaruan kontribusi mengikuti pekerjaan yang benar-benar dilakukan. Pekerjaan yang sudah selesai tetap mendapat pengakuan sesuai riwayatnya.

## Urutan Penyelesaian

Status awal: preprocessing, model, baseline 10 epoch, grid search 54 kombinasi, dan penyimpanan hasil sudah tersedia. Konsistensi ringkasan dengan riwayat eksperimen serta pemuatan kedua checkpoint sudah diperiksa. Branch aktif telah menambahkan penggunaan ulang checkpoint untuk melewati training/tuning; output tersimpan menunjukkan jalur tersebut digunakan. Pekerjaan berikutnya mengikuti urutan di bawah. Training/grid search diulang jika ditemukan masalah yang memengaruhi keabsahan hasil.

| Urutan | Penanggung jawab | Pekerjaan | Ketergantungan | Bukti selesai |
| --- | --- | --- | --- | --- |
| 1 | Debby — tugas pertama | Buat grafik train/validation loss baseline dan GS_036, grafik komponen BCE/KL, serta ringkasan 54 kombinasi tuning. Analisis konvergensi dan pengaruh hyperparameter dengan perbandingan terkontrol; jelaskan pemilihan GS_036 berdasarkan validation loss. | Riwayat dan konfigurasi eksperimen yang sudah tersedia. | Grafik berlabel jelas, tabel perbandingan, dan analisis training/tuning. |
| 2 | Debby — tugas kedua | Lanjutkan langsung ke evaluasi baseline dan GS_036 pada 10.000 sampel test resmi. Gunakan preprocessing, definisi loss, dan prosedur seed yang terdokumentasi. Laporkan rata-rata BCE, KL, dan total loss per gambar. Bandingkan gambar asli dengan rekonstruksi kedua model pada sampel yang sama dan jelaskan hasilnya. | Checkpoint terverifikasi yang sudah tersedia; dilaksanakan setelah tugas pertama Debby. | Tabel metrik test, konfigurasi evaluasi, gambar rekonstruksi, dan pembahasan. |
| 3 | Darel | Buat generasi 16 gambar dari sampel prior normal standar untuk baseline serta GS_036. Baca dimensi laten dari model, gunakan seed tetap, simpan gambar, dan tulis pengamatan kualitas serta keragamannya. | Checkpoint terverifikasi; dikerjakan setelah dua tugas Debby dalam urutan kerja kelompok. | Dua grid generasi 4 x 4 beserta konfigurasi dan analisis. |
| 4 | Darel | Buat interpolasi 10 titik antara mu dari dua gambar validation yang sama untuk kedua model. Sertakan kedua ujung, gunakan dimensi laten masing-masing model, simpan hasil, dan bahas kelancaran transisinya. | Model dan fungsi visualisasi dari langkah 3. | Urutan interpolasi baseline dan GS_036, referensi gambar asli, serta analisis. |
| 5 | Yasser | Gabungkan grafik dan analisis Debby serta Darel menjadi laporan dan slide; rapikan dokumentasi penggunaan; kompilasi riwayat chat asli, debugging, serta rekap kontribusi pada tahap akhir. | Hasil dan pembahasan langkah 1–4. | Dokumen final, petunjuk menjalankan proyek, lampiran chat, serta pembagian kontribusi aktual. |
| 6 | Darel sebagai integrator; seluruh anggota meninjau | Periksa kelengkapan dependensi termasuk petunjuk instalasi PyTorch CPU/CUDA, jalur eksekusi notebook dari kernel baru, penggunaan checkpoint, kecocokan hasil dengan laporan, dan kesiapan presentasi. Dianne memeriksa konsistensi konfigurasi serta metadata eksperimen. | Langkah 1–5 selesai. | Catatan pemeriksaan akhir, output notebook yang relevan tersimpan, dan setiap anggota dapat menjelaskan bagiannya. |

Urutan ini dipilih agar Debby menyelesaikan grafik/analisis dan evaluasi/rekonstruksi sebagai satu blok kerja, kemudian Darel menyelesaikan generasi dan interpolasi. Secara teknis, evaluasi test dan generasi sama-sama menggunakan checkpoint terverifikasi yang sudah tersedia. Penilaian pada test set tetap menjadi evaluasi akhir; konfigurasi model tidak dipilih ulang berdasarkan hasil test.

Perbandingan baseline dengan GS_036 mengevaluasi perubahan beberapa hyperparameter sekaligus, termasuk dimensi laten. Analisis pengaruh satu hyperparameter menggunakan hasil grid search dengan hyperparameter lain dibuat sama.

Darel mengoordinasikan penggabungan perubahan pada notebook bersama agar pekerjaan antaranggota tidak saling menimpa. Pemilik modul tetap menyerahkan hasil dan penjelasan teknisnya masing-masing.

Langkah pertama yang langsung dimulai adalah tugas pertama Debby: grafik dan analisis training/tuning. Setelah itu Debby langsung menyelesaikan evaluasi test serta rekonstruksi sebelum menyerahkan pekerjaan kepada Darel.

## Ukuran Selesai

- Notebook berjalan berurutan dan checkpoint dapat dimuat untuk evaluasi/generasi.
- Baseline dan model hasil tuning dibandingkan dengan prosedur evaluasi yang dijelaskan.
- Grafik loss, contoh rekonstruksi, generasi acak, dan interpolasi tersedia beserta analisisnya.
- README menjelaskan dependensi, dataset, cara menjalankan, dan lokasi hasil.
- Riwayat chat LLM, penanganan error yang benar-benar terjadi, dan kontribusi empat anggota terdokumentasi sebelum pengumpulan.
