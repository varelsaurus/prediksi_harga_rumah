# 🏠 Prediksi Harga Rumah Jabodetabek

Proyek ini bertujuan untuk mengestimasi harga rumah secara akurat di wilayah Jabodetabek menggunakan algoritma **XGBoost**. Berawal dari rasa penasaran tentang fluktuasi harga properti yang seringkali sulit ditebak, saya membangun model ini menggunakan data real dari listing properti untuk memberikan estimasi yang lebih transparan bagi calon pembeli maupun investor.

**Live Demo**: [Cek Estimasi Harga di Sini!](https://prediksihargarumahdijabodetabek.streamlit.app/)

---

## 🧐 Mengapa Proyek Ini?
Membeli rumah adalah salah satu keputusan finansial terbesar dalam hidup seseorang. Dengan banyaknya faktor yang mempengaruhi harga—mulai dari luas tanah hingga kondisi bangunan—memiliki tolok ukur harga pasar menjadi sangat krusial.

### Apa yang Bisa Dilakukan Dashboard Ini?
- **Estimasi Instan**: Masukkan detail rumah (lokasi, luas, kamar, dll) dan dapatkan prediksi harga dalam hitungan detik.
- **Transparansi Model**: Lihat faktor apa saja yang paling berkontribusi terhadap harga rumah tersebut menurut analisis data.
- **Rentang Harga**: Memberikan estimasi rentang harga pasar (±20%) agar user memiliki ruang negosiasi.

## 📊 Di Balik Layar: Data & Model
Model ini tidak sekadar menebak, tapi belajar dari 3.553 data listing properti nyata di Jabodetabek.

### Ringkasan Teknis:
- **Algoritma**: `XGBoost Regressor` dengan optimasi hyperparameter.
- **Preprocessing**: Menggunakan `Log Transformation` untuk menangani distribusi harga yang condong (*skewed*), memastikan model bekerja stabil baik untuk rumah subsidi maupun rumah mewah di atas Rp 10M.
- **Akurasi**:
  - **R² Score: 0.91** (Sangat kuat dalam menangkap pola harga pasar Jabodetabek).
  - **MAPE: ~21%** (Error rata-rata yang masih dalam batas wajar untuk variasi properti yang sangat beragam).

## 🛠 Instalasi Lokal
Jika ingin menjalankan dashboard ini di komputer kamu sendiri:

1. **Persiapan**:
   ```bash
   git clone https://github.com/varelsaurus/prediksi_harga_rumah.git
   cd prediksi_harga_rumah
   ```

2. **Setup Lingkungan**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Aplikasi**:
   ```bash
   streamlit run app.py
   ```

## 💡 Pelajaran yang Diambil
Selama pengerjaan proyek ini, saya menemukan bahwa **Luas Tanah** dan **Luas Bangunan** tetap menjadi faktor dominan utama di Jabodetabek, namun **Lokasi Kota** dan **Kelengkapan Sertifikat** (SHM vs Lainnya) memberikan *boost* harga yang signifikan.

---
*Dibuat oleh Varel - 2025. Data bersumber dari Rumah123.*
