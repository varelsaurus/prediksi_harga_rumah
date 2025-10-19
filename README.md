# 🏠 Prediksi Harga Rumah Jabodetabek

[![Streamlit](https://img.shields.io/badge/Streamlit-FF6B35?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io) [![XGBoost](https://img.shields.io/badge/XGBoost-1F77B4?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io) [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)

App web interaktif untuk estimasi harga rumah di Jabodetabek menggunakan model XGBoost. Dibangun dari data real Rumah123 (3.553 sampel), dengan akurasi tinggi: **R² 0.91 (log-scale)** dan **MAPE ~21%**. Input sederhana seperti letak daerah bangunan,luas tanah, jumlah kamar, kondisi properti—output harga Rp akurat plus rentang estimasi!

**Live Demo**: [Coba App Sekarang!](https://prediksihargarumahdijabodetabek.streamlit.app/)  
*(Atau deploy sendiri dari repo ini!)*

![App Screenshot](images/overview.png) <!-- Upload screenshot app lo di repo, ganti path ini -->

## 🚀 Fitur Utama
- **Input Sederhana**: Hanya 7 field (city, luas tanah/bangunan, sertifikat, kondisi, bedrooms, bathrooms)—sisanya auto-default.
- **Prediksi Cepat**: Model XGBoost dengan log-transform untuk handle skew harga (Rp 100jt - 16Mrd).
- **Visualisasi**: Bar chart faktor pengaruh harga (e.g., luas tanah dominan 40%+).
- **Estimasi Rentang**: Output harga + ±20% error berdasarkan MAPE model.
- **Open-Source**: Clone, run local, atau deploy ulang di Streamlit/Hugging Face.

## 📊 Hasil Analisis Notebook (Model Training)
Project ini dibangun dari notebook Jupyter/Colab (`notebook.ipynb`) yang handle data real Rumah123 Jabodetabek. Berikut ringkasan analisis & performa:

### Data Overview
- **Sumber**: Rumah123.com (3.553 listing rumah Jabodetabek).
- **Features**: 13 kolom utama (city, bedrooms, bathrooms, land_size_m2, building_size_m2, carports, certificate, electricity, maid_bedrooms, maid_bathrooms, property_condition, garages, furnishing).
- **Target**: `price_in_rp` (Rp 100jt - 16Mrd, skewed 24.74 → normal setelah log-transform).
- **Preprocessing**:
  - Handle missing: Fill mode/median/KNNImputer (e.g., property_condition → 'bagus').
  - Outliers: Clip IQR (land/building size, harga ekstrem).
  - Encoding: Ordinal untuk categorical (kondisi: 0=butuh renovasi → 4=bagus sekali; sertifikat: 0=lainnya → 2=SHM).
  - Log-transform: `np.log1p(price_in_rp)` fix skew → distribusi normal (skew ~0.78).

### Model Performance
- **XGBoost (Utama)**: R² 0.91 (log-scale), MSE 0.12, MAE 0.22.
- **Random Forest**: R² 0.90 (log-scale), MSE 0.12, MAE 0.20.
- **Evaluasi Asli (Rp)**: R² 0.57 (skewed penalti), MAPE 21% (error rata-rata 21% per rumah), MAE 1.4 Miliar Rp (realistis buat estimasi).

## 🛠 Cara Install & Run
1. **Clone Repo**:
   ```
   git clone https://github.com/varelsaurus/prediksi_harga_rumah.git
   cd house-price-app
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run Local**:
   ```
   streamlit run app.py
   ```
   - Buka [localhost:8501](http://localhost:8501) di browser.

4. **Deploy Ulang**:
   - Fork repo ini.
   - Connect ke Streamlit Cloud/Hugging Face Spaces.
   - Deploy gratis!

## 📈 Model Training (Notebook)
- **File**: `notebook.ipynb` (attach dari Colab).
- **Key Steps**:
  - Load data Rumah123 (3553 rows).
  - Preprocessing: Missing fill, log-transform y, clip outliers, ordinal encoding.
  - Train/Test Split: 80/20, random_state=42.
  - Tuning: RandomizedSearchCV untuk XGBoost/RF.
  - Evaluasi: R² log 0.91, MAPE 21% (error Rp ~1.4 Miliar rata-rata).
- **Visual**: Histogram skew sebelum/sesudah log, boxplot outliers, feature importance bar chart.

![Preprocessing Flow](images/before-after) <!-- Upload diagram dari notebook -->

## 🤝 Contributing
- Fork repo, tambah fitur (e.g., map lokasi), PR!
- Issues: Buka kalau bug atau saran (e.g., tambah RF toggle).

*(Update: October 2025. Dibangun dengan data real Rumah123. Disclaimer: Estimasi, bukan saran finansial.)*3
