import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(
    page_title="Estimasi Harga Rumah Jabodetabek",
    page_icon="🏠",
    layout="wide"
)

# Load model
@st.cache_resource
def muat_model():
    return joblib.load('xgb_house_model.pkl')

model_rumah = muat_model()

# Header
st.title("🏠 Dashboard Estimasi Harga Rumah Jabodetabek")
st.markdown("""
Dashboard ini menggunakan model **XGBoost** yang dilatih dengan data real dari Rumah123 untuk membantu Anda 
memperkirakan harga pasar properti secara objektif.
""")

# Setup Tabs
tab_prediksi, tab_insight = st.tabs(["🔮 Prediksi Harga", "📊 Insight & Analisis"])

with tab_prediksi:
    st.subheader("Detail Properti")
    
    # Grid layout untuk input
    baris1_kol1, baris1_kol2, baris1_kol3 = st.columns(3)
    
    with baris1_kol1:
        nama_kota = st.selectbox("Kota/Kabupaten", ['Jakarta', 'Bekasi', 'Depok', 'Tangerang', 'Bogor'])
        luas_tanah = st.number_input("Luas Tanah (m²)", min_value=10, max_value=2000, value=100)
        luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=10, max_value=1500, value=120)

    with baris1_kol2:
        kamar_tidur = st.slider("Kamar Tidur", 1, 10, 3)
        kamar_mandi = st.slider("Kamar Mandi", 1, 10, 2)
        jenis_sertifikat = st.selectbox("Jenis Sertifikat", 
                                       ['SHM - Sertifikat Hak Milik', 'HGB - Hak Guna Bangunan', 'Lainnya (PPJB, Girik, dll)'])

    with baris1_kol3:
        kondisi_properti = st.selectbox("Kondisi Properti", 
                                     ['Bagus sekali', 'Bagus', 'Baru', 'Sudah Renovasi', 'Butuh Renovasi'])
        jumlah_carport = st.number_input("Carport/Garasi", 0, 10, 1)
        daya_listrik = st.select_slider("Daya Listrik (VA)", [900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 11000])

    # Pemetaan internal (sesuai training)
    pemetaan_kota = {'bekasi': 0, 'jakarta': 1, 'depok': 2, 'tangerang': 3, 'bogor': 4}
    pemetaan_sertifikat = {'lainnya (ppjb, girik, dll)': 0, 'hgb - hak guna bangunan': 1, 'shm - sertifikat hak milik': 2}
    pemetaan_kondisi = {'butuh renovasi': 0, 'sudah renovasi': 1, 'baru': 2, 'bagus': 3, 'bagus sekali': 4}

    # Encode data
    kota_terencode = pemetaan_kota.get(nama_kota.lower(), 1)
    sertifikat_terencode = pemetaan_sertifikat.get(jenis_sertifikat.lower(), 2)
    kondisi_terencode = pemetaan_kondisi.get(kondisi_properti.lower(), 3)
    
    # Susun data input (match 13 fitur dari model)
    data_input = np.array([[kota_terencode, kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, jumlah_carport, 
                            sertifikat_terencode, daya_listrik, 0, 0, kondisi_terencode, 0, 0]])

    if st.button("🔮 Hitung Estimasi Harga", use_container_width=True):
        try:
            # Prediksi dalam log-scale -> Konversi ke Rupiah
            hasil_prediksi_log = model_rumah.predict(data_input)[0]
            harga_estimasi_rupiah = np.expm1(hasil_prediksi_log)
            
            # Tampilkan Hasil
            st.divider()
            hasil_kol1, hasil_kol2 = st.columns([1, 1])
            with hasil_kol1:
                st.metric("Estimasi Harga Pasar", f"Rp {harga_estimasi_rupiah:,.0f}")
            with hasil_kol2:
                st.info(f"**Rentang Wajar:** Rp {(harga_estimasi_rupiah*0.8):,.0f} - Rp {(harga_estimasi_rupiah*1.2):,.0f}")
            
            st.caption("*Disclaimer: Harga merupakan estimasi berdasarkan analisa model data science dan mungkin berbeda dengan kondisi real di lapangan.")

        except Exception as kendala:
            st.error(f"Terjadi kesalahan saat menghitung prediksi: {kendala}")

with tab_insight:
    st.subheader("Faktor Penentu Harga")
    st.markdown("Berikut adalah fitur-fitur yang paling mempengaruhi prediksi harga menurut model XGBoost:")
    
    try:
        # Ambil skor kepentingan fitur
        booster_model = model_rumah.get_booster()
        skor_kepentingan = booster_model.get_score(importance_type='weight')
        
        # Mapping nama fitur ke bahasa manusia
        nama_fitur_human = {
            'f0': 'Kota', 'f1': 'Kamar Tidur', 'f2': 'Kamar Mandi', 
            'f3': 'Luas Tanah', 'f4': 'Luas Bangunan', 'f5': 'Carport',
            'f6': 'Sertifikat', 'f7': 'Listrik', 'f10': 'Kondisi Properti'
        }
        
        list_kepentingan = []
        for id_fitur, skor in skor_kepentingan.items():
            nama_rapi = nama_fitur_human.get(id_fitur, id_fitur)
            list_kepentingan.append({'Fitur': nama_rapi, 'Skor': skor})
        
        if list_kepentingan:
            df_kepentingan = pd.DataFrame(list_kepentingan).sort_values(by='Skor', ascending=True)
            
            # Visualisasi
            fig_insight, ax_insight = plt.subplots(figsize=(10, 6))
            ax_insight.barh(df_kepentingan['Fitur'], df_kepentingan['Skor'], color='#0077b6')
            ax_insight.set_xlabel('Tingkat Pengaruh terhadap Harga (Weight)')
            st.pyplot(fig_insight)
            
            st.success("✅ **Insight:** Fitur dengan grafik terpanjang adalah faktor penentu harga paling dominan di Jabodetabek.")
        else:
            st.warning("Data statistik fitur tidak ditemukan.")

    except Exception as kendala_insight:
        st.warning(f"Visualisasi tidak dapat ditampilkan. (Detail: {kendala_insight})")

# Footer
st.divider()
st.markdown("Developed by Varel | Sumber Data: Rumah123 | Versi Model: XGB v1.2")