import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load('xgb_house_model.pkl')

model = load_model()

st.title("🏠 Prediksi Harga Rumah Jabodetabek")
st.markdown("Input 7 spesifikasi utama untuk estimasi harga (Rp). Model: XGBoost (R² 0.91 log-scale, MAPE ~21%).")

# Reverse mapping: Text input → encoded numeric (match training lo)
condition_map = {'butuh renovasi': 0, 'sudah renovasi': 1, 'baru': 2, 'bagus': 3, 'bagus sekali': 4}  # property_condition
certificate_map = {'lainnya (ppjb,girik,adat,dll)': 0, 'hgb - hak guna bangunan': 1, 'shm - sertifikat hak milik': 2}  # certificate
furnishing_map = {'unfurnished': 0, 'semi furnished': 1, 'furnished': 2}  # furnishing
city_map = {'bekasi': 0, 'jakarta': 1, 'depok': 2, 'tangerang': 3, 'bogor': 4}  # Adjust daftar city lo dari data

# Sidebar input (7 utama: city, land_size, building_size, certificate, property_condition, bedrooms, bathrooms)
st.sidebar.header("Input Spesifikasi Rumah")
city_text = st.sidebar.selectbox("City", ['Bekasi', 'Jakarta', 'Depok', 'Tangerang', 'Bogor'])  # Text, auto encoded
land_size = st.sidebar.slider("Luas Tanah (m²)", 30, 1000, 100)
building_size = st.sidebar.slider("Luas Bangunan (m²)", 50, 500, 120)
certificate_text = st.sidebar.selectbox("Sertifikat", ['Lainnya (PPJB,Girik,Adat,dll)', 'HGB - Hak Guna Bangunan', 'SHM - Sertifikat Hak Milik'], index=2)
property_condition_text = st.sidebar.selectbox("Kondisi Property", ['Butuh Renovasi', 'Sudah Renovasi', 'Baru', 'Bagus', 'Bagus Sekali'], index=3)
bedrooms = st.sidebar.slider("Bedrooms", 2, 8, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 6, 2)

# Convert text to encoded
en_city = city_map.get(city_text.lower(), 0)  # Default 0
en_certificate = certificate_map.get(certificate_text.lower(), 2)
en_property_condition = condition_map.get(property_condition_text.lower(), 3)
en_furnishing = furnishing_map.get('unfurnished', 0)  # Default

carports = 1  
electricity = 2200
maid_bedrooms = 0
maid_bathrooms = 0
garages = 1

input_data = np.array([[en_city, bedrooms, bathrooms, land_size, building_size, carports, en_certificate, electricity,
                        maid_bedrooms, maid_bathrooms, en_property_condition, garages, en_furnishing]])

if st.button("🔮 Prediksi Harga"):
    try:
        # Prediksi di log-scale
        y_pred_log = model.predict(input_data)[0]
        # Inverse ke Rp asli
        y_pred_rp = np.expm1(y_pred_log)
        
        st.success(f"**Prediksi Harga: Rp {y_pred_rp:,.0f}**")
        st.info(f"Estimasi Rentang: Rp {y_pred_rp * 0.8:,.0f} - Rp {y_pred_rp * 1.2:,.0f} (error ~20% berdasarkan MAPE model)")
        
        # FIXED: Plot importance (pake get_score method XGB, aman)
        st.subheader("Faktor Pengaruh Harga:")
        try:
            booster = model.get_booster()
            importance_dict = booster.get_score(importance_type='weight')
            importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False).head(7)
            st.bar_chart(importance_df.set_index('Feature'))
        except:
            st.warning("Importance plot gak tersedia—model versi issue.")
        
    except ValueError as e:
        st.error(f"Error prediksi: {e}. Pastiin input match model features.")
        st.info("Debug: Cek X.columns di Colab & input_data array.")

st.markdown("---\nModel berdasarkan data Rumah123 Jabodetabek. Update 2025.")