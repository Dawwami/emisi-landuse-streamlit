import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import joblib # type: ignore

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Emisi CO₂ Perubahan Lahan",
    page_icon="🌍",
    layout="centered"
)

# --- Path Model dan Scaler ---
MODEL_PATH = './gru_co2_model.h5'
SCALER_PATH = './co2_scaler.pkl'
SEQUENCE_LENGTH = 10
FEATURE_NAMES = ['CO2']  # GANTI agar sama dengan saat scaler.fit()

# --- Fungsi untuk Memuat Model dan Scaler ---
@st.cache_resource
def load_ml_assets(model_path, scaler_path):
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_ml_assets(MODEL_PATH, SCALER_PATH)

# --- Judul dan Deskripsi Aplikasi ---
st.title("🌍 Prediksi Emisi CO₂ dari Perubahan Penggunaan Lahan")
st.markdown("Aplikasi ini menggunakan model **GRU (Gated Recurrent Unit)** untuk memprediksi emisi CO₂ berdasarkan data historis perubahan penggunaan lahan.")


st.subheader("Bagaimana cara menggunakan?")
st.info(
    f"Aplikasi menggunakan data historis emisi CO₂ per benua. "
    f"Pilih benua untuk mendapatkan hasil emisi CO₂ yang diperkirakan selama 5 tahun ke depan."
)
# Load CSV
df_landuse = pd.read_csv("co2-land-use.csv")
df_region = pd.read_csv("country_regions.csv")

# Gabungkan berdasarkan kolom 'Code'
df_merged = df_landuse.merge(df_region, on="Code")

# Ambil data 10 tahun terakhir untuk masing-masing region
sequence_length = 10
region_list = df_merged['Region'].unique()

data_historis_co2 = {}

for region in region_list:
    # Ambil semua data dari region ini
    df_region_data = df_merged[df_merged['Region'] == region]

    # Hitung total emisi CO2 per tahun per region
    df_grouped = df_region_data.groupby('Year')['Annual CO₂ emissions from land-use change'].sum().reset_index()

    # Ambil N tahun terakhir
    last_n_years = df_grouped.sort_values('Year').tail(sequence_length)

    # Simpan ke dictionary
    data_historis_co2[region] = last_n_years['Annual CO₂ emissions from land-use change'].tolist()
# --- Data Historis Dummy per Benua ---
DATA_HISTORIS_CO2 = data_historis_co2

# --- Input User ---
st.header("Input Prediksi untuk 5 Tahun ke Depan")
benua = st.selectbox("Pilih Benua:", options=list(DATA_HISTORIS_CO2.keys()))

if st.button("Lakukan Prediksi"):
    try:
        data_input = DATA_HISTORIS_CO2[benua]

        # Ambil data 10 terakhir, ubah ke DataFrame sesuai fitur
        input_df = pd.DataFrame(data_input[-SEQUENCE_LENGTH:], columns=FEATURE_NAMES)

        # Normalisasi
        scaled_input = scaler.transform(input_df)

        # GRU input shape: (batch, time, features)
        input_model = scaled_input.reshape(1, SEQUENCE_LENGTH, 1)

        # Prediksi sekaligus 5 tahun
        prediction_scaled = model.predict(input_model)  # shape: (1, 5)

        # Buat DataFrame dengan 5 kolom agar cocok dengan scaler
        # Gunakan nama fitur yang cocok (misalnya CO2_t+1, CO2_t+2, dst)
        pred_feature_names = [f'CO2_t+{i+1}' for i in range(prediction_scaled.shape[1])]
        pred_df = pd.DataFrame(prediction_scaled, columns=pred_feature_names)

        # Invers normalisasi tiap kolom
        pred_df_inverse = pd.DataFrame()
        for i in range(pred_df.shape[1]):
            col_df = pd.DataFrame(pred_df.iloc[:, [i]].values, columns=FEATURE_NAMES)
            inversed = scaler.inverse_transform(col_df)[0][0]
            pred_df_inverse[f'Tahun 20{i+24}'] = [inversed]

        # Transpose untuk tampilan
        pred_df_inverse = pred_df_inverse.T
        pred_df_inverse.columns = ['Emisi CO2 (juta ton)']

        # Tampilkan hasil
        st.success(f"Prediksi emisi CO₂ selama {prediction_scaled.shape[1]} tahun ke depan di {benua}:")
        st.dataframe(pred_df_inverse)
        jumlah_prediksi = prediction_scaled.shape[1]
        prediksi_values = [pred_df_inverse.iloc[i, 0] for i in range(jumlah_prediksi)]
        
        # Visualisasi
        # Buat label tahun
        tahun_terakhir = max(df_grouped['Year'])
        #tahun_terakhir = 2024  # <- ganti sesuai dengan tahun terakhir data historis kamu
        tahun_historis = list(range(tahun_terakhir - SEQUENCE_LENGTH + 1, tahun_terakhir + 1))
        tahun_prediksi = list(range(tahun_terakhir + 1, tahun_terakhir + jumlah_prediksi + 1))
        tahun_total = tahun_historis + tahun_prediksi

        # Gabungkan data historis dan hasil prediksi
        full_series = DATA_HISTORIS_CO2[benua][:SEQUENCE_LENGTH] + prediksi_values

        # Buat dataframe dengan index tahun
        df_plot = pd.DataFrame({
        'Tahun': tahun_total,
        'CO₂ Emission': full_series
        })

        df_plot['Tahun'] = df_plot['Tahun'].astype(int)  # <-- PENTING!
        df_plot = df_plot.set_index('Tahun')
        # Tampilkan grafik
        st.line_chart(df_plot)

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Aplikasi prediksi ini dibuat untuk tujuan demonstrasi dan pendidikan. Akurasi prediksi sangat bergantung pada kualitas model, data pelatihan, dan relevansi fitur input.")
st.markdown("Dibuat dengan Streamlit.")
