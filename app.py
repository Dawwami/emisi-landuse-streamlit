import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Emisi CO₂ Perubahan Lahan",
    page_icon="🌍",
    layout="centered" # Atau "wide"
)

# --- Path Model dan Scaler ---
# PASTIKAN nama file ini sesuai dengan yang Anda simpan dari Colab
# PASTIKAN file-file ini ada di direktori yang sama dengan app.py
MODEL_PATH = 'gru_co2_model.h5' # Sesuaikan nama file model Anda
SCALER_PATH = 'co2_scaler.pkl' # Sesuaikan nama file scaler Anda
SEQUENCE_LENGTH = 10 # Sesuaikan dengan panjang sekuens (timesteps) yang digunakan saat pelatihan
# Jika model Anda dilatih dengan banyak fitur, sesuaikan daftar ini.
# Untuk contoh ini, kita asumsikan model memprediksi satu fitur (CO2 Emissions)
FEATURE_NAMES = ['Emisi CO2'] # Sesuaikan dengan nama fitur input Anda (misal: ['Deforestation', 'Agricultural Area', 'CO2 Emission'])

# --- Fungsi untuk Memuat Model dan Scaler (dengan cache agar efisien) ---
@st.cache_resource # Gunakan st.cache_resource untuk objek besar seperti model ML
def load_ml_assets(model_path, scaler_path):
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: File model atau scaler tidak ditemukan.")
        st.error(f"Pastikan '{os.path.basename(model_path)}' dan '{os.path.basename(scaler_path)}' ada di direktori yang sama dengan 'app.py'.")
        st.stop() # Hentikan eksekusi jika file tidak ditemukan
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau scaler: {e}")
        st.stop()

model, scaler = load_ml_assets(MODEL_PATH, SCALER_PATH)

# --- Judul dan Deskripsi Aplikasi ---
st.title("🌍 Prediksi Emisi CO₂ dari Perubahan Penggunaan Lahan")
st.markdown("Aplikasi ini menggunakan model **GRU (Gated Recurrent Unit)** untuk memprediksi emisi CO₂ berdasarkan data historis perubahan penggunaan lahan.")
st.markdown("Masukkan urutan data historis emisi CO₂ atau fitur terkait untuk mendapatkan prediksi emisi berikutnya.")

st.subheader("Bagaimana cara menggunakan?")
st.info(
    f"Masukkan {SEQUENCE_LENGTH} nilai historis emisi CO₂ (atau fitur relevan) Anda, dipisahkan dengan koma. "
    "Contoh: `100.5, 102.1, 101.8, 103.5, 105.0, 104.2, 106.7, 108.3, 107.9, 109.1`."
    "\n\nUnit dan skala input harus sesuai dengan data yang digunakan saat pelatihan model."
)

# --- Input Pengguna ---
st.header("Input Data Historis")

# Jika model Anda memiliki banyak fitur, Anda bisa membuat input terpisah untuk setiap fitur
# atau meminta pengguna mengunggah CSV. Untuk kesederhanaan, kita asumsikan 1 fitur input.
user_input_str = st.text_area(f"Masukkan {SEQUENCE_LENGTH} nilai historis {FEATURE_NAMES[0]} (dipisahkan koma):",
                               "2.5, 2.7, 2.6, 2.8, 2.9, 3.1, 3.0, 3.2, 3.3, 3.5") # Ganti contoh dengan data CO2 yang relevan

# --- Tombol Prediksi ---
if st.button("Prediksi Emisi CO₂ Berikutnya"):
    try:
        # Proses input pengguna
        input_values_list = [float(x.strip()) for x in user_input_str.split(',') if x.strip()]

        if len(input_values_list) != SEQUENCE_LENGTH:
            st.warning(f"Jumlah nilai yang dimasukkan tidak sesuai. Harap masukkan tepat {SEQUENCE_LENGTH} nilai. Anda memasukkan {len(input_values_list)} nilai.")
        else:
            # Ubah ke array NumPy dan reshape untuk scaler
            # Scaler biasanya mengharapkan input 2D (jumlah_sampel, jumlah_fitur)
            # Jika hanya 1 fitur, maka shape-nya (SEQUENCE_LENGTH, 1)
            input_array = np.array(input_values_list).reshape(-1, 1) # Jika 1 fitur

            # Jika Anda memiliki banyak fitur, Anda perlu memastikan shape-nya (SEQUENCE_LENGTH, NUM_FEATURES)
            # Contoh: input_array = np.array(input_values_list).reshape(SEQUENCE_LENGTH, len(FEATURE_NAMES))

            # Normalisasi input menggunakan scaler yang sudah dilatih
            scaled_input = scaler.transform(input_array)

            # Reshape untuk model GRU: (batch_size, sequence_length, num_features)
            # Untuk prediksi satu sampel, batch_size = 1
            scaled_input_for_model = scaled_input.reshape(1, SEQUENCE_LENGTH, len(FEATURE_NAMES))

            # Prediksi menggunakan model GRU
            with st.spinner('Melakukan prediksi...'):
                prediction_scaled = model.predict(scaled_input_for_model)

            # Invers normalisasi untuk mendapatkan nilai emisi CO2 asli
            # prediction_scaled akan berbentuk (1, num_features_output_dari_model)
            # Pastikan scaler bisa melakukan inverse_transform pada shape ini
            predicted_original_value = scaler.inverse_transform(prediction_scaled)[0][0]

            st.success(f"**Prediksi Emisi CO₂ untuk periode berikutnya adalah: `{predicted_original_value:.3f}`**")

            st.markdown("---")
            st.subheader("Visualisasi Input dan Output")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Urutan Input (Nilai Asli):")
                st.line_chart(pd.DataFrame(input_values_list, columns=[FEATURE_NAMES[0]]))
            with col2:
                st.write("Urutan Input (Nilai Ternormalisasi):")
                st.line_chart(pd.DataFrame(scaled_input.flatten(), columns=['Scaled Value']))

            # Menampilkan prediksi di grafik input (opsional, untuk konteks)
            # combined_data = input_values_list + [predicted_original_value]
            # st.line_chart(pd.DataFrame(combined_data, columns=['Input & Prediction']))

    except ValueError:
        st.error("Input tidak valid. Harap masukkan nilai numerik yang dipisahkan koma.")
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
        st.info("Pastikan format input sesuai dan model/scaler sudah dimuat dengan benar.")
        st.info("Jika Anda melihat pesan ini saat pertama kali menjalankan, pastikan file model dan scaler ada dan dapat diakses.")

st.markdown("---")
st.markdown("Aplikasi prediksi ini dibuat untuk tujuan demonstrasi dan pendidikan. Akurasi prediksi sangat bergantung pada kualitas model, data pelatihan, dan relevansi fitur input.")
st.markdown("Dibuat dengan ❤️ dan Streamlit.")