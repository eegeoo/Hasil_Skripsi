import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

# Set seed untuk memastikan hasil yang konsisten
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
df = pd.read_csv('dataperokok.csv')

def build_lstm_model(input_shape, units=10):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=units))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def analisis_provinsi_lstm(nama_provinsi, units=10, batch_size=4, epochs=30):
    try:
        # Filter data berdasarkan nama provinsi
        data_provinsi = df[df['38 Provinsi'].str.lower() == nama_provinsi.lower()].iloc[0]

        # Persiapkan data untuk model LSTM
        jumlah_perokok = data_provinsi[['2019', '2020', '2021', '2022']].values.reshape(-1, 1)

        # Normalisasi data menggunakan Min-Max Scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        jumlah_perokok_scaled = scaler.fit_transform(jumlah_perokok)

        # Membentuk data pelatihan dengan bentuk (samples, timesteps, features)
        X_train = []
        y_train = []
        for i in range(1, len(jumlah_perokok_scaled)):
            X_train.append(jumlah_perokok_scaled[i-1:i].reshape(1, -1))  # Menggunakan data sebelumnya sebagai fitur
            y_train.append(jumlah_perokok_scaled[i])  # Menggunakan data sekarang sebagai target

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Pastikan bentuk X_train adalah (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[2]))

        # Membuat dan melatih model LSTM
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), units=units)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Prediksi untuk tahun 2023
        last_scaled = jumlah_perokok_scaled[-1].reshape(1, 1, 1)  # Gunakan data terakhir (2022) untuk memprediksi 2023
        prediksi_2023_scaled = model.predict(last_scaled)
        prediksi_2023 = scaler.inverse_transform(prediksi_2023_scaled).reshape(-1, 1)

        # Evaluasi menggunakan data aktual tahun 2023
        y_true_2023 = np.array([[data_provinsi['2023']]])

        # Hitung MAPE dan MAE
        mape_2023 = mean_absolute_percentage_error(y_true_2023, prediksi_2023)
        mae_2023 = mean_absolute_error(y_true_2023, prediksi_2023)

        # Visualisasi hasil
        tahun_plot = np.array([2019, 2020, 2021, 2022, 2023])
        jumlah_perokok_plot = np.append(jumlah_perokok, prediksi_2023, axis=0)

        plt.figure(figsize=(12, 6))
        plt.scatter(tahun_plot[:-1], jumlah_perokok, color='blue', label='Data Aktual (2019-2022)')
        plt.plot(tahun_plot, jumlah_perokok_plot, color='red', linestyle='--', label='Prediksi LSTM')

        plt.scatter(2023, y_true_2023, color='green', s=100, label='Nilai Aktual 2023')
        plt.scatter(2023, prediksi_2023, color='orange', s=100, label='Prediksi 2023')

        plt.title(f'Analisis Jumlah Perokok di {data_provinsi["38 Provinsi"]}')
        plt.xlabel('Tahun')
        plt.ylabel('Jumlah Perokok')

        plt.xticks(tahun_plot)  # Menampilkan hanya tahun 2019, 2020, 2021, 2022, 2023
        plt.legend()

        # Tambahkan informasi MAPE, MAE, dan prediksi di sudut kanan atas dalam area plot
        info_text = f'MAPE 2023: {mape_2023:.2f}%\nMAE 2023: {mae_2023:.2f}%\nPrediksi 2023: {prediksi_2023[0][0]:.2f}%'
        plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        st.pyplot(plt)

        return {
            'provinsi': data_provinsi['38 Provinsi'],
            'mape_2023': mape_2023,
            'mae_2023': mae_2023,
            'prediksi_2023': prediksi_2023[0][0],
            'y_true_2023': y_true_2023[0][0],
            'plot': plt
        }

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return None

# Streamlit UI
st.title("Analisis Jumlah Perokok Di Indonesia")
st.write("Provinsi yang tersedia:")

# Menampilkan gambar provinsi yang tersedia
img = plt.imread('Streamlit-Perokok.png')  # Ganti dengan path gambar yang sesuai
st.image(img, caption='Provinsi Indonesia', use_column_width=True)

# Pilihan nama provinsi
nama_provinsi = st.selectbox("Pilih Provinsi:", df['38 Provinsi'].unique())

# Tampilkan hasil ketika tombol ditekan
if st.button("Tampilkan Hasil"):
    hasil = analisis_provinsi_lstm(nama_provinsi)
    if hasil:
        st.write(f"MAPE 2023: {hasil['mape_2023'] * 100:.2f} %")
        st.write(f"MAE 2023: {hasil['mae_2023']:.2f}%")
        st.write(f"Prediksi 2023: {hasil['prediksi_2023']:.2f}%")
        st.write(f"Nilai Aktual 2023: {hasil['y_true_2023']:.2f}%")
