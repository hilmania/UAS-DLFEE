# data_processing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def perform_exploratory_data_analysis(df):
    """
    Melakukan dan menampilkan Analisis Data Eksplorasi (EDA).
    """
    print("\n--- Memulai Analisis Data Eksplorasi (EDA) ---")

    # 1. Informasi Dasar & Statistik
    print("\n[EDA 1] Informasi Umum Dataset:")
    df.info()

    print("\n\n[EDA 2] Statistik Deskriptif Fitur Numerik:")
    print(df.describe())

    # 2. Distribusi Kelas Target
    print("\n\n[EDA 3] Menampilkan Plot Distribusi Status Kesehatan Mental...")
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df['Mental_Health_Status'], order=df['Mental_Health_Status'].value_counts().index, palette='viridis')
    plt.title('Distribusi Status Kesehatan Mental', fontsize=16)
    plt.xlabel('Jumlah Sampel', fontsize=12)
    plt.ylabel('Status', fontsize=12)
    plt.tight_layout()
    plt.show()

    # 3. Distribusi Fitur Numerik
    print("\n\n[EDA 4] Menampilkan Plot Distribusi Fitur Numerik...")
    numerical_features = df.select_dtypes(include=np.number).drop(columns=['UserID'], errors='ignore')
    numerical_features.hist(bins=20, figsize=(20, 15), layout=(-1, 4), color='skyblue')
    plt.suptitle('Distribusi Fitur Numerik', y=1.02, fontsize=20)
    plt.tight_layout()
    plt.show()

    # 4. Matriks Korelasi
    print("\n\n[EDA 5] Menampilkan Heatmap Korelasi...")
    plt.figure(figsize=(18, 15))
    corr_matrix = numerical_features.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
    plt.title('Heatmap Korelasi Antar Fitur Numerik', fontsize=16)
    plt.show()

    print("\n--- EDA Selesai ---\n")


def load_and_preprocess_data(file_path, test_size=0.2, random_state=42, perform_eda=False):
    """
    Memuat, membersihkan, dan memproses data dari file CSV.
    Sekarang termasuk opsi untuk melakukan EDA.

    Args:
        file_path (str): Path ke file dataset CSV.
        test_size (float): Proporsi dataset untuk dialokasikan ke set pengujian.
        random_state (int): Seed untuk reproduktifitas.
        perform_eda (bool): Jika True, jalankan dan tampilkan EDA.

    Returns:
        tuple: Berisi X_train, X_test, y_train, y_test, dan label_encoder.
    """
    # 1. Memuat Dataset
    df = pd.read_csv(file_path)

    # Menjalankan EDA jika diminta
    if perform_eda:
        # Kita jalankan EDA pada data sebelum di-drop kolom atau barisnya
        # untuk mendapatkan gambaran data asli.
        perform_exploratory_data_analysis(df.copy())

    # 2. Pra-pemrosesan Dasar
    df = df.drop(['UserID', 'Timestamp'], axis=1)
    df.dropna(inplace=True)

    # 3. Memisahkan Fitur (X) dan Target (y)
    X = df.drop('Mental_Health_Status', axis=1)
    y = df['Mental_Health_Status']

    # 4. Encoding Variabel Target (y)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_names = label_encoder.classes_
    print("Kelas target berhasil di-encode:")
    for i, name in enumerate(class_names):
        print(f"  {name} -> {i}")

    # 5. Penskalaan Fitur (X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Membagi Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # 7. Mengubah Bentuk Data untuk Input RNN
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print(f"\nBentuk data pelatihan (X_train): {X_train_reshaped.shape}")
    print(f"Bentuk data pengujian (X_test): {X_test_reshaped.shape}")

    return X_train_reshaped, X_test_reshaped, y_train, y_test, label_encoder
