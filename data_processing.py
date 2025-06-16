# data_processing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

def perform_exploratory_data_analysis(df, results_dir):
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
    fig_dist = plt.figure(figsize=(10, 6))
    sns.countplot(y=df['mental_health_status'], order=df['mental_health_status'].value_counts().index, palette='viridis')
    plt.title('Distribusi Status Kesehatan Mental', fontsize=16)
    plt.xlabel('Jumlah Sampel', fontsize=12)
    plt.ylabel('Status', fontsize=12)
    plt.tight_layout()

    # simpan plot ke direktori hasil
    save_path = os.path.join(results_dir, 'mental_health_status_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot distribusi status kesehatan mental disimpan di {save_path}")
    plt.close(fig_dist)

    # 3. Distribusi Fitur Numerik
    print("\n\n[EDA 4] Menampilkan Plot Distribusi Fitur Numerik...")
    numerical_features = df.select_dtypes(include=np.number).drop(columns=['UserID'], errors='ignore')
    if not numerical_features.empty:
        fig_hist = numerical_features.hist(bins=20, figsize=(20, 15), layout=(-1, 4), color='skyblue')
        plt.suptitle('Distribusi Fitur Numerik', y=1.02, fontsize=20)
        plt.tight_layout()
        save_path = os.path.join(results_dir, 'numerical_features_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot distribusi fitur numerik disimpan di {save_path}")
        plt.close()

    # 4. Matriks Korelasi
        print("\n\n[EDA 5] Menampilkan Heatmap Korelasi...")
        fig_corr = plt.figure(figsize=(18, 15))
        corr_matrix = numerical_features.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
        plt.title('Heatmap Korelasi Antar Fitur Numerik', fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(results_dir, 'correlation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap korelasi disimpan di {save_path}")
        plt.close(fig_corr)

    print("\n--- EDA Selesai ---\n")


def load_and_preprocess_data(file_path, test_size=0.2, random_state=42, perform_eda=False, results_dir=None):
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
    if perform_eda and results_dir:
        # Kita jalankan EDA pada data sebelum di-drop kolom atau barisnya
        # untuk mendapatkan gambaran data asli.
        perform_exploratory_data_analysis(df.copy(), results_dir)
    elif perform_eda:
        print("\nPeringatan: `perform_eda=True` tetapi `results_dir` tidak diberikan. Plot EDA tidak akan disimpan.")

    # 2. Pra-pemrosesan Dasar
    df = df.drop(['timestamp'], axis=1)
    df.dropna(inplace=True)

    # 3. Memisahkan Fitur (X) dan Target (y)
    X = df.drop('mental_health_status', axis=1)
    y = df['mental_health_status']

    # 4. Encoding Variabel Target (y)
    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(y)

    # class_names = label_encoder.classes_
    # print("Kelas target berhasil di-encode:")
    # for i, name in enumerate(class_names):
    #     print(f"  {name} -> {i}")

    # 5. Penskalaan Fitur (X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Membagi Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 7. Mengubah Bentuk Data untuk Input RNN
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print(f"\nBentuk data pelatihan (X_train): {X_train_reshaped.shape}")
    print(f"Bentuk data pengujian (X_test): {X_test_reshaped.shape}")

    return X_train_reshaped, X_test_reshaped, y_train, y_test
