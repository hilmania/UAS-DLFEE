# run_comparison.py (Versi Final yang Disesuaikan)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight # <-- 1. TAMBAHKAN IMPORT INI

# Impor modul kita
import data_processing
import model_builder # Pastikan file ini adalah versi terbaru
import os

def plot_training_history(history, save_path=None, model_name=''):
    """Membuat plot akurasi dan loss selama pelatihan."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Menambahkan judul utama untuk seluruh figure
    fig.suptitle(f'Training History for {model_name.upper()}', fontsize=16)

    # Plot Akurasi
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Training & Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Training & Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close(fig)

if __name__ == '__main__':
    # Definisi path dataset dan folder hasil
    DATASET_PATH = 'university_mental_health_iot_dataset.csv'
    RESULTS_DIR = 'comparation_results'

    # Membuat direktori 'results' jika belum ada
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Memuat Data
    print("Memuat dan memproses data...")
    X_train, X_test, y_train, y_test = data_processing.load_and_preprocess_data(
        DATASET_PATH,
        perform_eda=True, # Set True jika ingin plot EDA disimpan
        results_dir=RESULTS_DIR
    )
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    class_names = [str(c) for c in np.unique(y_train)]

    # Menghitung bobot kelas
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"\nBobot Kelas yang akan digunakan untuk perbandingan: {class_weights_dict}")
    # =========================================================================

    # Daftar Model yang Akan Dibandingkan
    models_to_compare = ['simple_rnn', 'bidirectional_rnn', 'lstm', 'gru']

    results = []

    # Looping untuk Melatih dan Mengevaluasi Setiap Model
    for model_name in models_to_compare:
        print(f"\n--- Melatih dan Mengevaluasi Model: {model_name.upper()} ---")

        # Bangun model
        tf_model = model_builder.build_model(
            model_type=model_name,
            input_shape=input_shape,
            num_classes=num_classes
        )

        # Latih model dengan menambahkan class_weight
        history = model_builder.train_model(
            tf_model,
            X_train,
            y_train,
            class_weight=class_weights_dict
        )

        history_plot_filename = f'{model_name}_training_history.png'
        history_plot_path = os.path.join(RESULTS_DIR, history_plot_filename)
        plot_training_history(history, save_path=history_plot_path, model_name=model_name)
        print(f"Plot riwayat pelatihan untuk {model_name.upper()} telah disimpan di: {history_plot_path}")

        # Evaluasi dengan classification report
        y_pred_proba = tf_model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        # Simpan metrik penting
        accuracy = report['accuracy']
        f1_score = report['weighted avg']['f1-score']

        results.append({
            'Model': model_name.upper(),
            'Accuracy': accuracy,
            'F1-Score (Weighted)': f1_score
        })
        print(f"Hasil untuk {model_name.upper()}: Akurasi={accuracy:.4f}, F1-Score={f1_score:.4f}")

    # Tampilkan Tabel Perbandingan Hasil Akhir
    results_df = pd.DataFrame(results)
    print("\n\n--- HASIL PERBANDINGAN AKHIR ---")
    print(results_df.to_string(index=False))

    # Visualisasi Perbandingan dan simpan ke file
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    sns.barplot(x='Model', y='Accuracy', data=results_df, ax=ax[0], palette='viridis')
    ax[0].set_title('Perbandingan Akurasi Model', fontsize=16)
    ax[0].set_ylim(bottom=max(0, results_df['Accuracy'].min() - 0.05))

    sns.barplot(x='Model', y='F1-Score (Weighted)', data=results_df, ax=ax[1], palette='plasma')
    ax[1].set_title('Perbandingan F1-Score (Weighted) Model', fontsize=16)
    ax[1].set_ylim(bottom=max(0, results_df['F1-Score (Weighted)'].min() - 0.05))

    plt.tight_layout()

    # Simpan plot perbandingan
    comparison_plot_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close(fig)

    print(f"\nPlot perbandingan model telah disimpan di: {comparison_plot_path}")
