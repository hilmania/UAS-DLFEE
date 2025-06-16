# main.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import os

# Mengimpor fungsi dari file lain
import data_processing
import model

def plot_training_history(history, save_path=None):
    """Membuat plot akurasi dan loss selama pelatihan."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

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

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot disimpan di {save_path}")
    else:
        plt.show()

    plt.close(fig)  # Tutup plot untuk menghemat memori

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Membuat plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix disimpan di {save_path}")
    else:
        plt.show()

    plt.close()  # Tutup plot untuk menghemat memori

if __name__ == '__main__':
    # Definisi path dataset
    DATASET_PATH = 'university_mental_health_iot_dataset.csv'
    RESULTS_DIR = 'results'

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # 1. Memuat dan Memproses Data
    X_train, X_test, y_train, y_test = data_processing.load_and_preprocess_data(
        DATASET_PATH,
        perform_eda=True,
        results_dir = RESULTS_DIR
    )

    # 2. Membangun Model
    input_shape = (X_train.shape[1], X_train.shape[2]) # (1, jumlah_fitur)
    num_classes = len(np.unique(y_train))

    rnn_model = model.build_rnn_model(input_shape=input_shape, num_classes=num_classes)
    rnn_model.summary()

    # Hitung bobot untuk setiap kelas
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"\nMenggunakan Class Weights untuk training: {class_weights_dict}")

    # 3. Melatih Model
    history = rnn_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights_dict,  # Ini bagian terpenting
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
    )

    # 4. Mengevaluasi Model pada Data Uji
    model.evaluate_model(rnn_model, X_test, y_test)

    # --- Visualisasi Performa ---

    # 5. Dapatkan Prediksi
    y_pred_proba = rnn_model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1) # Ambil kelas dengan probabilitas tertinggi

    # 6. Tampilkan Laporan Klasifikasi
    class_names = [str(c) for c in np.unique(y_train)]
    print("\n--- Laporan Klasifikasi ---")
    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0 # Mencegah warning/error jika ada kelas yg tidak terprediksi
    ))

    # 7. Plot Riwayat Pelatihan
    print("Menampilkan plot riwayat pelatihan...")
    history_plot_path = os.path.join(RESULTS_DIR, 'training_history.png')
    plot_training_history(history, save_path=history_plot_path)
    print("Plot riwayat pelatihan disimpan di:", history_plot_path)

    # 8. Plot Confusion Matrix
    print("Menampilkan confusion matrix...")
    cm_plot_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, class_names, save_path=cm_plot_path)
    print("Confusion matrix disimpan di:", cm_plot_path)
