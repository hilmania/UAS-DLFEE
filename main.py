# main.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Mengimpor fungsi dari file lain
import data_processing
import model

def plot_training_history(history):
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
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Membuat plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


if __name__ == '__main__':
    # Definisi path dataset
    DATASET_PATH = 'university_mental_health_iot_dataset.csv'

    # 1. Memuat dan Memproses Data
    X_train, X_test, y_train, y_test, label_encoder = data_processing.load_and_preprocess_data(
        DATASET_PATH,
        perform_eda=True
    )

    # 2. Membangun Model
    input_shape = (X_train.shape[1], X_train.shape[2]) # (1, jumlah_fitur)
    num_classes = len(label_encoder.classes_)

    rnn_model = model.build_rnn_model(input_shape=input_shape, num_classes=num_classes)
    rnn_model.summary()

    # 3. Melatih Model
    history = model.train_model(rnn_model, X_train, y_train)

    # 4. Mengevaluasi Model pada Data Uji
    model.evaluate_model(rnn_model, X_test, y_test)

    # --- Visualisasi Performa ---

    # 5. Dapatkan Prediksi
    y_pred_proba = rnn_model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1) # Ambil kelas dengan probabilitas tertinggi

    # 6. Tampilkan Laporan Klasifikasi
    class_names = label_encoder.classes_
    print("\n--- Laporan Klasifikasi ---")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 7. Plot Riwayat Pelatihan
    print("Menampilkan plot riwayat pelatihan...")
    plot_training_history(history)

    # 8. Plot Confusion Matrix
    print("Menampilkan confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, class_names)
