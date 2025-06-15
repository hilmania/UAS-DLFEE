# hyperparameter_tuning.py

import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Mengimpor modul yang sudah kita buat
import data_processing
import model

# Memuat data sekali saja di luar objective function untuk efisiensi
print("Memuat dan memproses data untuk tuning...")
X_train, X_test, y_train, y_test, encoder = data_processing.load_and_preprocess_data(
    'university_mental_health_iot_dataset.csv',
    perform_eda=False # Matikan EDA selama tuning
)
print("Data siap.")

def objective(trial):
    """
    Fungsi objective yang akan dioptimalkan oleh Optuna.
    Setiap 'trial' adalah satu set hyperparameter.
    """
    # 1. Definisikan Ruang Pencarian (Search Space) untuk Hyperparameter
    rnn_units_1 = trial.suggest_int('rnn_units_1', 32, 128, step=16)
    rnn_units_2 = trial.suggest_int('rnn_units_2', 16, 64, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Membersihkan session Keras untuk mencegah kebocoran model
    tf.keras.backend.clear_session()

    # 2. Bangun model dengan hyperparameter dari trial ini
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(encoder.classes_)

    rnn_model = model.build_rnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        rnn_units_1=rnn_units_1,
        rnn_units_2=rnn_units_2,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    # Callback untuk menghentikan training lebih awal jika tidak ada peningkatan
    # Ini penting untuk mempercepat proses tuning
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

    # 3. Latih model
    history = rnn_model.fit(
        X_train,
        y_train,
        epochs=30, # Gunakan epoch yang tidak terlalu banyak untuk tuning
        batch_size=batch_size,
        validation_split=0.2, # Gunakan sebagian data train untuk validasi
        callbacks=[early_stopping],
        verbose=0 # Matikan log training agar output bersih
    )

    # 4. Kembalikan metrik yang ingin dioptimalkan (akurasi validasi tertinggi)
    val_accuracy = np.max(history.history['val_accuracy'])
    return val_accuracy

if __name__ == '__main__':
    # Buat objek 'study' Optuna. Kita ingin memaksimalkan akurasi.
    study = optuna.create_study(direction='maximize')

    # Mulai proses optimisasi. Optuna akan memanggil fungsi 'objective' berulang kali.
    # n_trials adalah jumlah total kombinasi hyperparameter yang akan dicoba.
    print("\n--- Memulai Hyperparameter Tuning dengan Optuna ---")
    study.optimize(objective, n_trials=50) # Anda bisa menambah/mengurangi jumlah trial

    # Tampilkan hasil terbaik
    print("\n--- Tuning Selesai ---")
    print("Jumlah trial yang selesai: ", len(study.trials))

    print("\nTrial terbaik:")
    best_trial = study.best_trial

    print(f"  Value (Max Validation Accuracy): {best_trial.value:.4f}")

    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("\nSaran: Gunakan parameter di atas dalam file 'main.py' untuk melatih model final.")
