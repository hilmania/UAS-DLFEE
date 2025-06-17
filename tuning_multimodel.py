import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import os

import data_processing
import model_builder

# --- Memuat Data ---
print("Memuat dan memproses data untuk tuning...")
X_train, X_test, y_train, y_test = data_processing.load_and_preprocess_data(
    'university_mental_health_iot_dataset.csv',
    perform_eda=False
)
print("Data siap.")

# --- Menghitung Class Weight ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print(f"\nMenggunakan Class Weights untuk tuning: {class_weights_dict}")


def objective(trial):
    """
    Fungsi objective yang dioptimalkan Optuna.
    Sekarang juga memilih jenis model terbaik.
    """
    tf.keras.backend.clear_session()

    # ---'model_type' sebagai hyperparameter ---
    model_type = trial.suggest_categorical('model_type', ['simple_rnn', 'bidirectional_rnn', 'lstm', 'gru'])

    # Definisikan hyperparameter lainnya
    units = trial.suggest_int('units', 32, 128, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Bangun model menggunakan model_builder
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))

    rnn_model = model_builder.build_model(
        model_type=model_type,  # <-- Gunakan model yang dipilih trial
        input_shape=input_shape,
        num_classes=num_classes,
        units=units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Latih model
    history = rnn_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        class_weight=class_weights_dict,
        verbose=0
    )

    # Kembalikan metrik
    val_accuracy = np.max(history.history['val_accuracy'])
    return val_accuracy

if __name__ == '__main__':
    RESULTS_DIR = 'results_multi_model_tuning'
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    study_name = "multi_model_rnn_tuning_batch_2"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True
    )

    # Karena ruang pencarian lebih besar, mungkin perlu lebih banyak trial
    n_trials = 100
    print(f"\n--- Memulai Multi-Model Tuning untuk {n_trials} trial ---")
    print(f"Progres akan disimpan di file: {study_name}.db")

    study.optimize(objective, n_trials=n_trials)

    # Menampilkan hasil terbaik
    print("\n--- Tuning Selesai ---")
    print("Jumlah trial yang selesai: ", len(study.trials))
    print("\nTrial terbaik (Model & Hyperparameter):")
    best_trial = study.best_trial
    print(f"  Value (Max Validation Accuracy): {best_trial.value:.4f}")

    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("\n--- Menyimpan hasil terbaik ke file... ---")
    file_path = os.path.join(RESULTS_DIR, "best_hyperparameters.txt")

    # simpan hasil terbaik ke file
    try:
        with open(file_path, 'w') as f:
            f.write("=== HASIL HYPERPARAMETER TUNING TERBAIK ===\n\n")
            f.write(f"Studi: {study.study_name}\n")
            f.write(f"Jumlah Trial yang Dijalankan: {len(study.trials)}\n\n")
            f.write("=============================================\n")
            f.write("HASIL TERBAIK:\n")
            f.write("=============================================\n")
            f.write(f"Nilai (Max Validation Accuracy): {best_trial.value:.6f}\n\n")
            f.write("Parameter Terbaik:\n")
            for key, value in best_trial.params.items():
                f.write(f"  - {key}: {value}\n")

        print(f"Hasil terbaik telah disimpan di: {file_path}")
    except Exception as e:
        print(f"Gagal menyimpan file. Error: {e}")

    # Menyimpan Grafik Hasil Tuning
    print("\n--- Menyimpan Grafik Hasil Tuning ---")
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(os.path.join(RESULTS_DIR, "tuning_history_multi_model.png"))

        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(os.path.join(RESULTS_DIR, "tuning_param_importances_multi_model.png"))

        fig3 = optuna.visualization.plot_slice(study)
        fig3.write_image(os.path.join(RESULTS_DIR, "tuning_slice_plot_multi_model.png"))

        print(f"Semua grafik tuning telah disimpan di folder '{RESULTS_DIR}'.")
    except (ValueError, ImportError):
        print("\nTidak bisa membuat plot. Jalankan: pip install plotly kaleido")
