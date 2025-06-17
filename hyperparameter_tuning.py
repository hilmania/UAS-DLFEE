import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import os

# Impor modul kita
import data_processing
import model_tune

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

# --- Objective Function ---
def objective(trial):
    units = trial.suggest_int('units', 32, 128, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    tf.keras.backend.clear_session()

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))

    rnn_model = model_tune.build_rnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        rnn_units_1=units,
        rnn_units_2=int(units/2),
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = rnn_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        class_weight=class_weights_dict,
        verbose=0
    )

    val_accuracy = np.max(history.history['val_accuracy'])
    return val_accuracy

# --- Bagian Utama ---
if __name__ == '__main__':
    # Membuat folder results jika belum ada
    RESULTS_DIR = 'tuning_results'
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # --- Mendefinisikan Storage untuk Menyimpan Progres ---
    study_name = "rnn_hyperparameter_tuning"  # Nama unik untuk studi ini
    storage_name = f"sqlite:///{study_name}.db" # File database akan dibuat

    # Buat atau muat 'study' dari database
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True # Muat progres jika file .db sudah ada
    )

    print(f"\n--- Memulai/Melanjutkan Hyperparameter Tuning ---")
    print(f"Progres akan disimpan di file: {study_name}.db")
    # Jika sudah ada trial sebelumnya, Optuna akan melanjutkannya
    study.optimize(objective, n_trials=50)

    # Menampilkan hasil terbaik
    print("\n--- Tuning Selesai ---")
    print("Jumlah trial yang selesai: ", len(study.trials))
    print("\nTrial terbaik:")
    best_trial = study.best_trial
    print(f"  Value (Max Validation Accuracy): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    file_path = os.path.join(RESULTS_DIR, "hasil_tuning_terbaik.txt")
    try:
        with open(file_path, 'w') as f:
            f.write("=== Hasil Tuning Hyperparameter Terbaik ===\n\n")
            f.write(f"Studi: {study.study_name}\n")
            f.write(f"Jumlah Total Trial: {len(study.trials)}\n\n")
            f.write(f"Nilai Terbaik (Max Validation Accuracy): {best_trial.value}\n\n")
            f.write("Parameter Terbaik:\n")

            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
                f.write(f"  - {key}: {value}\n")

        print(f"\n[INFO] Hasil terbaik juga telah disimpan ke file: {file_path}")
    except Exception as e:
        print(f"\n[ERROR] Gagal menyimpan hasil ke file: {e}")

    # --- Menyimpan Hasil Tuning sebagai Grafik ---
    print("\n--- Menyimpan Grafik Hasil Tuning ---")

    # 1. Grafik Riwayat Optimisasi
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1_path = os.path.join(RESULTS_DIR, "tuning_history.png")
        fig1.write_image(fig1_path)
        print(f"Grafik riwayat optimisasi disimpan di: {fig1_path}")

        # 2. Grafik Pentingnya Hyperparameter
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2_path = os.path.join(RESULTS_DIR, "tuning_param_importances.png")
        fig2.write_image(fig2_path)
        print(f"Grafik pentingnya parameter disimpan di: {fig2_path}")

        # 3. Grafik Slice Plot
        fig3 = optuna.visualization.plot_slice(study)
        fig3_path = os.path.join(RESULTS_DIR, "tuning_slice_plot.png")
        fig3.write_image(fig3_path)
        print(f"Grafik slice plot disimpan di: {fig3_path}")
    except (ValueError, ImportError) as e:
        print(f"\nTidak bisa membuat plot, mungkin karena library visualisasi belum terinstal.")
        print("Silakan jalankan: pip install plotly kaleido")
        print(f"Error: {e}")
