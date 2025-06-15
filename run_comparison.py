# run_comparison.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Impor modul kita
import data_processing
import model_builder

if __name__ == '__main__':
    # 1. Memuat Data
    print("Memuat dan memproses data...")
    X_train, X_test, y_train, y_test, label_encoder = data_processing.load_and_preprocess_data(
        'university_mental_health_iot_dataset.csv'
    )
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_

    # 2. Daftar Model yang Akan Dibandingkan
    models_to_compare = ['simple_rnn', 'bidirectional_rnn', 'lstm', 'gru']

    results = []

    # 3. Looping untuk Melatih dan Mengevaluasi Setiap Model
    for model_name in models_to_compare:
        print(f"\n--- Melatih dan Mengevaluasi Model: {model_name.upper()} ---")

        # Bangun model
        tf_model = model_builder.build_model(
            model_type=model_name,
            input_shape=input_shape,
            num_classes=num_classes
        )

        # Latih model
        model_builder.train_model(tf_model, X_train, y_train)

        # Evaluasi dengan classification report
        y_pred_proba = tf_model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        # Simpan metrik penting
        accuracy = report['accuracy']
        f1_score = report['weighted avg']['f1-score']

        results.append({
            'Model': model_name.upper(),
            'Accuracy': accuracy,
            'F1-Score (Weighted)': f1_score
        })
        print(f"Hasil untuk {model_name.upper()}: Akurasi={accuracy:.4f}, F1-Score={f1_score:.4f}")

    # 4. Tampilkan Tabel Perbandingan Hasil Akhir
    results_df = pd.DataFrame(results)
    print("\n\n--- HASIL PERBANDINGAN AKHIR ---")
    print(results_df.to_string(index=False))

    # 5. Visualisasi Perbandingan
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Akurasi
    sns.barplot(x='Model', y='Accuracy', data=results_df, ax=ax[0], palette='viridis')
    ax[0].set_title('Perbandingan Akurasi Model', fontsize=16)
    ax[0].set_ylim(bottom=max(0, results_df['Accuracy'].min() - 0.05))


    # Plot F1-Score
    sns.barplot(x='Model', y='F1-Score (Weighted)', data=results_df, ax=ax[1], palette='plasma')
    ax[1].set_title('Perbandingan F1-Score (Weighted) Model', fontsize=16)
    ax[1].set_ylim(bottom=max(0, results_df['F1-Score (Weighted)'].min() - 0.05))

    plt.tight_layout()
    plt.show()
