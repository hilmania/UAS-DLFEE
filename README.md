# Analisis Kesehatan Mental Mahasiswa menggunakan Model RNN

Proyek ini bertujuan untuk mengklasifikasikan status kesehatan mental mahasiswa berdasarkan data survei dan sensor IoT dari dataset *University Mental Health*. Proyek ini mengimplementasikan, membandingkan, dan mengevaluasi beberapa jenis model Recurrent Neural Network (RNN) seperti Simple RNN, Bidirectional RNN, LSTM, dan GRU untuk menemukan arsitektur yang paling efektif.

## Fitur Utama
- **Pra-pemrosesan Data**: Membersihkan, melakukan scaling, dan membentuk data agar siap digunakan oleh model RNN.
- **Analisis Data Eksplorasi (EDA)**: Visualisasi distribusi data dan korelasi antar fitur yang hasilnya disimpan secara otomatis.
- **Implementasi Berbagai Model RNN**:
  - Simple RNN (Unidirectional)
  - Bidirectional RNN
  - Long Short-Term Memory (LSTM)
  - Gated Recurrent Unit (GRU)
- **Kerangka Perbandingan Model**: Skrip untuk melatih semua model secara berurutan dan membandingkan performanya dalam satu tabel dan visualisasi.
- **Tuning Hyperparameter**: Implementasi Optuna untuk mencari hyperparameter terbaik secara otomatis.
- **Penyimpanan Hasil**: Semua output, termasuk plot performa dan analisis, disimpan secara otomatis ke dalam folder `results`.

## Struktur Proyek
```
.
├── 📂 results/                # Folder untuk menyimpan semua output (plot, dll.)
├── 📜 data_processing.py       # Modul untuk memuat, membersihkan, dan memproses data.
├── 📜 model.py                 # Modul untuk membangun model RNN tunggal (digunakan oleh main.py).
├── 📜 model_builder.py         # Modul "pabrik" untuk membangun berbagai jenis model RNN (untuk perbandingan).
├── 📜 main.py                  # Skrip utama untuk menjalankan satu model Simple RNN.
├── 📜 run_comparison.py        # Skrip utama untuk membandingkan semua jenis model RNN.
├── 📜 hyperparameter_tuning.py # Skrip untuk melakukan tuning dengan Optuna.
├── 📜 debug_test.py            # Skrip utilitas untuk debugging instalasi library.
├── 📜 environment.yml          # File environment Conda untuk setup proyek.
├── 📜 university_mental_health_iot_dataset.csv # Dataset yang digunakan.
└── 📜 README.md                # File panduan ini.
```

## Instalasi & Setup
Proyek ini menggunakan **Conda** untuk manajemen environment dan dependensi. Ini memastikan semua library memiliki versi yang kompatibel.

### Langkah 1: Instalasi Conda
Jika Anda belum memiliki Conda, silakan instal **Miniconda** (versi ringan dari Anaconda).
- Kunjungi halaman unduh Miniconda: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- Pilih installer yang sesuai dengan sistem operasi Anda (Windows, macOS, atau Linux) dan ikuti petunjuk instalasinya.

### Langkah 2: Kloning Repositori
Buka terminal atau Git Bash, lalu kloning repositori ini ke komputer lokal Anda.
```bash
git clone https://github.com/hilmania/UAS-DLFEE.git
cd UAS-DLFEE
```

### Langkah 3: Buat dan Aktifkan Environment Conda
File `environment.yml` berisi semua library yang dibutuhkan proyek ini. Buat environment dari file tersebut dengan perintah berikut:

```bash
# Perintah ini akan membuat environment baru bernama 'uas-dl'
conda env create -f environment.yml
```
Proses ini mungkin memakan waktu beberapa menit karena akan mengunduh semua library yang diperlukan. Setelah selesai, aktifkan environment yang baru dibuat:
```bash
conda activate uas-deep-learning
```
**PENTING**: Setiap kali Anda ingin mengerjakan proyek ini di sesi terminal yang baru, Anda harus selalu mengaktifkan environment ini terlebih dahulu dengan perintah `conda activate uas-deep-learning`.

## Cara Penggunaan
Pastikan Anda sudah mengaktifkan environment Conda (`conda activate uas-deep-learning`) sebelum menjalankan skrip apa pun.

### 1. Menjalankan Satu Model Sederhana
Untuk melatih dan mengevaluasi satu model Simple RNN dan menyimpan plot performanya.
```bash
python main.py
```

### 2. Membandingkan Beberapa Model
Untuk melatih semua model (Simple RNN, Bidirectional, LSTM, GRU), membandingkan performanya, dan menyimpan semua plot yang relevan. Ini adalah skrip utama untuk eksperimen.
```bash
python run_comparison.py
```

### 3. Melakukan Tuning Hyperparameter
Untuk menjalankan Optuna dan secara otomatis mencari kombinasi hyperparameter terbaik untuk model.
```bash
python hyperparameter_tuning.py
```

## Output Proyek
Semua hasil visual dari eksekusi skrip akan disimpan di dalam folder `results`. Ini termasuk:
- **Plot EDA** (jika diaktifkan): `eda_target_distribution.png`, `eda_feature_distributions.png`, dll.
- **Plot Riwayat Training**: `history_simple_rnn.png`, `history_lstm.png`, dll. (dihasilkan oleh `run_comparison.py`).
- **Plot Perbandingan Model**: `model_comparison.png` yang merangkum akurasi dan F1-score semua model.
- **Plot dari `main.py`**: `training_history.png` dan `confusion_matrix.png`.

## Dependensi Utama
Semua dependensi diatur oleh `environment.yml`. Library utama yang digunakan adalah:
- Python 3.12
- TensorFlow (Keras)
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Optuna
