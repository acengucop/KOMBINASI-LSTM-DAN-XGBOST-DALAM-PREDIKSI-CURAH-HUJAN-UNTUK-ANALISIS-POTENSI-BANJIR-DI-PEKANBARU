# 🌧️ Prediksi Curah Hujan (30-Day Rolling Mean) dengan Hybrid LSTM Ensemble & XGBoost Residual Correction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Hybrid-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

Proyek ini mengembangkan sistem peramalan *time series* meteorologi untuk memprediksi **rata-rata curah hujan bergerak 30 hari (30-day rolling mean)**. 

Pendekatan utama yang digunakan adalah arsitektur **Hybrid Residual Learning**, menggabungkan kekuatan **Deep Learning (Pyramid LSTM Ensemble)** untuk menangkap pola sekuensial jangka panjang, dan **Gradient Boosting (XGBoost)** untuk mengoreksi *error* (residual), terutama pada kondisi curah hujan ekstrem (*heavy rainfall*).

---

## 📋 Daftar Isi
- [Latar Belakang](#-latar-belakang)
- [Arsitektur Model](#-arsitektur-model-hybrid-residual)
- [Dataset & Fitur](#-dataset--fitur)
- [Metodologi](#-metodologi)
- [Hasil Evaluasi](#-hasil-evaluasi)
- [Instalasi & Cara Menjalankan](#-instalasi--cara-menjalankan)
- [Struktur Folder](#-struktur-folder)

---

## 🧐 Latar Belakang

Memprediksi curah hujan harian sangat sulit karena sifat data yang *stochastic* dan penuh *noise*. Oleh karena itu, proyek ini berfokus pada **Rolling Mean 30 Hari**, yang lebih relevan untuk perencanaan pertanian, manajemen waduk, dan mitigasi bencana jangka menengah.

Tantangan utama dalam prediksi hujan adalah model sering kali *"under-predict"* (memprediksi terlalu rendah) pada saat terjadi hujan lebat. Solusi yang ditawarkan dalam proyek ini adalah:
1.  **Weighted Huber Loss:** Memberikan penalti lebih besar jika model salah memprediksi hujan deras.
2.  **Hybrid Correction:** Menggunakan XGBoost untuk mempelajari pola *error* dari LSTM dan mengoreksinya.

---

## 🧠 Arsitektur Model: Hybrid Residual

Sistem ini bekerja dalam dua tahap (Stage):

### Stage 1: Pyramid LSTM Ensemble
Model dasar adalah Ensemble dari 3 model LSTM dengan inisialisasi *seed* berbeda.
* **Input:** Sekuens data meteorologi 120 hari terakhir (*Lookback window*).
* **Struktur:** Arsitektur "Piramida" (Layer makin dalam makin kecil unitnya) untuk mengekstrak fitur secara hierarkis.
    * `LSTM (64, return_seq)` → `Dropout`
    * `LSTM (32)` → `Dropout`
    * `Dense (16, ReLU)` → `Output (1)`
* **Loss Function:** Weighted Huber Loss (Fokus pada nilai ekstrem).

### Stage 2: XGBoost Residual Correction
Model kedua (XGBoost) dilatih untuk memprediksi **selisih** antara prediksi LSTM dan nilai asli.
* **Formula:** $y_{final} = y_{LSTM} + y_{Residual\_XGB}$
* **Strategi:** XGBoost dilatih dengan *sample weights* yang berat pada data ekstrem (> P90), sehingga model sangat sensitif terhadap lonjakan curah hujan yang gagal ditangkap LSTM.

---

## 📊 Dataset & Fitur

Dataset mencakup data meteorologi harian dengan pembagian data secara **Kronologis** (Tanpa pengacakan/shuffle) untuk menjaga integritas waktu:
* **Train:** 70%
* **Validation:** 15%
* **Test:** 15%

### Feature Engineering
Fitur yang digunakan sangat komprehensif untuk menangkap dinamika cuaca:

| Kategori | Fitur | Deskripsi |
| :--- | :--- | :--- |
| **Meteorologi** | `TN`, `TX`, `TAVG` | Suhu Min, Max, Rata-rata |
| | `RH_AVG`, `SS` | Kelembapan rata-rata, Penyinaran Matahari |
| | `FF_X`, `FF_AVG` | Kecepatan Angin Max & Rata-rata |
| **Siklus Waktu** | `month_sin`, `month_cos` | Encoding siklik bulan (Januari dekat Desember) |
| **Angin (Vektor)** | `DDD_X_sin/cos` | Encoding arah angin agar 0° dekat dengan 360° |
| **Rain Memory** | `RR_lag` (7, 14, 30, 60, 90) | Nilai hujan masa lampau (Lag features) |
| **Rolling Features** | `RR_roll_mean` | Rata-rata hujan 7/14/30 hari (Shifted-1 untuk cegah *leakage*) |

> **Catatan Penting:** Semua fitur *rolling* dan *lag* curah hujan telah di-geser (shifted) minimal 1 hari ke belakang. Ini menjamin model tidak "mengintip" target hari ini (Data Leakage Prevention).

---

## ⚙️ Metodologi

1.  **Preprocessing:** Normalisasi (MinMax/Standard Scaler) dan pembentukan *sequence* 3D `(samples, 120, features)` untuk LSTM.
2.  **Training LSTM:** Melatih 3 model LSTM dengan seed `[7, 42, 202]` dan merata-ratakan prediksinya.
3.  **Residual Calculation:** Menghitung error prediksi LSTM: $Residual = Actual - Predicted_{LSTM}$.
4.  **Training XGBoost:** Melatih XGBoost menggunakan fitur tabular yang selaras (*aligned*) dengan *lookback window* LSTM untuk memprediksi nilai residual tersebut.
5.  **Final Prediction:** Menjumlahkan hasil prediksi LSTM Ensemble dengan prediksi koreksi dari XGBoost.

---

## 📈 Hasil Evaluasi

Pengujian dilakukan pada data **Test Set** (15% data terakhir).

| Model | R² Score | Keterangan |
| :--- | :--- | :--- |
| **LSTM Ensemble** | 0.9522 | Performa dasar yang sudah sangat baik. |
| **Hybrid Residual** | **0.9652** | Peningkatan akurasi setelah koreksi XGBoost. |

### Analisis Kemampuan Prediksi Puncak (Peak Analysis)
Evaluasi khusus pada data curah hujan ekstrem (Nilai > Persentil 90 Training):

| Model | R² (Peak Only) | Analisis |
| :--- | :--- | :--- |
| LSTM Ensemble | 0.4183 | Cenderung *underestimate* pada hujan lebat. |
| **Hybrid Residual** | **0.6198** | **Peningkatan Signifikan (+20%)** dalam menangkap ekstrem. |

---

## 💻 Instalasi & Cara Menjalankan

### 1. Clone Repository
```bash
git clone [https://github.com/username-kamu/nama-repo-kamu.git](https://github.com/username-kamu/nama-repo-kamu.git)
cd nama-repo-kamu
2. Install Dependencies
Pastikan Anda menggunakan Python 3.8+.

Bash

pip install -r requirements.txt
Dependencies utama: tensorflow, xgboost, pandas, numpy, scikit-learn.

3. Konfigurasi
Buka file script/notebook dan sesuaikan path dataset:

Python

CSV_PATH = "data/dataset_hujan.csv"
4. Jalankan Training
Anda dapat menjalankan notebook atau script python:

Bash

python src/train_hybrid_model.py
📂 Struktur Folder
├── data/
│   ├── raw_data.csv           # Data mentah
│   └── processed_data.csv     # Data bersih setelah cleaning
├── models/
│   ├── lstm_ensemble/         # Saved models (.h5)
│   └── xgboost_residual.json  # Saved XGBoost model
├── notebooks/
│   └── Rainfall_Hybrid.ipynb  # Notebook eksperimen utama
├── src/
│   ├── preprocessing.py       # Skrip normalisasi & windowing
│   ├── train.py               # Skrip training utama
│   └── utils.py               # Fungsi metrik & visualisasi
├── README.md
└── requirements.txt
📝 Catatan Tambahan
Leakage Prevention: Fitur rolling menggunakan shift(1) sangat krusial. Jangan dihapus.

Hardware: Training LSTM direkomendasikan menggunakan GPU (NVIDIA RTX/T4 di Colab) untuk kecepatan optimal. XGBoost berjalan cepat di CPU.

Dibuat oleh [Nama Kamu]


---

### Tips Tambahan dari Saya:
1.  **Gambar Arsitektur:** Jika kamu punya diagram model (walaupun coretan tangan yang dirapikan atau screenshot dari slide PPT), kamu bisa upload ke repo lalu masukkan link-nya di bagian "Arsitektur Model". Itu menaikkan nilai estetika repo secara drastis.
2.  **Requirements.txt:** Jangan lupa buat file `requirements.txt`. Kamu bisa membuatnya otomatis dengan mengetik `pip freeze > requirements.txt` di terminal kamu (sebaiknya di dalam virtual environment).
3.  **Nama Repo:** Ganti `username-kamu` dan `nama-repo-kamu` di bagian instalasi dengan link GitHub aslimu nanti.
