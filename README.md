# Prediksi Dropout Mahasiswa (XGBoost)

Proyek ini membangun sistem prediksi risiko dropout mahasiswa menggunakan algoritma XGBoost dan diimplementasikan dalam aplikasi web berbasis Streamlit.

Model dilatih menggunakan dataset Predict Students' Dropout and Academic Success dari UCI Machine Learning Repository dengan data akademik dan demografis mahasiswa.

---

## � Struktur File

```
dropout-predict/
├── app.py                      # Aplikasi Streamlit (Interface pengguna)
├── model.py                    # Script training model XGBoost
├── data.csv                    # Dataset historis untuk training model
├── xgb_dropout_model.pkl       # Model yang sudah dilatih (generated)
└── README.md                   # Dokumentasi proyek (file ini)
```

### Deskripsi File

| File                      | Fungsi                                                                                                                                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **app.py**                | Aplikasi web Streamlit yang menyediakan interface untuk menerima input mahasiswa dan menampilkan prediksi dropout/graduate. File ini memuat model dari `xgb_dropout_model.pkl` dan menjalankan prediksi real-time. |
| **model.py**              | Script untuk melatih model XGBoost menggunakan dataset dari `data.csv`. Menghasilkan file model `xgb_dropout_model.pkl` yang akan digunakan oleh `app.py`. Jalankan file ini sekali untuk generate model.          |
| **data.csv**              | Dataset historis yang berisi data mahasiswa dengan kolom-kolom demografis dan akademik. Digunakan untuk training model XGBoost. Format: CSV dengan delimiter `;`                                                   |
| **xgb_dropout_model.pkl** | Model machine learning yang sudah dilatih dan disimpan dalam format pickle. Dihasilkan dari `model.py` dan digunakan oleh `app.py` untuk prediksi. ⚠️ File ini di-generate otomatis, jangan diedit manual.         |
| **README.md**             | File dokumentasi yang berisi penjelasan lengkap tentang proyek, cara instalasi, penggunaan, arsitektur, dan teknologi yang digunakan.                                                                              |

---

## �🔧 Dokumentasi Teknis

### Arsitektur Sistem

Sistem terdiri dari dua komponen utama:

1. **Model Machine Learning** (`model.py`)

   - Melatih model XGBoost menggunakan dataset historis
   - Menghasilkan file model yang disimpan sebagai `xgb_dropout_model.pkl`

2. **Aplikasi Web** (`app.py`)
   - Interface pengguna berbasis Streamlit
   - Menerima input data mahasiswa dan memberikan prediksi real-time

### Alur Proses

```
Data Input Pengguna
    ↓
Normalisasi & Preprocessing
    ↓
Model XGBoost
    ↓
Prediksi (Graduate / Dropout)
    ↓
Hasil Ditampilkan di UI
```

### Teknologi & Library

| Komponen              | Library                  | Fungsi                           |
| --------------------- | ------------------------ | -------------------------------- |
| **Machine Learning**  | XGBoost                  | Algoritma klasifikasi            |
| **Data Processing**   | Pandas                   | Manipulasi dan normalisasi data  |
| **Model Persistence** | Joblib                   | Menyimpan dan memuat model       |
| **Web Framework**     | Streamlit                | Interface aplikasi web           |
| **Scientific**        | Scikit-learn             | Preprocessing dan evaluasi model |
| **Resampling**        | Imbalanced-learn (SMOTE) | Mengatasi data imbalance         |

### Metode & Evaluasi

- **Algoritma**: XGBoost (Extreme Gradient Boosting)
- **Jenis Masalah**: Klasifikasi biner (Graduate / Dropout)
- **Metrik Evaluasi**: Accuracy, Precision, Recall, F1-Score
- **Resampling**: SMOTE untuk menangani ketidakseimbangan data

### Fitur Input Model

**Data Demografis:**

- Admission grade
- Age at enrollment
- Gender
- Tuition fees up to date
- Debtor

**Data Akademik Semester 1:**

- Curricular units enrolled
- Curricular units approved
- Grade

**Data Akademik Semester 2:**

- Curricular units enrolled
- Curricular units approved
- Grade

---

## 👥 Dokumentasi Penggunaan

### Instalasi & Setup

1. **Install Dependencies**

   ```bash
   pip install streamlit joblib pandas xgboost scikit-learn imbalanced-learn
   ```

2. **Persiapan File**
   - Pastikan file `data.csv` tersedia untuk training
   - Jalankan `model.py` untuk generate model:
     ```bash
     python model.py
     ```
   - Ini akan menghasilkan file `xgb_dropout_model.pkl`

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan otomatis membuka di browser pada `http://localhost:8501`

### Cara Menggunakan Aplikasi

1. **Masukkan Informasi Mahasiswa**

   - **Nilai Masuk**: Geser slider untuk nilai (0–100), akan dikonversi proporsional ke skala dataset
   - **Usia Saat Masuk Kuliah**: Pilih usia mahasiswa (16–60 tahun)
   - **Jenis Kelamin**: Pilih Perempuan atau Laki-laki
   - **Status Pembayaran UKT**: Pilih Lunas atau Belum Lunas
   - **Status Hutang Akademik**: Pilih Tidak atau Ya

2. **Masukkan Data Akademik Semester 1**

   - **Jumlah Mata Kuliah Diambil**: Berapa mata kuliah yang diambil
   - **Jumlah Mata Kuliah Lulus**: Berapa mata kuliah yang berhasil diselesaikan
   - **Rata-rata Nilai Semester 1**: Nilai rata-rata (0–100)

3. **Masukkan Data Akademik Semester 2**

   - **Jumlah Mata Kuliah Diambil (Sem 2)**: Berapa mata kuliah yang diambil
   - **Jumlah Mata Kuliah Lulus (Sem 2)**: Berapa mata kuliah yang berhasil diselesaikan
   - **Rata-rata Nilai Semester 2**: Nilai rata-rata (0–100)

4. **Dapatkan Hasil Prediksi**
   - Sistem akan otomatis memproses data dan menampilkan prediksi
   - Hasil menunjukkan apakah mahasiswa berpotensi **Lulus (Graduate)** atau **Dropout**
"# prediksi-do-streamlit" 
