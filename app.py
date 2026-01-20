import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load Model
# ===============================
model = joblib.load("xgb_dropout_model.pkl")

st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    layout="centered"
)

st.title("🎓 Prediksi Dropout Mahasiswa")
st.write(
    "Aplikasi ini memprediksi apakah seorang mahasiswa "
    "berpotensi **Lulus (Graduate)** atau **Dropout** "
    "berdasarkan data akademik."
)

# ===============================
# Input Mahasiswa
# ===============================
st.subheader("📌 Informasi Mahasiswa")

admission_grade = st.slider(
    "Nilai Masuk (0–100)",
    0, 100, 75
)

# Dataset asli pakai skala 0–200 → dikonversi proporsional
admission_grade = admission_grade * 2

age = st.slider(
    "Usia Saat Masuk Kuliah",
    16, 60, 18
)

gender = st.selectbox(
    "Jenis Kelamin",
    ["Perempuan", "Laki-laki"]
)

tuition = st.selectbox(
    "Status Pembayaran UKT",
    ["Lunas", "Belum Lunas"]
)

debtor = st.selectbox(
    "Status Hutang Akademik",
    ["Tidak", "Ya"]
)

# ===============================
# Akademik Semester 1
# ===============================
st.markdown("### 📚 Akademik Semester 1")

sem1_enrolled = st.number_input(
    "Jumlah Mata Kuliah Diambil",
    0, 20, 6
)

sem1_approved = st.number_input(
    "Jumlah Mata Kuliah Lulus",
    0, 20, 5
)

nilai_sem1_indo = st.slider(
    "Rata-rata Nilai Semester 1 (0–100)",
    0, 100, 70
)

# Konversi ke skala dataset (0–20)
sem1_grade = nilai_sem1_indo / 5

# ===============================
# Akademik Semester 2
# ===============================
st.markdown("### 📚 Akademik Semester 2")

sem2_enrolled = st.number_input(
    "Jumlah Mata Kuliah Diambil (Sem 2)",
    0, 20, 6
)

sem2_approved = st.number_input(
    "Jumlah Mata Kuliah Lulus (Sem 2)",
    0, 20, 5
)

nilai_sem2_indo = st.slider(
    "Rata-rata Nilai Semester 2 (0–100)",
    0, 100, 70
)

sem2_grade = nilai_sem2_indo / 5

# ===============================
# Buat DataFrame Input
# ===============================
input_df = pd.DataFrame(
    0,
    index=[0],
    columns=model.feature_names_in_
)

# Isi fitur numerik
input_df["admission grade"] = admission_grade
input_df["age at enrollment"] = age

input_df["curricular units 1st sem (enrolled)"] = sem1_enrolled
input_df["curricular units 1st sem (approved)"] = sem1_approved
input_df["curricular units 1st sem (grade)"] = sem1_grade

input_df["curricular units 2nd sem (enrolled)"] = sem2_enrolled
input_df["curricular units 2nd sem (approved)"] = sem2_approved
input_df["curricular units 2nd sem (grade)"] = sem2_grade

# Encoding sesuai training
input_df["gender"] = 1 if gender == "Laki-laki" else 0
input_df["tuition fees up to date"] = 1 if tuition == "Lunas" else 0
input_df["debtor"] = 1 if debtor == "Ya" else 0

# ===============================
# Prediksi
# ===============================
if st.button("🔍 Prediksi Status Mahasiswa"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("❌ **Prediksi: Mahasiswa Berpotensi Dropout**")
    else:
        st.success("✅ **Prediksi: Mahasiswa Berpotensi Lulus (Graduate)**")

    st.markdown("### 📊 Probabilitas Prediksi")
    st.write(f"🎓 Lulus (Graduate): **{probability[0]*100:.2f}%**")
    st.write(f"⚠️ Dropout: **{probability[1]*100:.2f}%**")
