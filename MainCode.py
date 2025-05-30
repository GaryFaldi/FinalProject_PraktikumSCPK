import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("Sistem Pendukung Keputusan: Deteksi Mahasiswa Paling Rawan Depresi (Metode WP)")

# Load dataset
data = pd.read_csv("CleanedDataset.csv")



# Rename kolom agar mudah
data = data.rename(columns={
    "Academic Pressure": "academic_pressure",
    "CGPA": "cgpa",
    "Study Satisfaction": "study_satisfaction",
    "Sleep Duration": "sleep_duration",
    "Have you ever had suicidal thoughts ?": "suicidal_thoughts",
    "Work/Study Hours": "study_hours",
    "Financial Stress": "financial_stress"
})

# Mapping categorical data
data["suicidal_thoughts"] = data["suicidal_thoughts"].map({"Yes": 1, "No": 0})
data["sleep_duration"] = data["sleep_duration"].astype(str).str.replace("'", "").str.strip()
data["sleep_duration"] = data["sleep_duration"].map({
    "Less than 5 hours": 4,
    "5-6 hours": 5,
    "6-7 hours": 6,
    "7-8 hours": 7,
    "8-9 hours": 8,
    "More than 8 hours": 9
})

# Kriteria dan bobot (exclude suicidal_thoughts karena sudah langsung dianggap depresi)
criteria = {
    "academic_pressure": {"type": "cost", "weight": 0.25},
    "cgpa": {"type": "benefit", "weight": 0.15},
    "study_satisfaction": {"type": "benefit", "weight": 0.15},
    "sleep_duration": {"type": "benefit", "weight": 0.2},
    "study_hours": {"type": "cost", "weight": 0.15},
    "financial_stress": {"type": "cost", "weight": 0.10}
}

# Konversi kolom kriteria ke float
for col in criteria.keys():
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Buang baris dengan nilai kosong
data = data.dropna(subset=criteria.keys())

# Ambil 20 data acak
data_sample = data.sample(n=20, random_state=42).reset_index(drop=True)

# Normalisasi bobot
total_weight = sum(c["weight"] for c in criteria.values())
for c in criteria:
    criteria[c]["weight"] /= total_weight

# Alternatif nama
alternatives = [f"A{i+1}" for i in range(len(data_sample))]
data_sample.index = alternatives

# Tampilkan data
st.header("20 Alternatif Mahasiswa yang Diambil Secara Acak")
st.dataframe(data_sample)

# --- Hitung Vector S (manual loop mirip contoh rumah) ---
st.header("Hitung Vector S")

vector_s = []
for idx, row in data_sample.iterrows():
    s_val = 1
    for k, info in criteria.items():
        val = row[k]
        w = info["weight"]
        
        # Hindari nol dengan ganti ke nilai kecil
        if val == 0:
            val = 0.0001
        
        if info["type"] == "cost":
            s_val *= val ** (-w)
        else:
            s_val *= val ** w
    vector_s.append(s_val)


s_df = pd.DataFrame(vector_s, index=alternatives, columns=["Vektor S"])
st.write("Tabel Perhitungan Vektor S:")
st.dataframe(s_df)

# --- Hitung Vector V (manual pembagian mirip contoh rumah) ---
st.header("Hitung Vector V")

total_s = sum(vector_s)
vector_v = [s / total_s for s in vector_s]

v_df = pd.DataFrame(vector_v, index=alternatives, columns=["Vektor V"])
st.write("Tabel Perhitungan Vektor V:")
st.dataframe(v_df)

# Gabungkan hasil
result_df = data_sample.copy()
result_df["Vektor S"] = vector_s
result_df["Vektor V"] = vector_v

# Urutkan hasil
result_df = result_df.sort_values(by="Vektor V", ascending=False)

st.header("Ranking Mahasiswa Berdasarkan WP")
st.dataframe(result_df)

# Visualisasi
st.header("Visualisasi Ranking Mahasiswa")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(result_df.index, result_df["Vektor V"], color="tomato")
ax.invert_yaxis()
ax.set_xlabel("Vektor V")
ax.set_title("Ranking Mahasiswa Paling Rawan Depresi")
st.pyplot(fig)
