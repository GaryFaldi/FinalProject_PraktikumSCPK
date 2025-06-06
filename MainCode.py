import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ======= KONFIGURASI HALAMAN STREAMLIT =======
st.set_page_config(page_title="Deteksi Mahasiswa Rawan Depresi", layout="wide", page_icon="🧠")

# ======= NAVIGASI SIDEBAR =======
menu = st.sidebar.radio("Navigasi", [
    "📘 Deskripsi", 
    "📝 Masukkan Bobot",
    "📊 Data Alternatif",
    "🧮 Perhitungan Vector S", 
    "📈 Perhitungan Vector V & Ranking", 
    "🌟 Visualisasi"
])

# ======= LOAD DATA =======
data = pd.read_csv("CleanedDataset.csv")

# ======= PREPROCESSING DATA =======
# Rename kolom agar lebih mudah diproses
data = data.rename(columns={
    "Academic Pressure": "academic_pressure",
    "CGPA": "cgpa",
    "Study Satisfaction": "study_satisfaction",
    "Sleep Duration": "sleep_duration",
    "Have you ever had suicidal thoughts ?": "suicidal_thoughts",
    "Work/Study Hours": "study_hours",
    "Financial Stress": "financial_stress"
})

# Mapping data kategorikal suicidal thoughts ke numerik
data["suicidal_thoughts"] = data["suicidal_thoughts"].map({"Yes": 1, "No": 0})

# Mapping data sleep duration ke angka
data["sleep_duration"] = data["sleep_duration"].astype(str).str.replace("'", "").str.strip()
data["sleep_duration"] = data["sleep_duration"].map({
    "Less than 5 hours": 4,
    "5-6 hours": 5,
    "6-7 hours": 6,
    "7-8 hours": 7,
    "8-9 hours": 8,
    "More than 8 hours": 9
})

# ======= DEKLARASI KRITERIA DAN BOBOT =======
if "criteria" not in st.session_state:
    st.session_state.criteria = {
        "academic_pressure": {"type": "benefit", "weight": 0.2},
        "cgpa": {"type": "cost", "weight": 0.15},
        "study_satisfaction": {"type": "cost", "weight": 0.15},
        "sleep_duration": {"type": "cost", "weight": 0.2},
        "study_hours": {"type": "benefit", "weight": 0.15},
        "financial_stress": {"type": "benefit", "weight": 0.15}
    }
criteria = st.session_state.criteria

# Ubah semua kolom kriteria menjadi numerik dan hilangkan baris dengan nilai kosong
for col in criteria.keys():
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=criteria.keys())

# Simpan salinan data asli untuk tampilan UI
original_data = data.copy()

# ======= OPSI NORMALISASI =======
use_normalization = st.sidebar.checkbox("Gunakan normalisasi data", value=True)

if use_normalization:
    for k in criteria.keys():
        col = data[k]
        # Normalisasi Min-Max + epsilon agar tidak nol
        data[k] = 1 + 9 * ((col - col.min()) / (col.max() - col.min()))  # hasil normalisasi antara 1 hingga 10


# Normalisasi bobot agar total bobot = 1
total_weight = sum(c["weight"] for c in criteria.values())
for c in criteria:
    criteria[c]["weight"] /= total_weight

# Beri label alternatif berdasarkan index (A1, A2, ...)
alternatives = [f"A{i+1}" for i in range(len(data))]
data.index = alternatives

# Mapping kolom internal ke nama kolom tampilan untuk UI
display_columns = {
    "academic_pressure": "Academic Pressure",
    "cgpa": "CGPA/IPK",
    "study_satisfaction": "Study Satisfaction",
    "sleep_duration": "Sleep Duration",
    "study_hours": "Study Hours",
    "financial_stress": "Financial Stress",
    "suicidal_thoughts": "Suicidal Thoughts"
}

# # Format tampilan angka di UI
# format_dict = {
#     "academic_pressure": "{:.0f}",
#     "cgpa": "{:.0f}",
#     "study_satisfaction": "{:.0f}",
#     "sleep_duration": "{:.0f}",
#     "suicidal_thoughts": "{:.0f}",
#     "study_hours": "{:.0f}",
#     "financial_stress": "{:.0f}",
#     "Age": "{:.0f}",
#     "ID": "{:.0f}"
# }

# ======= FUNGSI HITUNG VECTOR S METODE WP =======
def hitung_vector_s(data, criteria):
    vector_s = []
    for idx, row in data.iterrows():
        s_val = 1
        for k, info in criteria.items():
            val = max(row[k], 1)  # Pastikan nilai minimal 1 agar pangkat valid
            w = info["weight"]
            # Jika cost, pangkatnya negatif; jika benefit positif
            s_val *= val ** (-w if info["type"] == "cost" else w)
        vector_s.append(s_val)
    return vector_s

vector_s = hitung_vector_s(data, criteria)

# ======= UI STREAMLIT BERDASARKAN MENU NAVIGASI =======

if menu == "📘 Deskripsi":
    # Tampilan halaman deskripsi aplikasi
    st.markdown("""
    <style>
    .big-title {
        font-size: 36px;
        font-weight: bold;
        color: #fffff;
    }
    .sub-text {
        font-size: 18px;
        color: #fffff;
    }
    </style>
    <div class='big-title'>🧠 Sistem Pendukung Keputusan Mahasiswa Paling Rawan Depresi</div>
    <div class='sub-text'>
    Aplikasi ini menganalisis data mahasiswa untuk menentukan siapa yang paling rawan mengalami depresi 
    menggunakan metode <strong>Weighted Product (WP)</strong>. Data alternatif didasarkan pada berbagai kriteria psikososial dan akademik.
    </div>
    """, unsafe_allow_html=True)

    # Penjelasan detail kriteria
    with st.expander("ℹ️ Penjelasan Kriteria"):
        st.markdown("""
        **Kriteria Penilaian WP:**
        - 🎓 **Academic Pressure** *(semakin tinggi tekanan, semakin rawan → benefit)*
        - 🧮 **CGPA/IPK** *(semakin tinggi, semakin baik → cost)*
        - 🙂 **Study Satisfaction** *(semakin puas, semakin baik → cost)*
        - 🛌 **Sleep Duration** *(semakin cukup tidur, semakin baik → cost)*
        - 📚 **Study Hours** *(semakin tinggi jam belajar bisa meningkatkan tekanan → benefit)*
        - 💰 **Financial Stress** *(semakin besar tekanan finansial, semakin rawan → benefit)*
        """)

elif menu == "📝 Masukkan Bobot":
    st.subheader("📝 Masukkan Bobot Kriteria")

    with st.form("form_bobot"):
        st.markdown("Masukkan bobot untuk masing-masing kriteria (total 100):")

        input_weights = {}
        for key, info in criteria.items():
            display_name = display_columns[key]
            default_val = int(info["weight"] * 100)
            input_weights[key] = st.number_input(f"{display_name} (%)", min_value=0, max_value=100, value=default_val, step=1)

        submitted = st.form_submit_button("Simpan Bobot")

    if submitted:
        total_input = sum(input_weights.values())

        if total_input == 0:
            st.error("Total bobot tidak boleh 0.")
        elif any(val < 1 for val in input_weights.values()):
            st.error("Setiap bobot minimal harus 1%.")  
        elif total_input != 100:
            st.warning(f"Total bobot saat ini: {total_input}. Harus berjumlah 100.")
        else:
            for k in criteria:
                criteria[k]["weight"] = input_weights[k] / 100  # Normalisasi bobot
            st.success("Bobot berhasil diperbarui!")

elif menu == "📊 Data Alternatif":
    # Tampilkan data alternatif mahasiswa
    st.subheader("📊 Data Mahasiswa")
    pd.set_option("styler.render.max_elements", 300000)

    # Rename kolom untuk tampil di UI
    df_renamed = original_data.rename(columns=display_columns)

    # Tampilkan dataframe di Streamlit
    st.dataframe(df_renamed)


elif menu == "🧮 Perhitungan Vector S":
    # Tampilkan hasil perhitungan Vector S
    st.subheader("🧮 Perhitungan Nilai Vektor S")
    vector_s = hitung_vector_s(data, criteria)
    s_df = pd.DataFrame(vector_s, index=alternatives, columns=["Vector S"])
    st.dataframe(s_df.style.format("{:.6f}"))

elif menu == "📈 Perhitungan Vector V & Ranking":
    # Hitung Vector V dari Vector S dan tampilkan ranking tertinggi
    st.subheader("📈 Perhitungan Nilai Vektor V dan Ranking")
    vector_s = hitung_vector_s(data, criteria)
    total_s = sum(vector_s)
    vector_v = [s / total_s for s in vector_s]
    v_df = pd.DataFrame(vector_v, index=alternatives, columns=["Vector V"])
    st.dataframe(v_df)

    # Gabungkan hasil ke dataframe asli dan urutkan berdasarkan Vector V
    result_df = data.copy()
    result_df["Vector S"] = vector_s
    result_df["Vector V"] = vector_v
    result_df = result_df.sort_values(by="Vector V", ascending=False).head(20)

    # Rename kolom untuk UI
    styled_result = result_df.rename(columns=display_columns)
    numeric_cols = styled_result.select_dtypes(include=['float', 'int']).columns
    format_dict = {col: "{:.6f}" if "Vector" in col else "{:.0f}" for col in numeric_cols}

    # Tampilkan dataframe hasil ranking
    st.dataframe(styled_result.style.format(format_dict))

elif menu == "🌟 Visualisasi":
    # Visualisasi bar chart ranking mahasiswa rawan depresi
    st.subheader("🌟 Visualisasi Ranking Mahasiswa")
    result_df = data.copy()
    vector_s = hitung_vector_s(data, criteria)
    result_df["Vector S"] = vector_s
    result_df["Vector V"] = [s / sum(vector_s) for s in vector_s]
    result_df = result_df.sort_values(by="Vector V", ascending=False).head(20)

    fig = px.bar(result_df, x=result_df.index, y="Vector V", color="Vector V", text="Vector V",
                 title="Top 20 Mahasiswa Paling Rawan Depresi", color_continuous_scale="Reds")
    fig.update_layout(xaxis_title="Alternatif Mahasiswa", yaxis_title="Nilai Vektor V")
    st.plotly_chart(fig, use_container_width=True)

# ======= FOOTER =======
st.markdown("""
---
<div style='text-align:center;'>
    Dibuat dengan semangat peduli kesehatan mental mahasiswa | Metode: <strong>Weighted Product (WP)</strong> | 📚 Studi Kasus: <em>Deteksi Rawan Depresi</em>
</div>
""", unsafe_allow_html=True)
