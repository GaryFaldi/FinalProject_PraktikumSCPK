import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ======= KONFIGURASI HALAMAN STREAMLIT =======
st.set_page_config(page_title="Deteksi Mahasiswa Rawan Depresi", layout="wide", page_icon="🧠")

# ======= NAVIGASI SIDEBAR =======
menu = st.sidebar.radio("Navigasi", [
    "📘 Deskripsi", 
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
criteria = {
    "academic_pressure": {"type": "cost", "weight": 0.2},
    "cgpa": {"type": "benefit", "weight": 0.15},
    "study_satisfaction": {"type": "benefit", "weight": 0.15},
    "sleep_duration": {"type": "benefit", "weight": 0.2},
    "study_hours": {"type": "cost", "weight": 0.15},
    "financial_stress": {"type": "cost", "weight": 0.15}
}

# Ubah semua kolom kriteria menjadi numerik dan hilangkan baris dengan nilai kosong
for col in criteria.keys():
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=criteria.keys())

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

# Format tampilan angka di UI
format_dict = {
    "academic_pressure": "{:.0f}",
    "cgpa": "{:.0f}",
    "study_satisfaction": "{:.0f}",
    "sleep_duration": "{:.0f}",
    "suicidal_thoughts": "{:.0f}",
    "study_hours": "{:.0f}",
    "financial_stress": "{:.0f}",
    "Age": "{:.0f}",
    "ID": "{:.0f}"
}

# ======= FUNGSI HITUNG VECTOR S METODE WP =======
@st.cache_data
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
        color: #2C3E50;
    }
    .sub-text {
        font-size: 18px;
        color: #34495E;
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
        - 🎓 **Academic Pressure** *(semakin tinggi tekanan, semakin rawan → cost)*
        - 🧮 **CGPA/IPK** *(semakin tinggi, semakin baik → benefit)*
        - 🙂 **Study Satisfaction** *(semakin puas, semakin baik → benefit)*
        - 🛌 **Sleep Duration** *(semakin cukup tidur, semakin baik → benefit)*
        - 📚 **Study Hours** *(semakin tinggi jam belajar bisa meningkatkan tekanan → cost)*
        - 💰 **Financial Stress** *(semakin besar tekanan finansial, semakin rawan → cost)*
        """)

elif menu == "📊 Data Alternatif":
    # Tampilkan data alternatif mahasiswa
    st.subheader("📊 Data Mahasiswa")
    pd.set_option("styler.render.max_elements", 300000)

    # Rename kolom untuk tampil di UI
    df_renamed = data.rename(columns=display_columns)

    # Format angka sesuai tampilan
    format_dict_after_rename = {
        "Academic Pressure": "{:.0f}",
        "CGPA/IPK": "{:.0f}",
        "Study Satisfaction": "{:.0f}",
        "Sleep Duration": "{:.0f}",
        "Suicidal Thoughts": "{:.0f}",
        "Study Hours": "{:.0f}",
        "Financial Stress": "{:.0f}",
        "Age": "{:.0f}",
        "ID": "{:.0f}"
    }

    # Styling tabel agar lebih menarik
    styled_data = df_renamed.style.format(format_dict_after_rename).set_table_styles([
        {
            'selector': 'thead th',
            'props': [('background-color', '#3498DB'), ('color', 'white')]
        },
        {
            'selector': 'tbody tr:nth-child(even)',
            'props': [('background-color', '#f2f2f2')]
        }
    ])

    # Tampilkan dataframe di Streamlit
    st.dataframe(styled_data)


elif menu == "🧮 Perhitungan Vector S":
    # Tampilkan hasil perhitungan Vector S
    st.subheader("🧮 Perhitungan Nilai Vektor S")
    s_df = pd.DataFrame(vector_s, index=alternatives, columns=["Vector S"])
    st.dataframe(s_df.style.format("{:.6f}").set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#1ABC9C'), ('color', 'white')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#eafaf1')]}
    ]))

elif menu == "📈 Perhitungan Vector V & Ranking":
    # Hitung Vector V dari Vector S dan tampilkan ranking tertinggi
    st.subheader("📈 Perhitungan Nilai Vektor V dan Ranking")

    total_s = sum(vector_s)
    vector_v = [s / total_s for s in vector_s]
    v_df = pd.DataFrame(vector_v, index=alternatives, columns=["Vector V"])

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
    st.dataframe(
        styled_result.style
            .format(format_dict)
            .set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#9B59B6'), ('color', 'white')]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f8f0fb')]}
            ])
    )


elif menu == "🌟 Visualisasi":
    # Visualisasi bar chart ranking mahasiswa rawan depresi
    st.subheader("🌟 Visualisasi Ranking Mahasiswa")
    result_df = data.copy()
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
