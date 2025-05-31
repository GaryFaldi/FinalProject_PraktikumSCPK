import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.set_page_config(page_title="Decision Support System", layout="wide")
st.title("Decision Support System: Detecting College Students Most Prone to Depression (WP Method)")
st.sidebar.header("Input Weight Criteria")
w1 = st.sidebar.slider("Academic Pressure", 0.0, 1.0, 0.25)
w2 = st.sidebar.slider("CGPA/IPK", 0.0, 1.0, 0.25)
w3 = st.sidebar.slider("Study Satisfaction", 0.0, 1.0, 0.25)
w4 = st.sidebar.slider("Sleep Duration", 0.0, 1.0, 0.25)
w5 = st.sidebar.slider("Study Hours", 0.0, 1.0, 0.25)
w6 = st.sidebar.slider("Financial Stress", 0.0, 1.0, 0.25)

page = st.sidebar.selectbox("Select Page", [
    "Calculate Available Datasets", 
    "Input Your Own Data", 
])

display_columns = {
    "academic_pressure": "Academic Pressure",
    "cgpa": "CGPA/IPK",
    "study_satisfaction": "Study Satisfaction",
    "sleep_duration": "Sleep Duration",
    "study_hours": "Study Hours",
    "financial_stress": "Financial Stress",
    "suicidal_thoughts": "Suicidal Thoughts"
}


if page == "Calculate Available Datasets":
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

    criteria = {
        "academic_pressure": {"type": "benefit", "weight": w1},
        "cgpa": {"type": "cost", "weight": w2},
        "study_satisfaction": {"type": "cost", "weight": w3},
        "sleep_duration": {"type": "cost", "weight": w4},
        "study_hours": {"type": "benefit", "weight": w5},
        "financial_stress": {"type": "benefit", "weight": w6}
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
    st.header("20 Random Student Alternative Data")
    st.dataframe(data_sample.rename(columns=display_columns))


    st.header("Calculate Vector S")

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


    s_df = pd.DataFrame(vector_s, index=alternatives, columns=["Vector S"])
    st.write("S Vector Calculation Table:")
    st.dataframe(s_df)

    st.header("Calculate Vector V")

    total_s = sum(vector_s)
    vector_v = [s / total_s for s in vector_s]

    v_df = pd.DataFrame(vector_v, index=alternatives, columns=["Vector V"])
    st.write("V Vector Calculation Table:")
    st.dataframe(v_df)

    # Gabungkan hasil
    result_df = data_sample.copy()
    result_df["Vector S"] = vector_s
    result_df["Vector V"] = vector_v

    # Urutkan hasil
    result_df = result_df.sort_values(by="Vector V", ascending=False)

    st.header("College Student Ranking Based on WP")
    st.dataframe(result_df.rename(columns=display_columns))

    # Visualisasi
    st.header("Student Ranking Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(result_df.index, result_df["Vector V"], color="skyblue")
    ax.invert_yaxis()
    ax.set_xlabel("Vector V")
    ax.set_title("Ranking of Students Most Prone to Depression")
    st.pyplot(fig)


elif page == "Input Your Own Data":
    st.header("Input Your Own Data")

    num_data = st.number_input("How many students do you want to input?", min_value=2, max_value=20, value=2, step=1)

    input_data = []

    for i in range(int(num_data)):
        st.subheader(f"Student {i+1}")
        academic_pressure = st.slider(f"Academic Pressure (Student {i+1})", 1.0, 5.0, 3.0)
        cgpa = st.slider(f"CGPA/IPK (Student {i+1})", 1.0, 10.0, 2.5)
        study_satisfaction = st.slider(f"Study Satisfaction (Student {i+1})", 1.0, 5.0, 3.0)
        sleep_duration = st.slider(f"Sleep Duration (Student {i+1}, in hours)", 4.0, 10.0, 7.0)
        study_hours = st.slider(f"Study Hours (Student {i+1})", 1.0, 12.0, 5.0)
        financial_stress = st.slider(f"Financial Stress (Student {i+1})", 1.0, 5.0, 3.0)

        input_data.append({
            "academic_pressure": academic_pressure,
            "cgpa": cgpa,
            "study_satisfaction": study_satisfaction,
            "sleep_duration": sleep_duration,
            "study_hours": study_hours,
            "financial_stress": financial_stress
        })
        st.markdown("---")

    input_df = pd.DataFrame(input_data)
    input_df.index = [f"A{i+1}" for i in range(len(input_df))]

    st.subheader("Your Input Data")
    st.dataframe(input_df.rename(columns=display_columns))

    # Definisi kriteria dan bobot
    criteria = {
        "academic_pressure": {"type": "benefit", "weight": w1},
        "cgpa": {"type": "cost", "weight": w2},
        "study_satisfaction": {"type": "cost", "weight": w3},
        "sleep_duration": {"type": "cost", "weight": w4},
        "study_hours": {"type": "benefit", "weight": w5},
        "financial_stress": {"type": "benefit", "weight": w6}
    }

    # Normalisasi bobot
    total_weight = sum(c["weight"] for c in criteria.values())
    for c in criteria:
        criteria[c]["weight"] /= total_weight

    # Hitung Vector S
    vector_s = []
    for idx, row in input_df.iterrows():
        s_val = 1
        for k, info in criteria.items():
            val = row[k]
            if val == 0:
                val = 0.0001
            if info["type"] == "cost":
                s_val *= val ** (-info["weight"])
            else:
                s_val *= val ** info["weight"]
        vector_s.append(s_val)

    # Tampilkan tabel Vector S
    s_df = pd.DataFrame(vector_s, index=input_df.index, columns=["Vector S"])
    st.subheader("S Vector Calculation Table")
    st.dataframe(s_df)

    # Hitung Vector V
    total_s = sum(vector_s)
    vector_v = [s / total_s for s in vector_s]

    v_df = pd.DataFrame(vector_v, index=input_df.index, columns=["Vector V"])
    st.subheader("V Vector Calculation Table")
    st.dataframe(v_df)

    # Gabungkan semua ke dalam 1 tabel hasil akhir
    result_df = input_df.copy()
    result_df["Vector S"] = vector_s
    result_df["Vector V"] = vector_v

    # Urutkan
    result_df = result_df.sort_values(by="Vector V", ascending=False)

    st.subheader("Student Ranking Based on WP")
    st.dataframe(result_df.rename(columns=display_columns))

    # Visualisasi Ranking
    st.subheader("Ranking Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(result_df.index, result_df["Vector V"], color="lightgreen")
    ax.invert_yaxis()
    ax.set_xlabel("Vector V")
    ax.set_title("Ranking of Custom Students Most Prone to Depression")
    st.pyplot(fig)
