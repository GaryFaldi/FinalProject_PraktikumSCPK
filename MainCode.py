import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.set_page_config(page_title="Decision Support System", layout="wide")
st.title("Decision Support System: Detecting College Students Most Prone to Depression (WP Method)")

display_columns = {
    "academic_pressure": "Academic Pressure",
    "cgpa": "CGPA/IPK",
    "study_satisfaction": "Study Satisfaction",
    "sleep_duration": "Sleep Duration",
    "study_hours": "Study Hours",
    "financial_stress": "Financial Stress",
    "suicidal_thoughts": "Suicidal Thoughts"
}

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
    "academic_pressure": {"type": "cost", "weight": 0.2},
    "cgpa": {"type": "benefit", "weight": 0.15},
    "study_satisfaction": {"type": "benefit", "weight": 0.15},
    "sleep_duration": {"type": "benefit", "weight": 0.2},
    "study_hours": {"type": "cost", "weight": 0.15},
    "financial_stress": {"type": "cost", "weight": 0.15}
}

# Konversi kolom kriteria ke float
for col in criteria.keys():
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Buang baris dengan nilai kosong
data = data.dropna(subset=criteria.keys())

# Ambil 20 data acak
# data = data.sample(n=20, random_state=42).reset_index(drop=True)

# Normalisasi bobot
total_weight = sum(c["weight"] for c in criteria.values())
for c in criteria:
    criteria[c]["weight"] /= total_weight

# Alternatif nama
alternatives = [f"A{i+1}" for i in range(len(data))]
data.index = alternatives

# Tampilkan data
st.header("Alternative Data")
st.dataframe(data.rename(columns=display_columns))


st.header("Calculate Vector S")

vector_s = []
for idx, row in data.iterrows():
    s_val = 1
    for k, info in criteria.items():
        val = row[k]
        w = info["weight"]
        
        val = max(val, 1)
        
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
# st.write(total_s)
vector_v = [s / total_s for s in vector_s]

v_df = pd.DataFrame(vector_v, index=alternatives, columns=["Vector V"])
st.write("V Vector Calculation Table:")
st.dataframe(v_df)

# Gabungkan hasil
result_df = data.copy()
result_df["Vector S"] = vector_s
result_df["Vector V"] = vector_v

# Urutkan hasil
result_df = result_df.sort_values(by="Vector V", ascending=False).head(20)

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