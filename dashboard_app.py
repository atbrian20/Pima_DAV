"""
Bonus Task: Interactive Dashboard
Pima Indians Diabetes Dataset — Streamlit App

Run locally with:  streamlit run dashboard_app.py
Or deploy for free on Streamlit Community Cloud (https://streamlit.io/cloud).

Functionality: lets the user filter patients by Outcome and Age range, then
dynamically explore three Part-C visualizations (histogram, boxplot, and a
correlation heatmap) that update live based on the selected filters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Pima Diabetes Explorer", layout="wide")
sns.set_theme(style="whitegrid")

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_clean.csv")
    return df

df = load_data()

st.title("Pima Indians Diabetes — Interactive Explorer")
st.caption("Dashboard for the Data Analytics & Visualization take-home exam (DS17).")

# --- Sidebar controls (filters) ---
st.sidebar.header("Filters")

outcome_filter = st.sidebar.selectbox(
    "Outcome",
    options=["All", "Diabetic", "Non-Diabetic"],
    index=0,
)

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

numeric_col = st.sidebar.selectbox(
    "Numerical variable to explore",
    options=["Glucose", "BMI", "Insulin", "BloodPressure", "SkinThickness",
             "Pregnancies", "DiabetesPedigreeFunction", "Age"],
    index=0,
)

# --- Apply filters ---
filtered = df.copy()
if outcome_filter != "All":
    filtered = filtered[filtered["Outcome_Label"] == outcome_filter]
filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]

st.markdown(f"**{len(filtered)} patients** match the current filters (out of {len(df)} total).")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Distribution of {numeric_col}")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered[numeric_col], bins=25, kde=True, ax=ax, color="#3b6ea5")
    ax.set_xlabel(numeric_col)
    st.pyplot(fig)

with col2:
    st.subheader(f"{numeric_col} by Outcome")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=filtered, x="Outcome_Label", y=numeric_col, hue="Outcome_Label",
                legend=False, ax=ax)
    ax.set_xlabel("Outcome")
    st.pyplot(fig)

st.subheader("Correlation Heatmap (filtered subset)")
num_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
fig, ax = plt.subplots(figsize=(9, 6))
corr = filtered[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)

st.markdown("---")
st.caption("Data source: UCI ML Repository / Kaggle — Pima Indians Diabetes Database.")
