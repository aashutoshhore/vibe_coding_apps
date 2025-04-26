import streamlit as st
import pandas as pd
from pii_detector import predict_pii

st.set_page_config(page_title="PII Detector from CSV", layout="wide")
st.title("🔐 PII Detector from CSV")
st.markdown("Upload a CSV file and detect which columns may contain PII using both **column names** and **sample values**.")

st.sidebar.header("🔧 PII Detection Settings")
name_threshold = st.sidebar.slider("Name Score Threshold", 0.0, 1.0, 0.6, 0.01)
sample_threshold = st.sidebar.slider("Sample Score Threshold", 0.0, 1.0, 0.6, 0.01)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Prepare column metadata
    columns = []
    for col in df.columns:
        samples = df[col].dropna().astype(str).unique().tolist()[:5]  # max 5 unique samples
        dtype = str(df[col].dtype)
        inferred_type = "TEXT" if "object" in dtype else ("FLOAT" if "float" in dtype else "INTEGER")
        
        columns.append({
            "name": col,
            "type": inferred_type,
            "samples": samples
        })

    if st.button("🔍 Detect PII"):
        results = predict_pii(columns, name_score_threshold=name_threshold, sample_score_threshold=sample_threshold)
        st.subheader("🧠 Detection Results")
        st.dataframe(results)
