import streamlit as st
import numpy as np
import pandas as pd

st.title("Dataset Cleaner and Analyser")
st.write("This app helps you in making your dataset cleaner, outlier free and ready for training")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

file = st.file_uploader("Upload Your CSV File")
if file is not None:
    df = load_data(file)
    df_num = df.select_dtypes(include=[np.number])
    columns = df.columns
    st.write("Preview of your dataset:")
    st.dataframe(df, use_container_width=True)

    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        ["General Analysis", "Visual Representation", "Facts", "Outlier Analysis", "Make Predictions"]
    )

    with tab0:
        st.write(df.describe())

    with tab1:
        st.line_chart(df.select_dtypes(include='number'), height=250, use_container_width=True)
    with tab3:
        columns = df.select_dtypes(include=[np.number]).columns
        outlier_report = []
        for col in columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_report.append({"column": col, "num_outliers": n_outliers})
        outliers = pd.DataFrame(outlier_report)
        st.write(outliers)
