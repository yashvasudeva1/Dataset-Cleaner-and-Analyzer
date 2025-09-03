import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    tab0, tab1, tab2, tab3, tab4 , tab5 = st.tabs(
        ["General Analysis", "Visual Representation", "Facts", "Outlier Analysis", "Make Predictions","Check the Type of Distribution"]
    )
    data = np.random.normal(0, 1, size=100)  # Use your real column here

    fig, ax = plt.subplots()
    ax.hist(data, bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Histogram Example')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    st.pyplot(fig)
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
            outlier_report.append({"Column Name": col, "Number of Outliers": n_outliers})
        outliers = pd.DataFrame(outlier_report)
        st.write(outliers)
        st.button("Remove the Outliers")
    with tab5:
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            st.subheader(f"Histogram and KDE for '{col}'")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, bins=20, color="skyblue", ax=ax)
            ax.set_title(f"Histogram and KDE of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close(fig)
