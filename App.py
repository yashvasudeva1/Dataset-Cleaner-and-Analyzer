import streamlit as st
import numpy as np
import pandas as pd
st.title("Dataset Cleaner and Analyser")
st.write("This app helps you in making your dataset cleaner, outlier free and ready for training")
file=st.file_uploader("Upload Your CSV File")
if file is not None:
    df = pd.read_csv(file)
    column=df.columns
    st.write(df)
    tab0,tab1,tab2,tab3,tab4=st.tabs(["General Analysis","Visual Representation","Facts", "Outlier Abalysis","Make Predictions"])
    if tab0:
        st.write(df.describe())
    if tab1:
        st.line_graph(df)
