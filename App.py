import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
st.title("Dataset Cleaner and Analyser")
st.write("This app helps you in making your dataset cleaner, outlier free and ready for training")
file=st.file_uploader("Upload Your CSV File")
df=pd.read_csv(file)
df_num=df.select_dtypes(include='number')
column=df.columns
st.write(df.describe())
tab1,tab2,tab3,tab4=st.tabs(["Visual Representation","General Analysis", "Outlier Abalysis","Make Predictions"])
if tab1:
  for i in column:
    st.line_chart(df)
