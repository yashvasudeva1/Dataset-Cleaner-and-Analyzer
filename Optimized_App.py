import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import math
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import shapiro

st.set_page_config(page_title="QuickML", layout="wide")
st.title("QuickMl")
st.text("An app that enables you to clean, analyze & visualize your dataset and make predictions based on your preferred ML model")
st.divider()
uploaded_file=st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
@st.cache_data
def load_csv(uploaded_file):
    try:
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        else:
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    if not df.empty:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.session_state["df"] = df
    else:
        st.warning("Uploaded file could not be processed.")
else:
    st.info("Please upload a dataset to begin.")
    df = pd.DataFrame()

tab1,tab2,tab3,tab4,tab4,tab6=st.tabs(["Overview","Visualization","Cleaning","Normality","Prediction","AI Assistant"])
with tab1:
    st.title("Data Overview")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### First 5 Rows")
        st.dataframe(df.head())
    with col_b:
        st.write("### Data Types")
        st.dataframe(df.dtypes.astype(str), height=200)

    st.write("### Summary Statistics")
    st.dataframe(df.describe(include='all'))

with tab2:
    st.title("Bivariate Analysis")
    numeric_columns=df.select_dtypes(include='number').columns.tolist()
    if len(numeric_columns)>=2:
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("X Axis", numeric_columns, index=0)
        y_col = col2.selectbox("Y Axis", numeric_columns, index=1)
        # Downsample for Altair if data is huge to prevent crashes
        plot_df = df.copy()
        if len(plot_df)>5000:
            st.warning("Dataset > 5000 rows. Plotting a random sample of 5000 points for performance.")
            plot_df = plot_df.sample(n=5000, random_state=42)
        chart = (
            alt.Chart(plot_df)
            .mark_line()
            .encode(
                x=x_col,
                y=y_col,
                tooltip=[x_col, y_col]
        ).interactive().properties(height=400))
        st.altair_chart(chart, width='stretch')
