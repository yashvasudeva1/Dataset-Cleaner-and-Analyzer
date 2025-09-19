import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import io
import altair as alt
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title(":material/folder: Dataset Cleaner and Analyser")
st.write("This app helps in making your dataset cleaner, outlier-free, and ready for training")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def calculate_outliers(df_numeric):
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))
    outliers_count = outlier_mask.sum()
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound, outlier_mask, outliers_count
    
def clean_outliers(df, target_col):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=np.number).columns.drop(target_col, errors='ignore')
    for col in numeric_cols:
        lower_bound = df_clean[col].quantile(0.25) - 1.5 * (df_clean[col].quantile(0.75) - df_clean[col].quantile(0.25))
        upper_bound = df_clean[col].quantile(0.75) + 1.5 * (df_clean[col].quantile(0.75) - df_clean[col].quantile(0.25))
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def prepare_data(df, target_col, test_size=0.3, random_state=42):
    df_clean = clean_outliers(df, target_col)
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, df_clean
