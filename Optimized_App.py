import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
#Basic EDA Libraries 
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
#Different ML Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
#Model Evaluation Metrics
def shapiro_safe(x):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*computed p-value may not be accurate.*")
        return stats.shapiro(x)

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
    return outlier_mask, outliers_count

def generate_report(df, actions):
    report_df = pd.DataFrame(index=df.columns)
    if "NaN Values" in actions:
        report_df["NaN Values"] = df.isnull().sum()
    if "Duplicates" in actions:
        dup_mask = df.duplicated(keep=False)
        dup_counts = df.loc[dup_mask].count()
        report_df["Duplicates"] = dup_counts
    if "Outliers" in actions:
        numerics = df.select_dtypes(include=np.number)
        _, outliers_count = calculate_outliers(numerics)
        report_df["Outliers"] = np.nan
        for col in outliers_count.index:
            report_df.at[col, "Outliers"] = outliers_count[col]
    return report_df.fillna('-').astype(str)

file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if file is not None:
    df = load_data(file)
    if "clean_df" not in st.session_state:
        st.session_state["clean_df"] = df.copy()
    st.write("Preview of your dataset:")
    st.dataframe(st.session_state["clean_df"], use_container_width=True)

    tab_analysis, tab_visual, tab_chat, tab_clean, tab_predictor, tab_distribution = st.tabs(
        ["Analysis", "Visualisation", "Chat", "Cleaning", "Predictor", "Distribution"]
    )

    with tab_analysis:
        st.write(st.session_state["clean_df"].describe())

    with tab_visual:
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        selected_two = st.multiselect(
            "Select exactly two columns to plot one against the other", numeric_columns
        )
        if selected_two:
            if len(selected_two) == 2:
                x_col, y_col = selected_two
                chart = (
                    alt.Chart(df)
                    .mark_line()
                    .encode(
                        x=alt.X(x_col, title=x_col),
                        y=alt.Y(y_col, title=y_col),
                    )
                    .properties(
                        title=f"Line plot of {y_col} vs {x_col}",
                        width=600,
                        height=300
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("Please select exactly two columns for this plot.")
        else:
            st.info("Please select at least one column to display the chart.")

    with tab_clean:
    # Use the latest cleaned dataframe if available, else original
        df = st.session_state.get("cleaned_df", st.session_state["clean_df"])
        
        actions = st.multiselect("Select Actions:", ["NaN Values", "Duplicates", "Outliers"])
        
        # Report before cleaning
        report_before = generate_report(df, actions)
        st.write("### Report Before Cleaning")
        st.dataframe(report_before)
        
        if st.button("Clean"):
            cleaned = df.copy()
    
            if "Duplicates" in actions:
                cleaned = cleaned.drop_duplicates()
            if "Outliers" in actions:
                numerics_cleaned = cleaned.select_dtypes(include=np.number)
                outlier_mask, _ = calculate_outliers(numerics_cleaned)
                keep_mask = ~outlier_mask.any(axis=1)
                cleaned = cleaned.loc[keep_mask]
            if "NaN Values" in actions:
                cleaned = cleaned.dropna()
            
            st.session_state["cleaned_df"] = cleaned  # mark that cleaning is done
        
        # Only show cleaned data and report if cleaned_df is in session state (clean button pressed once)
        if "cleaned_df" in st.session_state:
            cleaned_latest = st.session_state["cleaned_df"]
            report_after = generate_report(cleaned_latest, actions)
    
            st.write("### Report After Cleaning")
            st.dataframe(report_after)
    
            st.write("### Cleaned Data")
            st.dataframe(cleaned_latest)
    
            csv_string = cleaned_latest.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Data",
                data=csv_string,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

