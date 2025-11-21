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
import pickle
import sys
sys.path.append("backend functions/functionalities")
sys.path.append("backend functions/classification models")
sys.path.append("backend functions/regression models")

st.set_page_config(page_title="QuickML", layout="wide")
logo_path = "logo.png"
st.markdown(
    """
    <div style='display: flex; align-items: center;'>
    <span style="font-size:110px; font-weight:bold; font-style: italic; color:#fff; font-family: Arial, Helvetica, sans-serif;">QuickML</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.subheader("An app that enables you to clean, analyze & visualize your dataset and make predictions based on your preferred ML model")
st.divider()
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

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

if not df.empty:

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview","Visualization","Cleaning","Normality","Prediction","AI Assistant"])

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
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        
        if len(numeric_columns) >= 2:
            col1, col2 = st.columns(2)
            x_col = col1.selectbox("X Axis", numeric_columns, index=0)
            y_col = col2.selectbox("Y Axis", numeric_columns, index=1)

            plot_df = df.copy()
            if len(plot_df) > 5000:
                st.warning("Dataset > 5000 rows. Plotting a random sample of 5000 points for performance.")
                plot_df = plot_df.sample(n=5000, random_state=42)
            chart = (
                alt.Chart(plot_df)
                .mark_line()
                .encode(
                    x=x_col,
                    y=y_col,
                    tooltip=[x_col, y_col]
                )
                .interactive()
                .properties(height=400)
            )
            st.altair_chart(chart, width='stretch')
    with tab3:
        from countsofnullduplicateandoutlier import total_null,total_outliers,total_duplicates
        from handlenullduplicateoutlier import handle_null_and_duplicates_and_outliers
        current_df = st.session_state.get("df", df)
        before_nulls = total_null(current_df)["count"].sum()
        before_outliers = total_outliers(current_df)[0].sum()
        before_duplicates = total_duplicates(current_df)
        summary_df = pd.DataFrame({
            "Metric": ["Total Null Values","Total Outliers","Total Duplicates"],
            "Count": [before_nulls,before_outliers,before_duplicates]
        })
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Report Before Cleaning")
            st.dataframe(summary_df)
            if st.button("Clean Data"):
                cleaned_df = handle_null_and_duplicates_and_outliers(current_df)
                st.session_state["df"] = cleaned_df
                after_nulls = total_null(cleaned_df)["count"].sum()
                after_outliers = total_outliers(cleaned_df)[0].sum()
                after_duplicates = total_duplicates(cleaned_df)
                st.session_state["after_df"] = pd.DataFrame({
                    "Metric": ["Total Null Values","Total Outliers","Total Duplicates"],
                    "Count": [after_nulls,after_outliers,after_duplicates]
                })
                st.session_state["clean_preview"] = cleaned_df.head()
                st.rerun()
        with col2:
            st.subheader("Report After Cleaning")
            if "after_df" in st.session_state:
                st.dataframe(st.session_state["after_df"])
            else:
                st.info("Click Clean Data to generate the report.")
        if "clean_preview" in st.session_state:
            st.success("Dataset Cleaned Successfully!")
            st.write("### Preview of Cleaned Data")
            st.dataframe(st.session_state["clean_preview"])
            if "df" in st.session_state:
                cleaned_df = st.session_state["df"]
                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv"
                )
    with tab4:
        st.title("Normality Check")
        from typeofdata import analyze_distribution
        current_df = st.session_state.get("df", df)
        result_df = analyze_distribution(current_df)
        st.dataframe(result_df, use_container_width=True)
        st.write("### Histogram Preview")
        numeric_cols = current_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            st.info("No numeric columns found for histogram.")
        else:
            selected_hist = st.selectbox("Select Column", numeric_cols)
            chart = (
                alt.Chart(current_df)
                .mark_bar()
                .encode(
                    x=alt.X(selected_hist, bin=alt.Bin(maxbins=30)),
                    y='count()'
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
    with tab5:
        st.title("Prediction")
        current_df = st.session_state.get("df", df)
        target_column = st.selectbox("Select Target Column", current_df.columns)
        if target_column:
            y = current_df[target_column]
            if y.dtype in ["int64", "float64"]:
                problem_type = "Regression"
            else:
                problem_type = "Classification"
            st.subheader("Problem Type Detected")
            st.success(f"This is a **{problem_type}** problem.")
            from traintestsplit import create_train_test_split
            from preprocessdata import preprocess_data
            X_train, X_test, y_train, y_test = create_train_test_split(
                current_df, target_column, test_size=0.2
            )
            X_train_prep, X_test_prep, y_train_prep, y_test_prep, encoders, scaler = preprocess_data(
                X_train, X_test, y_train, y_test
            )
            if problem_type == "Regression":
                from linearregression import linear_regression_model
                from ridgeregression import ridge_regression_model
                from lassoregression import lasso_regression_model
                from elasticnetregression import elasticnet_regression_model
                from decisiontreeregression import decision_tree_regression_model
                from randomforestregression import random_forest_regression_model
                from gradientboostregression import gradient_boosting_regression_model
                from adaboostregression import adaboost_regression_model
                from knnregression import knn_regression_model
                from svrregression import svr_regression_model
                model_options = [
                    "Linear Regression",
                    "Ridge Regression",
                    "Lasso Regression",
                    "ElasticNet Regression",
                    "Decision Tree Regressor",
                    "Random Forest Regressor",
                    "Gradient Boosting Regressor",
                    "AdaBoost Regressor",
                    "KNN Regressor",
                    "SVR Regressor"
                ]
                model_map = {
                    "Linear Regression": linear_regression_model,
                    "Ridge Regression": ridge_regression_model,
                    "Lasso Regression": lasso_regression_model,
                    "ElasticNet Regression": elasticnet_regression_model,
                    "Decision Tree Regressor": decision_tree_regression_model,
                    "Random Forest Regressor": random_forest_regression_model,
                    "Gradient Boosting Regressor": gradient_boosting_regression_model,
                    "AdaBoost Regressor": adaboost_regression_model,
                    "KNN Regressor": knn_regression_model,
                    "SVR Regressor": svr_regression_model
                }
            else:
                from logisticregression import tune_logistic_regression
                from decisiontree import tune_decision_tree
                from randomforest import tune_random_forest
                from gradientboosting import tune_gradient_boosting
                from adaboost import tune_adaboost
                from knn import tune_knn
                from svm import tune_svm
                from naivebayes import tune_naive_bayes
                from mlp import tune_mlp
                model_options = [
                    "Logistic Regression",
                    "Decision Tree Classifier",
                    "Random Forest Classifier",
                    "Gradient Boosting Classifier",
                    "AdaBoost Classifier",
                    "KNN Classifier",
                    "SVM Classifier",
                    "Naive Bayes",
                    "Neural Network (MLP)"
                ]
                model_map = {
                    "Logistic Regression": tune_logistic_regression,
                    "Decision Tree Classifier": tune_decision_tree,
                    "Random Forest Classifier": tune_random_forest,
                    "Gradient Boosting Classifier": tune_gradient_boosting,
                    "AdaBoost Classifier": tune_adaboost,
                    "KNN Classifier": tune_knn,
                    "SVM Classifier": tune_svm,
                    "Naive Bayes": tune_naive_bayes,
                    "Neural Network (MLP)": tune_mlp
                }
            selected_model_name = st.selectbox("Select Model", model_options)
            model_function = model_map[selected_model_name]
            if st.button("Train Model"):
                model, metrics_df = model_function(
                    X_train_prep, y_train_prep, X_test_prep, y_test_prep
                )
                st.success("Model Trained Successfully!")
                with st.sidebar:
                    st.subheader(f"{selected_model_name} — Metrics")

                for key, value in metrics.items():
                    st.metric(label=key, value=value)
                st.dataframe(metrics_df, use_container_width=True)

    


   



