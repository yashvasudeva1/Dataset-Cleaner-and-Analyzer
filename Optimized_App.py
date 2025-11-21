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
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from scipy.stats import shapiro
import pickle
import sys

sys.path.append("backend functions/functionalities")
sys.path.append("backend functions/classification models")
sys.path.append("backend functions/regression models")
def sanitize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(":", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    return df
st.set_page_config(page_title="QuickML", layout="wide")
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
        df = sanitize_columns(df)  # FIX: clean columns globally
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.session_state["df"] = df
    else:
        st.warning("Uploaded file could not be processed.")
else:
    st.info("Please upload a dataset to begin.")
    df = pd.DataFrame()
if not df.empty:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Visualization", "Cleaning", "Normality", "Prediction", "AI Assistant"
    ])
    with tab1:
        st.title("Data Overview")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("### First 5 Rows")
            st.dataframe(df.head(), width="stretch")
        with col_b:
            st.write("### Data Types")
            st.dataframe(df.dtypes.astype(str), height=200, width="stretch")
        st.write("### Summary Statistics")
        st.dataframe(df.describe(include='all'), width="stretch")
    with tab2:
        st.title("Bivariate Analysis")
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_columns) >= 2:
            col1, col2 = st.columns(2)
            x_col = col1.selectbox("X Axis", numeric_columns, index=0)
            y_col = col2.selectbox("Y Axis", numeric_columns, index=1)
            plot_df = df.copy()
            if len(plot_df) > 5000:
                st.warning("Dataset > 5000 rows. Using random sample of 5000 for performance.")
                plot_df = plot_df.sample(n=5000, random_state=42)
            clean_plot_df = sanitize_columns(plot_df)
            chart = (
                alt.Chart(clean_plot_df)
                .mark_circle(size=40)
                .encode(
                    x=alt.X(x_col, type="quantitative"),
                    y=alt.Y(y_col, type="quantitative")
                )
                .interactive()
            )
            st.altair_chart(chart, width='stretch')
    with tab3:
        from countsofnullduplicateandoutlier import total_null, total_outliers, total_duplicates
        from handlenullduplicateoutlier import handle_null_and_duplicates_and_outliers
        current_df = st.session_state.get("df", df)
        before_nulls = total_null(current_df)["count"].sum()
        before_outliers = total_outliers(current_df)[0].sum()
        before_duplicates = total_duplicates(current_df)
        summary_df = pd.DataFrame({
            "Metric": ["Total Null Values", "Total Outliers", "Total Duplicates"],
            "Count": [before_nulls, before_outliers, before_duplicates]
        })
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Report Before Cleaning")
            st.dataframe(summary_df, width="stretch")
            if st.button("Clean Data"):
                cleaned_df = handle_null_and_duplicates_and_outliers(current_df)
                cleaned_df = sanitize_columns(cleaned_df)  # FIX: sanitize after cleaning
                st.session_state["df"] = cleaned_df
                after_nulls = total_null(cleaned_df)["count"].sum()
                after_outliers = total_outliers(cleaned_df)[0].sum()
                after_duplicates = total_duplicates(cleaned_df)
                st.session_state["after_df"] = pd.DataFrame({
                    "Metric": ["Total Null Values", "Total Outliers", "Total Duplicates"],
                    "Count": [after_nulls, after_outliers, after_duplicates]
                })
                st.session_state["clean_preview"] = cleaned_df.head()
                st.rerun()
        with col2:
            st.subheader("Report After Cleaning")
            if "after_df" in st.session_state:
                st.dataframe(st.session_state["after_df"], width="stretch")
            else:
                st.info("Click Clean Data to generate the report.")
        if "clean_preview" in st.session_state:
            st.success("Dataset Cleaned Successfully!")
            st.write("### Preview of Cleaned Data")
            st.dataframe(st.session_state["clean_preview"], width="stretch")
            cleaned_df = st.session_state["df"]
            csv = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Cleaned Dataset", csv, "cleaned_dataset.csv", "text/csv")
    with tab4:
        st.title("Normality Check")
        from typeofdata import analyze_distribution
        current_df = st.session_state.get("df", df)
        result_df = analyze_distribution(current_df)
        st.dataframe(result_df, width="stretch")
        numeric_cols = current_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            selected_hist = st.selectbox("Select Column", numeric_cols)
            clean_hist_df = sanitize_columns(current_df)
            chart = (
                alt.Chart(clean_hist_df)
                .mark_bar()
                .encode(
                    x=alt.X(selected_hist, bin=alt.Bin(maxbins=30)),
                    y="count()"
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width='stretch')
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
            X_train_prep, X_test_prep, y_train_prep, y_test_prep, encoders, scaler, target_encoder = preprocess_data(X_train, X_test, y_train, y_test)
            st.session_state["target_encoder"] = target_encoder
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
           # ----------------------- TRAIN MODEL -----------------------
            if st.button("Train Model"):
                model, metrics_df = model_function(
                    X_train_prep, y_train_prep, X_test_prep, y_test_prep
                )
            
                st.success("Model Trained Successfully!")
            
                # Save to session_state so they persist after rerun
                st.session_state["trained_model"] = model
                st.session_state["trained_encoders"] = encoders
                st.session_state["trained_scaler"] = scaler
                st.session_state["trained_features"] = X_train.columns.tolist()
                st.session_state["target_type"] = problem_type
                st.session_state["target_column"] = target_column
            
                # ---- Clean Metrics ----
                metrics_clean = metrics_df.transpose().reset_index()
                metrics_clean.columns = ["Parameter", "Value"]
            
                # Save metrics for persistent sidebar
                st.session_state["model_metrics"] = metrics_clean
            
                # ---- Add Train/Test Accuracy for Classification ----
                if problem_type == "Classification":
                    train_pred = model.predict(X_train_prep)
                    test_pred = model.predict(X_test_prep)
            
                    train_acc = accuracy_score(y_train_prep, train_pred)
                    test_acc = accuracy_score(y_test_prep, test_pred)
            
                    acc_df = pd.DataFrame({
                        "Set": ["Training Accuracy", "Testing Accuracy"],
                        "Accuracy": [train_acc, test_acc]
                    })
            
                    st.session_state["accuracy_metrics"] = acc_df
                else:
                    st.session_state["accuracy_metrics"] = None
            
            
            # ----------------------- ALWAYS SHOW SIDEBAR METRICS -----------------------
            with st.sidebar:
                st.title("Metrics")
                if "model_metrics" in st.session_state:
                    st.dataframe(st.session_state["model_metrics"], width="stretch")
            
                    # Show accuracy only for classification
                    if st.session_state["accuracy_metrics"] is not None:
                        st.write("### Train vs Test Accuracy")
                        st.dataframe(st.session_state["accuracy_metrics"], width="stretch")
                else:
                    st.info("Train a model to view metrics.")
            
            
            # ----------------------- PREDICTION UI -----------------------
            if "trained_model" in st.session_state:

                st.write("## Make a Prediction")

                model = st.session_state["trained_model"]
                encoders = st.session_state["trained_encoders"]
                scaler = st.session_state["trained_scaler"]
                feature_columns = st.session_state["trained_features"]
                problem_type = st.session_state["target_type"]
                target_column = st.session_state["target_column"]

                st.write("### Enter Input Values:")

                user_input = {}
                cols = st.columns(3)
                col_idx = 0

                for col in feature_columns:
                    with cols[col_idx]:
                        if col in X_train.select_dtypes(include=["float64", "int64"]).columns:
                            user_input[col] = st.number_input(col, value=float(X_train[col].mean()))
                        else:
                            choices = current_df[col].dropna().unique().tolist()
                            user_input[col] = st.selectbox(col, choices)
                    col_idx = (col_idx + 1) % 3

                # Save original input (before encoding/scaling) for download & debugging
                input_df_original = pd.DataFrame([user_input])
                st.session_state["last_input_df"] = input_df_original.copy()

                # Prepare input for model (encode + scale)
                input_df = input_df_original.copy()

                # Apply encoders safely (map unseen -> unknown if encoder has that)
                for col, encoder in encoders.items():
                    if col in input_df.columns:
                        try:
                            # encoder expects 1D array; many of your encoders were fitted on Series
                            input_df[col] = encoder.transform(input_df[[col]])
                        except Exception as e:
                            # try map unseen values to '___unknown___' if available
                            if hasattr(encoder, "classes_") and "___unknown___" in encoder.classes_:
                                safe_val = input_df[col].map(lambda x: x if x in encoder.classes_ else "___unknown___")
                                input_df[col] = encoder.transform(safe_val)
                            else:
                                st.error(f"Encoding failed for column '{col}': {e}")
                                st.stop()

                # Scale numerical columns if scaler exists
                if scaler is not None:
                    # Only scale numeric cols that the scaler expects
                    numeric_cols = input_df.select_dtypes(include=["float64", "int64"]).columns
                    if len(numeric_cols) > 0:
                        try:
                            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
                        except Exception as e:
                            # If scaler expects additional columns or different order, align with training columns
                            try:
                                input_df = input_df[X_train_prep.columns]  # attempt align
                                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
                            except Exception as e2:
                                st.error(f"Scaler transform failed: {e} / {e2}")
                                st.stop()

                # Ensure column order matches model's training data (if possible)
                try:
                    input_df = input_df[X_train_prep.columns]
                except Exception:
                    # If X_train_prep not accessible here, just keep current order
                    pass

                # Predict button
                if st.button("Predict Value"):
                    try:
                        raw_pred = model.predict(input_df)[0]
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.stop()
                
                    # Decode classification prediction
                    if problem_type == "Classification":
                        target_encoder = st.session_state.get("target_encoder", None)
                        if target_encoder is not None:
                            try:
                                decoded_pred = target_encoder.inverse_transform([raw_pred])[0]
                            except Exception:
                                decoded_pred = raw_pred
                        else:
                            decoded_pred = raw_pred
                
                        st.success(f"### Predicted Class: **{decoded_pred}**")
                        st.session_state["last_prediction"] = decoded_pred
                
                    else:
                        # Regression
                        st.success(f"### Predicted Value: **{raw_pred}**")
                        st.session_state["last_prediction"] = raw_pred
                
                    # Save user input for download
                    st.session_state["last_input_df"] = input_df.copy()
                
                
                
                # ----------------------- DOWNLOAD RESULT SUMMARY -----------------------
                # Show download section AFTER prediction
                if (
                    "last_prediction" in st.session_state
                    and "last_input_df" in st.session_state
                    and "model_metrics" in st.session_state
                ):
                    st.subheader("Download Prediction Summary")
                
                    metrics_df = st.session_state.get("model_metrics", pd.DataFrame())
                    accuracy_df = st.session_state.get("accuracy_metrics", pd.DataFrame())
                    input_df = st.session_state.get("last_input_df", pd.DataFrame())
                    prediction = st.session_state.get("last_prediction", "")
                
                    summary_rows = []
                
                    # Prediction
                    summary_rows.append(["Prediction", "Predicted Output", prediction])
                    summary_rows.append(["Prediction", "Problem Type", st.session_state.get("target_type", "")])
                
                    # Metrics
                    if not metrics_df.empty:
                        for _, row in metrics_df.iterrows():
                            summary_rows.append(["Metrics", row["Parameter"], row["Value"]])
                
                    # Accuracy (classification only)
                    if accuracy_df is not None and not accuracy_df.empty:
                        for _, row in accuracy_df.iterrows():
                            summary_rows.append(["Accuracy", row["Set"], row["Accuracy"]])
                
                    # Input values
                    if not input_df.empty:
                        for col in input_df.columns:
                            summary_rows.append(["Input Values", col, input_df.iloc[0][col]])
                
                    # Final summary DF
                    export_df = pd.DataFrame(summary_rows, columns=["Section", "Name", "Value"])
                
                    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                
                    st.download_button(
                        label="⬇️ Download Result Summary (CSV)",
                        data=csv_bytes,
                        file_name="prediction_summary.csv",
                        mime="text/csv",
                    )



                
                
                                
                                
