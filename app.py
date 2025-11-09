import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt

st.set_page_config(page_title="QuickML", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: white;'>QuickML</h1>"
    "<p style='text-align: center;'>Analyze, visualize, and predict with pre-trained ML models</p>",
    unsafe_allow_html=True
)

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_resource
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

tabs = st.tabs(["Data Analysis", "Visualization", "Cleaning", "Model Prediction"])

# ------------------ 1. DATA ANALYSIS ------------------
with tabs[0]:
    if file:
        df = load_csv(file)
        st.dataframe(df.head())
        st.write("Summary Statistics")
        st.write(df.describe(include='all'))
    else:
        st.info("Upload a CSV file to begin.")

# ------------------ 2. VISUALIZATION ------------------
with tabs[1]:
    if file:
        df = load_csv(file)
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        selected_two = st.multiselect("Select exactly two numeric columns to plot", numeric_columns)

        if len(selected_two) == 2:
            x_col, y_col = selected_two
            df_sorted = df.sort_values(by=x_col, ascending=True)
            chart = (
                alt.Chart(df_sorted)
                .mark_line()
                .encode(x=alt.X(x_col), y=alt.Y(y_col))
                .properties(title=f"{y_col} vs {x_col}", width=700, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Please select exactly two numeric columns.")
    else:
        st.info("Upload your dataset first.")

# ------------------ 3. CLEANING ------------------
with tabs[2]:
    if file:
        df = load_csv(file)
        actions = st.multiselect("Select cleaning actions", ["NaN Values", "Duplicates", "Outliers"])
        st.write("Data Before Cleaning")
        st.dataframe(df.head())

        if st.button("Apply Cleaning"):
            clean_df = df.copy()
            if "NaN Values" in actions:
                clean_df = clean_df.dropna()
            if "Duplicates" in actions:
                clean_df = clean_df.drop_duplicates()
            if "Outliers" in actions:
                num_df = clean_df.select_dtypes(include=np.number)
                Q1 = num_df.quantile(0.25)
                Q3 = num_df.quantile(0.75)
                IQR = Q3 - Q1
                clean_df = clean_df[~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)]
            st.success("Data cleaned successfully.")
            st.dataframe(clean_df)
            csv = clean_df.to_csv(index=False)
            st.download_button("Download Cleaned Data", csv, "cleaned_data.csv", "text/csv")
    else:
        st.info("Upload your dataset first.")

# ------------------ 4. MODEL PREDICTION ------------------
with tabs[3]:
    st.subheader("Use Pre-Trained Models")
    model_choice = st.selectbox(
        "Select a Pre-Trained Model",
        [
            "Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net Regression",
            "Decision Tree Regression", "Random Forest Regression", "Gradient Boosting Regression",
            "KNN Regression", "AdaBoost Regression",
            "Logistic Regression", "SVM Classifier", "KNN Classifier", "Random Forest Classifier",
            "Naive Bayes", "LDA", "QDA", "MLP", "XGBoost", "LightGBM"
        ]
    )

    model_file_map = {
        "Linear Regression": "linearregression.pkl",
        "Ridge Regression": "ridgeregression.pkl",
        "Lasso Regression": "lassoregession.pkl",
        "Elastic Net Regression": "elasticnetregression.pkl",
        "Decision Tree Regression": "decisiontreeregression.pkl",
        "Random Forest Regression": "randomforestregression.pkl",
        "Gradient Boosting Regression": "gradientboostregression.pkl",
        "KNN Regression": "knnregression.pkl",
        "AdaBoost Regression": "adaboostregression.pkl",
        "Logistic Regression": "logisticregression.pkl",
        "SVM Classifier": "svm.pkl",
        "KNN Classifier": "knn.pkl",
        "Random Forest Classifier": "randomforest.pkl",
        "Naive Bayes": "naivebayes.pkl",
        "LDA": "lda.pkl",
        "QDA": "qda.pkl",
        "MLP": "mlp.pkl",
        "XGBoost": "xgboost.pkl",
        "LightGBM": "lightgbm.pkl"
    }

    if model_choice:
        model_file = model_file_map.get(model_choice)
        try:
            model = load_model(model_file)
            st.success(f"Loaded {model_choice} successfully.")
        except Exception:
            st.error("Error loading model. Ensure the .pkl file is present.")

        if file:
            df = load_csv(file)
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                st.subheader("Input values for prediction")
                input_data = {}
                for col in numeric_cols:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    default_val = float(df[col].median())
                    input_data[col] = st.slider(col, min_val, max_val, default_val)
                input_df = pd.DataFrame([input_data])
                if st.button("Predict"):
                    try:
                        pred = model.predict(input_df)
                        st.success(f"Prediction: {pred}")
                    except Exception as e:
                        st.error(f"Error in prediction: {e}")
            else:
                st.warning("Dataset has no numeric columns for prediction.")
        else:
            st.info("Upload your dataset first.")
