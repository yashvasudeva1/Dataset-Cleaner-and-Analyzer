import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
st.title(":material/folder: Dataset Cleaner and Analyser")
st.write("This app helps you in making your dataset cleaner, outlier free and ready for training")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

file = st.file_uploader("")
if file is not None:
    df = load_data(file)
    # Initialize cleaned_df in session_state
    if "clean_df" not in st.session_state:
        st.session_state["clean_df"] = df.copy()

    st.write("Preview of your dataset:")
    st.dataframe(st.session_state["clean_df"], use_container_width=True)
    
    tab0, tab1, tab3, tab4, tab5 = st.tabs(
        ["Analysis", "Visualisation", "Outliers", "Predictor", "Distribution"]
    )

    with tab0:
        st.write(st.session_state["clean_df"].describe())

    with tab1:
        with st.container(border=True):
            numeric_columns = st.session_state["clean_df"].select_dtypes(include='number').columns.tolist()
            selected_columns = st.multiselect("Columns", numeric_columns, default=numeric_columns)
            if selected_columns:
                st.line_chart(st.session_state["clean_df"][selected_columns], height=250, use_container_width=True)
            else:
                st.info("Please select at least one column to display the chart.")

    with tab3:
        columns = st.session_state["clean_df"].select_dtypes(include=[np.number]).columns
        # Initial outlier report (before removal)
        outlier_report = []
        for col in columns:
            q1, q3 = st.session_state["clean_df"][col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            n_outliers = ((st.session_state["clean_df"][col] < lower) | (st.session_state["clean_df"][col] > upper)).sum()
            outlier_report.append({"Column Name": col, "Number of Outliers": n_outliers})
        outliers = pd.DataFrame(outlier_report)
        st.write("Current outliers:")
        st.write(outliers)

        remove_outlier = st.button("Remove the Outliers")
        if remove_outlier:  # Clean data and update session_state
            temp_df = st.session_state["clean_df"]
            for col in columns:
                q1, q3 = temp_df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                temp_df = temp_df[(temp_df[col] >= lower) & (temp_df[col] <= upper)]
            st.session_state["clean_df"] = temp_df  # Update the cleaned DataFrame

            # Report remaining outliers
            outlier_report = []
            for col in columns:
                q1, q3 = st.session_state["clean_df"][col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                n_outliers = ((st.session_state["clean_df"][col] < lower) | (st.session_state["clean_df"][col] > upper)).sum()
                outlier_report.append({"Column Name": col, "Number of Outliers": n_outliers})
            outliers = pd.DataFrame(outlier_report)
            st.write("Remaining outliers after removal:")
            st.write(outliers)
            st.subheader("Cleaned Dataset")
            st.write(temp_df)
    with tab4:
        columns = df.columns

        dataset_choice = st.selectbox(
            "Choose the Type of Data you uploaded",
            ["Numeric Type", "Classification Type", "None"]
        )
        
        if dataset_choice == "Numeric Type":
            model_selection = st.selectbox(
                "Choose the Machine Learning Model you want the prediction from :",
                [
                    "Linear Regression",
                    "Polynomial Regression",
                    "Ridge Regression",
                    "Lasso Regression",
                    "Elastic Net Regression",
                    "Decision Tree Regression",
                    "Random Forest Regression",
                    "Gradient Boosting Regression",
                    "Support Vector Regression",
                    "K-Nearest Neighbors Regression",
                    "AdaBoost Regression",
                    "Neural Network Regression"
                ]
            )
            target_column = st.selectbox("Select the Target Column:", columns)
        
        elif dataset_choice == "Classification Type":
            model_selection = st.selectbox(
                "Choose the Machine Learning Model you want the prediction from :",
                [
                    "Logistic Regression",
                    "K-Nearest Neighbors (KNN)",
                    "Support Vector Machine (SVM)",
                    "Decision Tree Classifier",
                    "Random Forest Classifier",
                    "Gradient Boosting Classifier",
                    "AdaBoost Classifier",
                    "Naive Bayes",
                    "Linear Discriminant Analysis",
                    "Quadratic Discriminant Analysis",
                    "XGBoost Classifier",
                    "LightGBM Classifier",
                    "Neural Network (MLPClassifier)"
                ]
            )
            target_column = st.selectbox("Select the Target Column:", columns)
        else:
            st.warning("Please Select your Data Type First")

    with tab5:
        num_cols = df.select_dtypes(include="number").columns  # numeric columns
        
        distribution_report = []
        alpha = 0.05
        
        for col in num_cols:
            x = df[col].dropna().values
            # guard: need at least 8 for normaltest; Shapiro requires n>=3
            shapiro_stat = shapiro_p = np.nan
            k2_stat = k2_p = np.nan
        
            if x.size >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(x)  # may warn if n>5000
            if x.size >= 8:  # scipy.stats.normaltest requires n>=8
                k2_stat, k2_p = stats.normaltest(x)
        
            # Decision by p-values if available
            decisions = []
            if not np.isnan(shapiro_p):
                decisions.append(shapiro_p > alpha)
            if not np.isnan(k2_p):
                decisions.append(k2_p > alpha)
        
            if decisions and all(decisions):
                verdict = "Likely normal"
            else:
                verdict = "Likely not normal"
        
            distribution_report.append({
                "Column": col,
                "n": int(x.size),
                "Shapiro W": shapiro_stat,
                "Shapiro p": shapiro_p,
                "K^2": k2_stat,
                "K^2 p": k2_p,
                "Distribution": verdict
            })
        
        distribution = pd.DataFrame(distribution_report)
        st.write(distribution)
        
        
        # numeric_cols = st.session_state["clean_df"].select_dtypes(include=np.number).columns
        # for col in numeric_cols:
        #     st.subheader(f"Histogram and KDE for {col}")
        #     fig, ax = plt.subplots()
        #     sns.histplot(st.session_state["clean_df"][col].dropna(), kde=True, bins=20, color="grey", ax=ax)
        #     ax.set_title(f"Histogram and KDE of {col}")
        #     ax.set_xlabel(col)
        #     ax.set_ylabel("Frequency")
        #     st.pyplot(fig)
        #     plt.close(fig)
