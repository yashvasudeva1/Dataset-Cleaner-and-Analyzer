import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import io
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
def shapiro_safe(x):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*computed p-value may not be accurate.*")
        return stats.shapiro(x)
st.title(":material/folder: Dataset Cleaner and Analyser")
st.write("This app helps you in making your dataset cleaner, outlier free and ready for training")
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)
file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
if file is not None:
    df = load_data(file)
    if "clean_df" not in st.session_state:
        st.session_state["clean_df"] = df.copy()
    st.write("Preview of your dataset:")
    st.dataframe(st.session_state["clean_df"], use_container_width=True)
    
    tab0, tab1,tab2 , tab3, tab4, tab5 = st.tabs(
        ["Analysis", "Visualisation", "Chat" ,"Outliers", "Predictor", "Distribution"]
    )
    with tab0:
        st.write(st.session_state["clean_df"].describe())
    with tab1:
        with st.container(border=True):
            numeric_columns = df.select_dtypes(include='number').columns.tolist()
            selected_columns = st.multiselect("Columns", numeric_columns, default=numeric_columns)
            if selected_columns:
                st.line_chart(df[selected_columns], height=250, use_container_width=True) 
            else:
                st.info("Please select at least one column to display the chart.")
    with tab2:
        st.title("Chat with your Dataset")
        Model=st.selectbox("Select the Model you want to chat with :",["Gemini",'Llama',"Deepseek",'Qwen'])
        apikey=st.text_input(f"Enter {Model} API key")
    with tab3:
        columns = st.session_state["clean_df"].select_dtypes(include=[np.number]).columns
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
        if remove_outlier: 
            temp_df = st.session_state["clean_df"]
            for col in columns:
                q1, q3 = temp_df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                temp_df = temp_df[(temp_df[col] >= lower) & (temp_df[col] <= upper)]
            st.session_state["clean_df"] = temp_df 
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
            # csv_data = temp_df.to_csv(index=False).encode('utf-8')
            # st.download_button(
            #     label="Download Cleaned Data",
            #     data=csv_data,
            #     file_name='cleaned.csv',
            #     mime='text/csv'
            # )

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
            if model_selection == 'Linear Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.columns:
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                x = df_cleaned.drop(target_column, axis=1)
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = LinearRegression()
                model.fit(x_train, y_train)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                    
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
        num_cols = df.select_dtypes(include="number").columns
        distribution_report = []
        alpha = 0.05
        for col in num_cols:
            x = df[col].dropna().values
            shapiro_stat = shapiro_p = np.nan
            k2_stat = k2_p = np.nan
            if x.size >= 3:
                shapiro_stat, shapiro_p = shapiro_safe(x)
            if x.size >= 8:
                k2_stat, k2_p = stats.normaltest(x) 
            decisions = []
            if not np.isnan(shapiro_p):
                decisions.append(shapiro_p > alpha)
            if not np.isnan(k2_p):
                decisions.append(k2_p > alpha)
            verdict = "Likely normal" if (decisions and all(decisions)) else "Likely not normal"
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
        st.write(distribution.style.format({"Shapiro p": "{:.2f}".format}))
        num_cols = df.select_dtypes(include="number").columns.tolist()  # numeric columns [web:53]
        if not num_cols:
            st.info("No numeric columns to plot.")
            st.stop()
        bins = st.slider("Bins", 10, 100, 20, 5)
        for i in range(0, len(num_cols), 2):
            pair = num_cols[i:i+2]
            cols = st.columns(len(pair))
            for holder, col in zip(cols, pair):
                holder.caption(f"Histogram: {col}")
                data = df[[col]].rename(columns={col: "value"}).dropna()
                holder.vega_lite_chart(
                    data,
                    {
                        "mark": "bar",
                        "encoding": {
                            "x": {
                                "field": "value",
                                "type": "quantitative",
                                "bin": {"maxbins": int(bins)}, 
                                "title": col
                            },
                            "y": {"aggregate": "count", "type": "quantitative", "title": "Count"},
                        },
                        "width": "container",
                        "height": 280,
                    },
                    use_container_width=True,
                )

