import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import io
import requests
from PIL import Image
Image.MAX_IMAGE_PIXELS = 200_000_000
from sklearn.utils.multiclass import type_of_target
import altair as alt
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def shapiro_safe(x):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*computed p-value may not be accurate.*")
        return stats.shapiro(x)


class DataApp:
    def __init__(self):
        self.data = None
        self.cleaned_data = None
        self.model = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self._setup_ui()

    def _setup_ui(self):
        logo_path = "logo.png"
        col1, col2 = st.columns([1, 6])
        with col1:
            st.image(logo_path, width='content')
        with col2:
            st.markdown(
                """
                <div style="display: flex; align-items: center;">
                    <span style="font-size:60px; font-weight:bold; font-style: italic; color:#fff; font-family: Arial, Helvetica, sans-serif;">QuickML</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.write("An app that enables you to clean, analyze, visualize your dataset and make predictions based on your preferred ML model")

    def load_data(self):
        file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
        if file is not None:
            self.data = pd.read_csv(file)
            st.write("Preview of your dataset")
            st.dataframe(self.data, width='100%')
            if 'cleaned_data' not in st.session_state:
                st.session_state.cleaned_data = self.data.copy()

    def show_analysis(self):
        if self.data is not None:
            st.write(self.data.describe())

    def show_visualization(self):
        if self.data is not None:
            numeric_columns = self.data.select_dtypes(include='number').columns.tolist()
            selected_two = st.multiselect("Select exactly two columns to plot one against the other", numeric_columns)
            if len(selected_two) == 2:
                x_col, y_col = selected_two
                df_sorted = self.data.sort_values(by=x_col, ascending=True)
                chart = alt.Chart(df_sorted).mark_line().encode(
                    x=alt.X(x_col, title=x_col),
                    y=alt.Y(y_col, title=y_col),
                ).properties(
                    title=f"Line plot of {y_col} vs {x_col}",
                    width='container',
                    height=300
                )
                st.altair_chart(chart, use_container_width=True)
            elif len(selected_two) > 0:
                st.warning("Please select exactly two columns for this plot.")
            else:
                st.info("Please select at least one column to display the chart.")

    def clean_data(self):
        if self.data is not None:
            df = st.session_state.cleaned_data.copy()
            actions = st.multiselect("Select Actions", ["NaN Values", "Duplicates", "Outliers"])
            report_before = pd.DataFrame(index=df.columns)
            if "NaN Values" in actions:
                report_before["NaN Values"] = df.isnull().sum()
            if "Duplicates" in actions:
                dup_mask = df.duplicated(keep=False)
                dup_counts = df.loc[dup_mask].count()
                report_before["Duplicates"] = dup_counts
            if "Outliers" in actions:
                numerics = df.select_dtypes(include=np.number)
                Q1 = numerics.quantile(0.25)
                Q3 = numerics.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (numerics < (Q1 - 1.5 * IQR)) | (numerics > (Q3 + 1.5 * IQR))
                outliers_count = outlier_mask.sum()
                report_before["Outliers"] = np.nan
                for col in outliers_count.index:
                    report_before.at[col, "Outliers"] = outliers_count[col]
            st.write("Report Before Cleaning")
            st.dataframe(report_before.fillna("-").astype(str))

            if st.button("Clean"):
                cleaned = df.copy()
                if "Duplicates" in actions:
                    cleaned = cleaned.drop_duplicates()
                if "Outliers" in actions:
                    numerics_cleaned = cleaned.select_dtypes(include=np.number)
                    Q1 = numerics_cleaned.quantile(0.25)
                    Q3 = numerics_cleaned.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_mask = (numerics_cleaned < (Q1 - 1.5 * IQR)) | (numerics_cleaned > (Q3 + 1.5 * IQR))
                    keep_mask = ~outlier_mask.any(axis=1)
                    cleaned = cleaned[keep_mask]
                if "NaN Values" in actions:
                    cleaned = cleaned.dropna()
                st.session_state.cleaned_data = cleaned
                st.success("Data cleaned successfully!")

            cleaned_latest = st.session_state.get('cleaned_data', df)
            report_after = pd.DataFrame(index=cleaned_latest.columns)
            if "NaN Values" in actions:
                report_after["NaN Values"] = cleaned_latest.isnull().sum()
            if "Duplicates" in actions:
                dup_mask = cleaned_latest.duplicated(keep=False)
                dup_counts = cleaned_latest.loc[dup_mask].count()
                report_after["Duplicates"] = dup_counts
            if "Outliers" in actions:
                numerics_after = cleaned_latest.select_dtypes(include=np.number)
                Q1 = numerics_after.quantile(0.25)
                Q3 = numerics_after.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask_after = (numerics_after < (Q1 - 1.5 * IQR)) | (numerics_after > (Q3 + 1.5 * IQR))
                outliers_count_after = outlier_mask_after.sum()
                report_after["Outliers"] = np.nan
                for col in outliers_count_after.index:
                    report_after.at[col, "Outliers"] = outliers_count_after[col]
            st.write("Report After Cleaning")
            st.dataframe(report_after.fillna("-").astype(str))
            st.write("Cleaned Data")
            st.dataframe(cleaned_latest)
            csv_string = cleaned_latest.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Data",
                data=csv_string,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

    def _train_and_evaluate_regression(self, model, X_train, X_test, y_train, y_test, model_name, target_column):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        n = len(y_test)
        p = X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        st.sidebar.subheader(f"{model_name} Metrics")
        st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.sidebar.write(f"R-squared (R²): {r2:.4f}")
        st.sidebar.write(f"Adjusted R²: {adj_r2:.4f}")
        st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        return model, y_pred

    def _get_input_for_prediction(self, df, target_column):
        total_columns = df.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
        input_data = {}
        for col in total_columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            if pd.api.types.is_integer_dtype(df[col]):
                step = 1
                min_val = int(np.floor(lower_bound))
                max_val = int(np.ceil(upper_bound))
                default_val = int(df[col].median())
            else:
                step = 0.01
                min_val = float(lower_bound)
                max_val = float(upper_bound)
                default_val = float(df[col].median())
            if min_val >= max_val:
                max_val = min_val + step
            if default_val < min_val or default_val > max_val:
                default_val = min_val
            input_data[col] = st.slider(label=col, min_value=min_val, max_value=max_val, value=default_val, step=step)
        return pd.DataFrame([input_data])

    def predict(self):
        if self.data is not None:
            columns = self.data.columns
            dataset_choice = st.selectbox("Choose the Type of Data you uploaded", ["None", "Numeric Type", "Classification Type"])
            if dataset_choice == "Numeric Type":
                model_selection = st.selectbox("Choose the Machine Learning Model you want the prediction from", [
                    "None", "Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression",
                    "Elastic Net Regression", "Decision Tree Regression", "Random Forest Regression",
                    "Gradient Boosting Regression", "Support Vector Regression", "K-Nearest Neighbors Regression",
                    "AdaBoost Regression"
                ])
                target_column = st.selectbox("Select the Target Column", options=columns)
                if model_selection != "None":
                    df_cleaned = st.session_state.cleaned_data.copy()
                    for col in df_cleaned.select_dtypes(include='number'):
                        q1, q3 = df_cleaned[col].quantile(0.25), df_cleaned[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                    numeric_cols = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                    X = df_cleaned[numeric_cols]
                    y = df_cleaned[target_column]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    if model_selection == "Linear Regression":
                        model = LinearRegression()
                        model, y_pred = self._train_and_evaluate_regression(model, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression", target_column)
                        st.success("Model Trained Successfully! You can now Proceed to Predict the Target column")
                    elif model_selection == "Polynomial Regression":
                        poly = PolynomialFeatures(degree=2, include_bias=False)
                        X_train_poly = poly.fit_transform(X_train_scaled)
                        X_test_poly = poly.transform(X_test_scaled)
                        model = LinearRegression()
                        model.fit(X_train_poly, y_train)
                        y_pred = model.predict(X_test_poly)
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        n = len(y_test)
                        p = X_test_poly.shape[1]
                        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                        st.sidebar.subheader("Polynomial Regression Metrics")
                        st.sidebar.write(f"MAE: {mae:.4f}")
                        st.sidebar.write(f"MSE: {mse:.4f}")
                        st.sidebar.write(f"RMSE: {rmse:.4f}")
                        st.sidebar.write(f"R²: {r2:.4f}")
                        st.sidebar.write(f"Adjusted R²: {adj_r2:.4f}")
                        st.sidebar.write(f"MAPE: {mape:.2f}%")
                        st.success("Model Trained Successfully with Polynomial Regression! You can now Proceed to Predict the Target column")
                    # Add other models similarly...
                    input_df = self._get_input_for_prediction(df_cleaned, target_column)
                    input_scaled = self.scaler.transform(input_df)
                    if model_selection == "Polynomial Regression":
                        input_scaled = poly.transform(input_scaled)
                    user_prediction = model.predict(input_scaled)
                    st.success(f"Predicted Value for the given Target Class is {user_prediction[0]:.4f}")
            elif dataset_choice == "Classification Type":
                # Similar structure for classification
                pass

    def show_distribution(self):
        if self.data is not None:
            num_cols = self.data.select_dtypes(include='number').columns
            distribution_report = []
            alpha = 0.05
            for col in num_cols:
                x = self.data[col].dropna().values
                shapiro_stat, shapiro_p = np.nan, np.nan
                k2_stat, k2_p = np.nan, np.nan
                if x.size >= 3:
                    shapiro_stat, shapiro_p = shapiro_safe(x)
                if x.size >= 8:
                    k2_stat, k2_p = stats.normaltest(x)
                decisions = []
                if not np.isnan(shapiro_p):
                    decisions.append(shapiro_p > alpha)
                if not np.isnan(k2_p):
                    decisions.append(k2_p > alpha)
                verdict = "Likely normal" if decisions and all(decisions) else "Likely not normal"
                distribution_report.append({
                    "Column": col,
                    "n": int(x.size),
                    "Shapiro W": shapiro_stat,
                    "Shapiro p": shapiro_p,
                    "K2": k2_stat,
                    "K2 p": k2_p,
                    "Distribution": verdict
                })
            distribution_df = pd.DataFrame(distribution_report)
            st.write(distribution_df.style.format({"Shapiro p": "{:.2f}"}))
            bins = st.slider("Bins", 10, 100, 20, 5)
            for i in range(0, len(num_cols), 2):
                pair = num_cols[i:i+2]
                cols = st.columns(len(pair))
                for holder, col in zip(cols, pair):
                    holder.caption(f"Histogram: {col}")
                    data = self.data[col].rename(col).dropna()
                    holder.vega_lite_chart(
                        data,
                        mark='bar',
                        encoding={
                            'x': {'field': col, 'type': 'quantitative', 'bin': {'maxbins': int(bins)}},
                            'y': {'aggregate': 'count', 'type': 'quantitative', 'title': 'Count'}
                        },
                        height=280,
                        use_container_width=True
                    )

    def run(self):
        self.load_data()
        if self.data is not None:
            tab0, tab1, tab3, tab4, tab5 = st.tabs(["Analysis", "Visualization", "Cleaning", "Predictor", "Distribution"])
            with tab0:
                self.show_analysis()
            with tab1:
                self.show_visualization()
            with tab3:
                self.clean_data()
            with tab4:
                self.predict()
            with tab5:
                self.show_distribution()


# Run the app
if __name__ == "__main__":
    app = DataApp()
    app.run()
