import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import io
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def shapiro_safe(x):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*computed p-value may not be accurate.*")
        return stats.shapiro(x)


class DataApp:
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self._setup_ui()

    def _setup_ui(self):
        logo_path = "logo.png"
        col1, col2 = st.columns([1, 6])
        with col1:
            try:
                st.image(logo_path, width=120)
            except:
                st.write("🖼️")
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
            st.dataframe(self.data, use_container_width=True)
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
            st.dataframe(report_before.fillna("-").astype(str), use_container_width=True)

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
            st.dataframe(report_after.fillna("-").astype(str), use_container_width=True)
            st.write("Cleaned Data")
            st.dataframe(cleaned_latest, use_container_width=True)
            csv_string = cleaned_latest.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Data",
                data=csv_string,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

    def _get_classification_model(self, model_name):
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            return DecisionTreeClassifier()
        elif model_name == "Random Forest":
            return RandomForestClassifier()
        elif model_name == "Gradient Boosting":
            return GradientBoostingClassifier()
        elif model_name == "AdaBoost":
            return AdaBoostClassifier()
        elif model_name == "K-Nearest Neighbors":
            return KNeighborsClassifier()
        elif model_name == "Naive Bayes":
            return GaussianNB()
        elif model_name == "SVM":
            return SVC()
        elif model_name == "XGBoost":
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif model_name == "LightGBM":
            return LGBMClassifier()
        elif model_name == "MLP Neural Network":
            return MLPClassifier(max_iter=500)
        elif model_name == "Linear Discriminant Analysis":
            return LinearDiscriminantAnalysis()
        elif model_name == "Quadratic Discriminant Analysis":
            return QuadraticDiscriminantAnalysis()
        else:
            st.error("Invalid model selected.")
            return None

    def predict(self):
        if self.data is not None:
            if 'dataset_choice' not in st.session_state:
                st.session_state.dataset_choice = "None"

            dataset_choice = st.selectbox(
                "Choose the Type of Data you uploaded",
                ["None", "Numeric Type", "Classification Type"],
                key="dataset_type_select"
            )

            if dataset_choice != st.session_state.get('dataset_choice'):
                st.session_state.dataset_choice = dataset_choice
                st.rerun()

            if dataset_choice == "None":
                return

            if dataset_choice == "Numeric Type":
                model_selection = st.selectbox(
                    "Choose the Machine Learning Model for Regression",
                    [
                        "None", "Linear Regression", "Polynomial Regression", "Ridge Regression",
                        "Lasso Regression", "Elastic Net Regression", "Decision Tree Regression",
                        "Random Forest Regression", "Gradient Boosting Regression",
                        "Support Vector Regression", "K-Nearest Neighbors Regression",
                        "AdaBoost Regression"
                    ],
                    key="reg_model_select"
                )
                if model_selection == "None":
                    return

                target_column = st.selectbox(
                    "Select the Target Column (Numeric)",
                    options=self.data.columns,
                    key="reg_target_select"
                )

                if not np.issubdtype(self.data[target_column].dtype, np.number):
                    st.error("Selected target column is not numeric. Choose a valid numeric column.")
                    return

                st.info(f"Regression: {model_selection} on {target_column}")
                # Add regression logic later

            elif dataset_choice == "Classification Type":
                model_selection = st.selectbox(
                    "Choose the Machine Learning Model for Classification",
                    [
                        "None", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting",
                        "AdaBoost", "K-Nearest Neighbors", "Naive Bayes", "SVM", "XGBoost", "LightGBM",
                        "MLP Neural Network", "Linear Discriminant Analysis", "Quadratic Discriminant Analysis"
                    ],
                    key="clf_model_select"
                )
                if model_selection == "None":
                    return

                target_column = st.selectbox(
                    "Select the Target Column (Class Label)",
                    options=self.data.columns,
                    key="clf_target_select"
                )

                if self.data[target_column].nunique() < 2:
                    st.error("Target column must have at least 2 unique classes.")
                    return

                st.info(f"Training {model_selection} on {target_column}...")

                df_cleaned = st.session_state.cleaned_data.copy()
                if df_cleaned[target_column].dtype == 'object':
                    y = self.le.fit_transform(df_cleaned[target_column])
                else:
                    y = df_cleaned[target_column].values

                X = df_cleaned.select_dtypes(include='number').drop(columns=[target_column], errors='ignore')
                if X.empty:
                    st.error("No numeric features available for training.")
                    return

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                model = self._get_classification_model(model_selection)
                if model is None:
                    return

                with st.spinner("Training model..."):
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                st.sidebar.subheader(f"{model_selection} Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1-Score: {f1:.4f}")

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f"Confusion Matrix - {model_selection}")
                st.pyplot(fig)

                st.success("✅ Classification Model Trained Successfully!")

                st.write("### Enter Feature Values for Prediction")
                input_data = {}
                for col in X.columns:
                    min_val = float(df_cleaned[col].min())
                    max_val = float(df_cleaned[col].max())
                    default_val = float(df_cleaned[col].median())
                    if min_val == max_val:
                        max_val = min_val + 1.0
                    input_data[col] = st.slider(
                        f"{col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=0.01,
                        key=f"slider_{col}"
                    )

                input_df = pd.DataFrame([input_data])
                input_scaled = self.scaler.transform(input_df)

                if st.button("🔮 Predict Class"):
                    pred = model.predict(input_scaled)[0]
                    predicted_label = pred
                    if df_cleaned[target_column].dtype == 'object':
                        predicted_label = self.le.inverse_transform([pred])[0]
                    st.success(f"🎯 Predicted Class: **{predicted_label}**")

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_scaled)[0]
                        class_names = self.le.classes_ if hasattr(self.le, 'classes_') else np.unique(y)
                        proba_str = ", ".join([f"{cls}: {p:.2f}" for cls, p in zip(class_names, proba)])
                        st.info(f"📊 Prediction Probabilities: {proba_str}")

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
                    data = self.data[col].dropna()
                    chart = alt.Chart(data.to_frame(name=col)).mark_bar().encode(
                        x=alt.X(f'{col}:Q', bin=alt.Bin(maxbins=int(bins))),
                        y=alt.Y('count():Q', title='Count')
                    ).properties(height=280)
                    holder.altair_chart(chart, use_container_width=True)

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


if __name__ == "__main__":
    app = DataApp()
    app.run()
