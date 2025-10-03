import streamlit as st
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from sklearn.utils.multiclass import type_of_target
import altair as alt
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             AdaBoostRegressor, RandomForestClassifier, 
                             GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, mean_absolute_error, 
                            mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000


class DataProcessor:
    """Handles data loading and cleaning operations"""
    
    @staticmethod
    @st.cache_data
    def load_data(uploaded_file):
        return pd.read_csv(uploaded_file)
    
    @staticmethod
    def clean_data(df, actions):
        cleaned = df.copy()
        
        if "Duplicates" in actions:
            cleaned = cleaned.drop_duplicates()
        
        if "Outliers" in actions:
            numerics = cleaned.select_dtypes(include=np.number)
            Q1 = numerics.quantile(0.25)
            Q3 = numerics.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (numerics < (Q1 - 1.5 * IQR)) | (numerics > (Q3 + 1.5 * IQR))
            keep_mask = ~outlier_mask.any(axis=1)
            cleaned = cleaned.loc[keep_mask]
        
        if "NaN Values" in actions:
            cleaned = cleaned.dropna()
        
        return cleaned
    
    @staticmethod
    def generate_cleaning_report(df, actions):
        report = pd.DataFrame(index=df.columns)
        
        if "NaN Values" in actions:
            report["NaN Values"] = df.isnull().sum()
        
        if "Duplicates" in actions:
            dup_mask = df.duplicated(keep=False)
            dup_counts = df.loc[dup_mask].count()
            report["Duplicates"] = dup_counts
        
        if "Outliers" in actions:
            numerics = df.select_dtypes(include=np.number)
            Q1 = numerics.quantile(0.25)
            Q3 = numerics.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (numerics < (Q1 - 1.5 * IQR)) | (numerics > (Q3 + 1.5 * IQR))
            outliers_count = outlier_mask.sum()
            report["Outliers"] = np.nan
            for col in outliers_count.index:
                report.at[col, "Outliers"] = outliers_count[col]
        
        return report.fillna('-').astype(str)


class ModelBase:
    """Base class for all ML models"""
    
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.model = None
        self.scaler = StandardScaler()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
    
    def remove_outliers(self):
        df_cleaned = self.df.copy()
        for col in df_cleaned.select_dtypes(include='number'):
            q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & 
                                   (df_cleaned[col] <= upper_bound)]
        return df_cleaned
    
    def prepare_data(self):
        df_cleaned = self.remove_outliers()
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(
            self.target_column, errors='ignore'
        )
        x = df_cleaned[numeric_cols]
        y = df_cleaned[self.target_column]
        return x, y, df_cleaned
    
    def split_and_scale(self, x, y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.3, random_state=42
        )
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
    
    def train(self):
        raise NotImplementedError("Subclasses must implement train()")
    
    def display_metrics(self):
        raise NotImplementedError("Subclasses must implement display_metrics()")
    
    def get_input_sliders(self, df_cleaned):
        """Generate sliders for user input"""
        totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(
            self.target_column, errors='ignore'
        )
        st.header("Input feature values for prediction")
        input_data = {}
        
        for col in totalcolumns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if pd.api.types.is_integer_dtype(self.df[col]):
                step = 1
                min_val = int(np.floor(lower_bound))
                max_val = int(np.ceil(upper_bound))
                default_val = int(self.df[col].median())
            else:
                step = 0.01
                min_val = float(lower_bound)
                max_val = float(upper_bound)
                default_val = float(self.df[col].median())
            
            if min_val >= max_val:
                max_val = min_val + step if pd.api.types.is_integer_dtype(self.df[col]) else min_val + 0.01
                if default_val < min_val or default_val > max_val:
                    default_val = min_val
            
            input_data[col] = st.slider(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=step
            )
        
        return pd.DataFrame([input_data])


class RegressionModel(ModelBase):
    """Base class for regression models"""
    
    def display_metrics(self):
        def adjusted_r2_score(y_true, y_pred, x):
            n = len(y_true)
            p = x.shape[1]
            r2 = r2_score(y_true, y_pred)
            return 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)
        adj_r2 = adjusted_r2_score(self.y_test, self.y_pred, self.x_test)
        mape = mean_absolute_percentage_error(np.array(self.y_test), np.array(self.y_pred))
        
        st.sidebar.header(f"{self.__class__.__name__} Metrics")
        st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.sidebar.write(f"R-squared (R²): {r2:.4f}")
        st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
        st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "adj_r2": adj_r2, "mape": mape}


class LinearRegressionModel(RegressionModel):
    def train(self):
        x, y, df_cleaned = self.prepare_data()
        self.split_and_scale(x, y)
        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        st.success("Model Trained Successfully\nYou can now Proceed to Predict the Target column")
        self.display_metrics()
        return df_cleaned


class PolynomialRegressionModel(RegressionModel):
    def __init__(self, df, target_column, degree=2):
        super().__init__(df, target_column)
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    def train(self):
        x, y, df_cleaned = self.prepare_data()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.3, random_state=42
        )
        x_train_poly = self.poly.fit_transform(self.x_train)
        x_test_poly = self.poly.transform(self.x_test)
        self.x_train = self.scaler.fit_transform(x_train_poly)
        self.x_test = self.scaler.transform(x_test_poly)
        
        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        st.success("Model Trained Successfully with Polynomial Regression")
        self.display_metrics()
        return df_cleaned
    
    def predict(self, input_df):
        input_poly = self.poly.transform(input_df)
        input_scaled = self.scaler.transform(input_poly)
        return self.model.predict(input_scaled)


class RidgeRegressionModel(RegressionModel):
    def train(self):
        x, y, df_cleaned = self.prepare_data()
        self.split_and_scale(x, y)
        ridge = Ridge()
        param = np.arange(0.000000001, 101, 1)
        parameters = {'alpha': param}
        self.model = GridSearchCV(ridge, parameters, scoring='neg_root_mean_squared_error', cv=10)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        st.success("Model Trained Successfully")
        metrics = self.display_metrics()
        best_alpha = self.model.best_params_['alpha']
        st.sidebar.write(f"Best Parameter for HPT: {best_alpha:.2f}")
        return df_cleaned


class LassoRegressionModel(RegressionModel):
    def train(self):
        x, y, df_cleaned = self.prepare_data()
        self.split_and_scale(x, y)
        lasso = Lasso()
        param = np.arange(0.000000001, 101, 1)
        parameters = {'alpha': param}
        self.model = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=10)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        st.success("Model Trained Successfully")
        metrics = self.display_metrics()
        best_alpha = self.model.best_params_['alpha']
        st.sidebar.write(f"Best Parameter for HPT: {best_alpha:.2f}")
        return df_cleaned


class ClassificationModel(ModelBase):
    """Base class for classification models"""
    
    def validate_target(self, y):
        t = type_of_target(y)
        too_many_classes = (pd.api.types.is_integer_dtype(y) and y.nunique() > 20)
        if t == 'continuous' or too_many_classes:
            st.warning(
                f"Detected target type: {t}. The selected target appears numeric/continuous "
                "and is not suitable for classification. Please choose a categorical target "
                "or switch to a regression model."
            )
            st.stop()
    
    def display_metrics(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        prec = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        rec = recall_score(self.y_test, self.y_pred, average='weighted')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        st.sidebar.header(f"{self.__class__.__name__} Metrics")
        st.sidebar.write(f"Accuracy: {acc:.4f}")
        st.sidebar.write(f"Precision: {prec:.4f}")
        st.sidebar.write(f"Recall: {rec:.4f}")
        st.sidebar.write(f"F1 Score: {f1:.4f}")
        st.sidebar.write(f"Confusion Matrix:\n{cm}")


class LogisticRegressionModel(ClassificationModel):
    def train(self):
        x, y, df_cleaned = self.prepare_data()
        self.validate_target(y)
        self.split_and_scale(x, y)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        self.display_metrics()
        return df_cleaned


class KNNClassifierModel(ClassificationModel):
    def train(self):
        x, y, df_cleaned = self.prepare_data()
        self.validate_target(y)
        self.split_and_scale(x, y)
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        self.display_metrics()
        return df_cleaned


class XGBoostClassifierModel(ClassificationModel):
    def __init__(self, df, target_column):
        super().__init__(df, target_column)
        self.label_encoder = None
    
    def train(self):
        x, y, df_cleaned = self.prepare_data()
        self.validate_target(y)
        
        if 'xgb_le' not in st.session_state:
            st.session_state['xgb_le'] = LabelEncoder().fit(y)
        self.label_encoder = st.session_state['xgb_le']
        
        y_encoded = self.label_encoder.transform(y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        self.display_metrics()
        return df_cleaned
    
    def predict(self, input_df):
        input_scaled = self.scaler.transform(input_df)
        user_pred_codes = self.model.predict(input_scaled)
        return self.label_encoder.inverse_transform(user_pred_codes)


class ModelFactory:
    """Factory class to create appropriate model instances"""
    
    REGRESSION_MODELS = {
        "Linear Regression": LinearRegressionModel,
        "Polynomial Regression": PolynomialRegressionModel,
        "Ridge Regression": RidgeRegressionModel,
        "Lasso Regression": LassoRegressionModel,
        "Elastic Net Regression": lambda df, target: RegressionModel(df, target),
        "Decision Tree Regression": lambda df, target: RegressionModel(df, target),
        "Random Forest Regression": lambda df, target: RegressionModel(df, target),
        "Gradient Boosting Regression": lambda df, target: RegressionModel(df, target),
        "Support Vector Regression": lambda df, target: RegressionModel(df, target),
        "K-Nearest Neighbors Regression": lambda df, target: RegressionModel(df, target),
        "AdaBoost Regression": lambda df, target: RegressionModel(df, target),
    }
    
    CLASSIFICATION_MODELS = {
        "Logistic Regression": LogisticRegressionModel,
        "K-Nearest Neighbors (KNN)": KNNClassifierModel,
        "XGBoost Classifier": XGBoostClassifierModel,
    }
    
    @staticmethod
    def create_model(model_name, df, target_column, model_type='regression'):
        if model_type == 'regression':
            model_class = ModelFactory.REGRESSION_MODELS.get(model_name)
        else:
            model_class = ModelFactory.CLASSIFICATION_MODELS.get(model_name)
        
        if model_class:
            return model_class(df, target_column)
        return None


class QuickMLApp:
    """Main application class"""
    
    def __init__(self):
        self.setup_page()
    
    def setup_page(self):
        logo_path = "logo.png"
        col1, col2 = st.columns([1, 6])
        with col1:
            st.image(logo_path, width=100)
        with col2:
            st.markdown(
                """
                <div style='display: flex; align-items: center;'>
                    <span style="font-size:60px; font-weight:bold; font-style: italic; 
                                 color:#fff; font-family: Arial, Helvetica, sans-serif;">QuickML</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.write("An app that enables you to clean, analyze & visualize your dataset and make predictions based on your preferred ML model")
    
    def run(self):
        file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        
        if file is not None:
            df = DataProcessor.load_data(file)
            if "clean_df" not in st.session_state:
                st.session_state["clean_df"] = df.copy()
            
            st.write("Preview of your dataset:")
            st.dataframe(st.session_state["clean_df"], width=1000)
            
            tab0, tab1, tab3, tab4, tab5 = st.tabs(
                ["Analysis", "Visualisation", "Cleaning", "Predictor", "Distribution"]
            )
            
            with tab0:
                self.show_analysis(st.session_state["clean_df"])
            
            with tab1:
                self.show_visualization(df)
            
            with tab3:
                self.show_cleaning(df)
            
            with tab4:
                self.show_predictor(df)
            
            with tab5:
                self.show_distribution(df)
    
    def show_analysis(self, df):
        st.write(df.describe())
    
    def show_visualization(self, df):
        with st.container(border=True):
            numeric_columns = df.select_dtypes(include='number').columns.tolist()
            selected_two = st.multiselect("Select exactly two columns to plot one against the other", numeric_columns)
            
            if selected_two:
                if len(selected_two) == 2:
                    x_col, y_col = selected_two
                    df_sorted = df.sort_values(by=x_col, ascending=True)
                    chart = (
                        alt.Chart(df_sorted)
                        .mark_line()
                        .encode(
                            x=alt.X(x_col, title=x_col),
                            y=alt.Y(y_col, title=y_col),
                        )
                        .properties(
                            title=f"Line plot of {y_col} vs {x_col}",
                            width="container",
                            height=300
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("Please select exactly two columns for this plot.")
            else:
                st.info("Please select at least one column to display the chart.")
    
    def show_cleaning(self, df):
        if "cleaned_df" in st.session_state:
            df = st.session_state["cleaned_df"]
        else:
            df = st.session_state["clean_df"]
        
        actions = st.multiselect("Select Actions:", ["NaN Values", "Duplicates", "Outliers"])
        
        st.write("### Report Before Cleaning")
        report_before = DataProcessor.generate_cleaning_report(df, actions)
        st.dataframe(report_before)
        
        if st.button("Clean"):
            cleaned = DataProcessor.clean_data(df, actions)
            st.session_state["cleaned_df"] = cleaned
        
        cleaned_latest = st.session_state.get("cleaned_df", st.session_state["clean_df"])
        
        st.write("### Report After Cleaning")
        report_after = DataProcessor.generate_cleaning_report(cleaned_latest, actions)
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
    
    def show_predictor(self, df):
        columns = df.columns
        dataset_choice = st.selectbox(
            "Choose the Type of Data you uploaded",
            ["None", "Numeric Type", "Classification Type"]
        )
        
        if dataset_choice == "Numeric Type":
            model_selection = st.selectbox(
                "Choose the Machine Learning Model you want the prediction from:",
                [
                    "None",
                    "Linear Regression",
                    "Polynomial Regression",
                    "Ridge Regression",
                    "Lasso Regression",
                ]
            )
            
            if model_selection != "None":
                options = df.select_dtypes(include='number').columns
                target_column = st.selectbox("Select the Target Column:", options)
                
                model = ModelFactory.create_model(model_selection, df, target_column, 'regression')
                if model:
                    df_cleaned = model.train()
                    input_df = model.get_input_sliders(df_cleaned)
                    
                    if isinstance(model, PolynomialRegressionModel):
                        user_prediction = model.predict(input_df)
                    else:
                        input_scaled = model.scaler.transform(input_df)
                        user_prediction = model.model.predict(input_scaled)
                    
                    st.success(f"Predicted Value for the given Target Class is {user_prediction}")
        
        elif dataset_choice == "Classification Type":
            model_selection = st.selectbox(
                "Choose the Machine Learning Model you want the prediction from:",
                [
                    "Logistic Regression",
                    "K-Nearest Neighbors (KNN)",
                    "XGBoost Classifier",
                ]
            )
            
            target_column = st.selectbox("Select the Target Column:", columns)
            
            model = ModelFactory.create_model(model_selection, df, target_column, 'classification')
            if model:
                df_cleaned = model.train()
                input_df = model.get_input_sliders(df_cleaned)
                
                if isinstance(model, XGBoostClassifierModel):
                    user_prediction = model.predict(input_df)
                else:
                    input_scaled = model.scaler.transform(input_df)
                    user_prediction = model.model.predict(input_scaled)
                
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
    
    def show_distribution(self, df):
        num_cols = df.select_dtypes(include="number").columns
        distribution_report = []
        alpha = 0.05
        
        for col in num_cols:
            x = df[col].dropna().values
            shapiro_stat = shapiro_p = np.nan
            k2_stat = k2_p = np.nan
            
            if x.size >= 3:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    shapiro_stat, shapiro_p = stats.shapiro(x)
            
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
        
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.info("No numeric columns to plot.")
            return
        
        bins = st.slider("Bins", 10, 100, 20, 5)
        for i in range(0, len(num_cols), 2):
            pair = num_cols[i:i + 2]
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
                        "height": 280
                    },
                    use_container_width=True,
                )


if __name__ == "__main__":
    app = QuickMLApp()
    app.run()
