import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import altair as alt
import warnings

warnings.filterwarnings("ignore")

class DataHandler:
    def __init__(self):
        self.df = None
        self.cleaned_df = None

    @st.cache_data
    def load(self, uploaded_file):
        return pd.read_csv(uploaded_file)

    def set_df(self, df):
        self.df = df.copy()
        if self.cleaned_df is None:
            self.cleaned_df = self.df.copy()

    def report_before(self, actions):
        df = self.cleaned_df.copy()
        report = pd.DataFrame(index=df.columns)
        if "NaN Values" in actions:
            report["NaN Values"] = df.isnull().sum()
        if "Duplicates" in actions:
            dup_mask = df.duplicated(keep=False)
            report["Duplicates"] = df.loc[dup_mask].count()
        if "Outliers" in actions:
            nums = df.select_dtypes(include=np.number)
            q1 = nums.quantile(0.25)
            q3 = nums.quantile(0.75)
            iqr = q3 - q1
            mask = (nums < (q1 - 1.5 * iqr)) | (nums > (q3 + 1.5 * iqr))
            out = mask.sum()
            report["Outliers"] = np.nan
            for c in out.index:
                report.at[c, "Outliers"] = out[c]
        return report.fillna("-").astype(str)

    def clean(self, actions):
        df = self.cleaned_df.copy()
        if "Duplicates" in actions:
            df = df.drop_duplicates()
        if "Outliers" in actions:
            nums = df.select_dtypes(include=np.number)
            q1 = nums.quantile(0.25)
            q3 = nums.quantile(0.75)
            iqr = q3 - q1
            mask = (nums < (q1 - 1.5 * iqr)) | (nums > (q3 + 1.5 * iqr))
            keep = ~mask.any(axis=1)
            df = df.loc[keep]
        if "NaN Values" in actions:
            df = df.dropna()
        self.cleaned_df = df
        return df

class UIHelpers:
    @staticmethod
    def select_two_line_plot(df):
        nums = df.select_dtypes(include='number').columns.tolist()
        sel = st.multiselect("Select exactly two columns to plot", nums)
        if len(sel) == 2:
            x, y = sel
            sorted_df = df.sort_values(by=x)
            chart = alt.Chart(sorted_df).mark_line().encode(x=alt.X(x), y=alt.Y(y)).properties(width="container", height=300, title=f"{y} vs {x}")
            st.altair_chart(chart, use_container_width=True)
        elif len(sel) > 0:
            st.warning("Please select exactly two columns.")
        else:
            st.info("Select two numeric columns to plot.")

    @staticmethod
    def make_input_sliders(df, scaler=None, exclude=None):
        exclude = exclude or []
        cols = df.select_dtypes(include='number').columns.drop(exclude, errors='ignore')
        inputs = {}
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lb = q1 - 1.5 * iqr
            ub = q3 + 1.5 * iqr
            if pd.api.types.is_integer_dtype(df[col]):
                step = 1
                vmin = int(np.floor(lb))
                vmax = int(np.ceil(ub))
                default = int(df[col].median())
            else:
                step = 0.01
                vmin = float(lb)
                vmax = float(ub)
                default = float(df[col].median())
            if vmin >= vmax:
                vmax = vmin + (step if step == 1 else 0.01)
                if default < vmin or default > vmax:
                    default = vmin
            inputs[col] = st.slider(col, min_value=vmin, max_value=vmax, value=default, step=step)
        input_df = pd.DataFrame([inputs])
        if scaler is not None:
            input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        return input_df

class MetricCalculator:
    @staticmethod
    def regression(y_true, y_pred, x):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        p = x.shape[1] if hasattr(x, "shape") else 0
        adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if p and n - p - 1 != 0 else np.nan
        mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.array(y_true) + 1e-9))) * 100
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Adj R2": adj, "MAPE%": mape}

    @staticmethod
    def classification(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "Confusion Matrix": cm}

class ModelPipeline:
    registry = {
        "Linear Regression": (LinearRegression, "reg"),
        "Polynomial Regression": ("poly", "reg"),
        "Ridge Regression": (Ridge, "reg"),
        "Lasso Regression": (Lasso, "reg"),
        "Elastic Net": (ElasticNet, "reg"),
        "Decision Tree Regression": (DecisionTreeRegressor, "reg"),
        "Random Forest Regression": (RandomForestRegressor, "reg"),
        "Gradient Boosting Regression": (GradientBoostingRegressor, "reg"),
        "SVR": (SVR, "reg"),
        "KNN Regression": (KNeighborsRegressor, "reg"),
        "AdaBoost Regression": (AdaBoostRegressor, "reg"),
        "Logistic Regression": (LogisticRegression, "clf"),
        "KNN Classifier": (KNeighborsClassifier, "clf"),
        "SVM Classifier": (SVC, "clf"),
        "Decision Tree Classifier": (DecisionTreeClassifier, "clf"),
        "Random Forest Classifier": (RandomForestClassifier, "clf"),
        "Gradient Boosting Classifier": (GradientBoostingClassifier, "clf"),
        "AdaBoost Classifier": (AdaBoostClassifier, "clf"),
        "Naive Bayes": (GaussianNB, "clf"),
        "LDA": (LinearDiscriminantAnalysis, "clf"),
        "QDA": (QuadraticDiscriminantAnalysis, "clf"),
        "XGBoost": (XGBClassifier, "clf"),
        "LightGBM": (LGBMClassifier, "clf"),
        "MLP": (MLPClassifier, "clf"),
    }

    def __init__(self, model_name, df, target, poly_degree=None):
        self.name = model_name
        self.df = df.copy()
        self.target = target
        self.poly_degree = poly_degree
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = None

    def _prepare(self):
        nums = self.df.select_dtypes(include='number')
        X = nums.drop(columns=[self.target], errors='ignore')
        y = self.df[self.target].copy()
        return X, y

    def train(self):
        X, y = self._prepare()
        t = ModelPipeline.registry.get(self.name)
        if t is None:
            st.error("Model not supported")
            return None
        cls_or_flag, kind = t
        if self.name == "Polynomial Regression":
            poly = PolynomialFeatures(degree=self.poly_degree or 2, include_bias=False)
            X = poly.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            self.model = LinearRegression()
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            if kind == "reg":
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
            else:
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
            if kind == "clf":
                if isinstance(cls_or_flag, type):
                    self.model = cls_or_flag()
                else:
                    self.model = cls_or_flag
            else:
                self.model = cls_or_flag()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        if kind == "reg":
            metrics = MetricCalculator.regression(y_test, y_pred, X_test)
        else:
            metrics = MetricCalculator.classification(y_test, y_pred)
        return metrics

    def predict(self, input_df):
        if self.model is None:
            st.warning("Train model first")
            return None
        scaled = self.scaler.transform(input_df)
        pred = self.model.predict(scaled)
        return pred

def app():
    st.set_page_config(layout="wide")
    st.title("QuickML (Refactored)")
    logo, _ = st.columns([1, 6])
    with logo:
        pass

    uploader = st.file_uploader("Upload CSV", type=["csv"])
    dh = DataHandler()
    if uploader:
        df = dh.load(uploader)
        dh.set_df(df)
        st.write("Preview:")
        st.dataframe(dh.cleaned_df)

        tabs = st.tabs(["Analysis", "Visualisation", "Cleaning", "Predictor", "Distribution"])
        with tabs[0]:
            st.write(dh.cleaned_df.describe())
        with tabs[1]:
            UIHelpers.select_two_line_plot(dh.cleaned_df)
        with tabs[2]:
            actions = st.multiselect("Select Actions", ["NaN Values", "Duplicates", "Outliers"])
            st.write("Report before cleaning")
            st.dataframe(dh.report_before(actions))
            if st.button("Clean"):
                dh.clean(actions)
            st.write("Report after cleaning")
            st.dataframe(dh.report_before(actions))
            st.download_button("Download cleaned CSV", dh.cleaned_df.to_csv(index=False), file_name="cleaned.csv", mime="text/csv")
        with tabs[3]:
            kind = st.selectbox("Data type", ["None", "Numeric", "Classification"])
            if kind == "Numeric":
                model_name = st.selectbox("Model", [k for k, v in ModelPipeline.registry.items() if v[1] == "reg"] + ["Polynomial Regression"])
            elif kind == "Classification":
                model_name = st.selectbox("Model", [k for k, v in ModelPipeline.registry.items() if v[1] == "clf"])
            else:
                st.info("Choose Numeric or Classification")
                model_name = None
            if model_name:
                target = st.selectbox("Target column", dh.cleaned_df.columns)
                if st.button("Train Model"):
                    mp = ModelPipeline(model_name, dh.cleaned_df, target, poly_degree=2)
                    metrics = mp.train()
                    st.sidebar.write(metrics)
                    st.success("Model trained")
                    input_df = UIHelpers.make_input_sliders(dh.cleaned_df, scaler=mp.scaler, exclude=[target])
                    pred = mp.predict(input_df)
                    st.write("Prediction:", pred)
        with tabs[4]:
            nums = dh.cleaned_df.select_dtypes(include='number').columns.tolist()
            if not nums:
                st.info("No numeric columns")
            else:
                bins = st.slider("Bins", 10, 100, 20)
                for i in range(0, len(nums), 2):
                    pair = nums[i:i+2]
                    cols = st.columns(len(pair))
                    for holder, col in zip(cols, pair):
                        holder.caption(col)
                        data = dh.cleaned_df[[col]].rename(columns={col: "value"}).dropna()
                        holder.vega_lite_chart(data, {"mark":"bar", "encoding":{"x":{"field":"value","type":"quantitative","bin":{"maxbins":int(bins)},"title":col},"y":{"aggregate":"count","type":"quantitative","title":"Count"}}}, use_container_width=True)

if __name__ == "__main__":
    app()
