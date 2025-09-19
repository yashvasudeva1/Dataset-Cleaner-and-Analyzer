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
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))
    outliers_count = outlier_mask.sum()
    return outlier_mask, outliers_count, lower_bound, upper_bound
def clean_outliers(df, target_col):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=np.number).columns.drop(target_col, errors='ignore')
    for col in numeric_cols:
        lower_bound, upper_bound = calculate_outlier_bounds(df_clean, col)
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def prepare_data(df, target_col, test_size=0.3, random_state=42):
    df_clean = clean_outliers(df, target_col)
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
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
def adjusted_r2_score(y_true, y_pred, X):
    n = len(y_true)
    p = X.shape[1] if len(X.shape) > 1 else 1
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, X_test=None):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'Adj_R2': adjusted_r2_score(y_test, y_pred, X_test if X_test is not None else X_train_scaled),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred)
    }
    return y_pred, metrics

def get_slider_params(df, col):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    median = df[col].median()
    if pd.api.types.is_integer_dtype(df[col]):
        step = 1
        min_val = int(np.floor(lower_bound))
        max_val = int(np.ceil(upper_bound))
        default_val = int(median)
    else:
        step = 0.01
        min_val = float(lower_bound)
        max_val = float(upper_bound)
        default_val = float(median)
    # Clamp defaults
    if default_val < min_val or default_val > max_val:
        default_val = min_val
    return min_val, max_val, default_val, step

def show_metrics_sidebar(metrics, model_name, extra_params=None):
    st.sidebar.header(f"{model_name} Metrics")
    for k, v in metrics.items():
        fmt_str = f"{k} {v:.4f}" if k != 'MAPE' else f"{k} {v:.2f}%"
        st.sidebar.write(fmt_str)
    if extra_params:
        for param, val in extra_params.items():
            st.sidebar.write(f"{param}: {val}")

def predictor_tab(df):
    st.header("Predictive Modelling")

    dataset_type = st.selectbox("Choose the Type of Data you uploaded", ["None", "Numeric Type", "Classification Type"])
    if dataset_type != "Numeric Type":
        st.warning("Currently only Numeric Type datasets are supported.")
        return

    model_options = [
        "None", "Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression",
        "Elastic Net Regression", "Decision Tree Regression", "Random Forest Regression",
        "Gradient Boosting Regression", "Support Vector Regression", "K-Nearest Neighbors Regression",
        "AdaBoost Regression"
    ]

    selected_model = st.selectbox("Choose the Machine Learning Model you want for prediction", model_options)

    if selected_model == "None":
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target_col = st.selectbox("Select the Target Column", options=numeric_cols)

    # Prepare data
    df_clean = df.copy()
    # Outlier removal can optionally be handled here as needed
    numeric_cols_no_target = [col for col in numeric_cols if col != target_col]

    # Train Test Split and Scaling
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(df_clean, target_col)

    # Model selection mapping
    models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression": ("poly", LinearRegression()),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Elastic Net Regression": ElasticNet(),
        "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
        "Random Forest Regression": RandomForestRegressor(random_state=42),
        "Gradient Boosting Regression": GradientBoostingRegressor(random_state=42),
        "Support Vector Regression": SVR(kernel='rbf'),
        "K-Nearest Neighbors Regression": KNeighborsRegressor(n_neighbors=5),
        "AdaBoost Regression": AdaBoostRegressor(random_state=42)
    }

    if selected_model not in models:
        st.error("Model not implemented yet.")
        return

    # Handle Polynomial Regression separately due to PolyFeatures
    if selected_model == "Polynomial Regression":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        scaler_poly = StandardScaler()
        X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
        X_test_poly_scaled = scaler_poly.transform(X_test_poly)
        model = models[selected_model][1]
        y_pred, metrics = train_and_evaluate_model(model, X_train_poly_scaled, y_train, X_test_poly_scaled, y_test, X_test_poly)
        scaler_used = scaler_poly
        input_features = numeric_cols_no_target
    else:
        model = models[selected_model]
        y_pred, metrics = train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, X_test)
        scaler_used = scaler
        input_features = numeric_cols_no_target

    st.success(f"Model Trained Successfully: {selected_model}. You can now proceed to predict the target.")

    show_metrics_sidebar(metrics, selected_model)

    # Input sliders for prediction
    st.header("Input feature values for prediction")
    input_data = {}
    for col in input_features:
        min_val, max_val, default_val, step = get_slider_params(df_clean, col)
        input_data[col] = st.slider(label=col, min_value=min_val, max_value=max_val, value=default_val, step=step)

    input_df = pd.DataFrame([input_data])
    if selected_model == "Polynomial Regression":
        input_poly = poly.transform(input_df)
        input_scaled = scaler_used.transform(input_poly)
        user_prediction = model.predict(input_scaled)[0]
    else:
        input_scaled = scaler_used.transform(input_df)
        user_prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Value for the given input is: {user_prediction:.4f}")
