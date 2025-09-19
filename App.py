import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import io
import altair as alt
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier       # requires xgboost package
from lightgbm import LGBMClassifier     # requires lightgbm package
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def shapiro_safe(x):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*computed p-value may not be accurate.*")
        return stats.shapiro(x)
st.logo("Untitled design (3).svg", size='large')
st.title("Datset Cleaner and Analyser")
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
        ["Analysis", "Visualisation", "Chat" ,"Cleaning", "Predictor", "Distribution"]
    )
    with tab0:
        st.write(st.session_state["clean_df"].describe())
    with tab1:
        with st.container(border=True):
            numeric_columns = df.select_dtypes(include='number').columns.tolist()    
            selected_two = st.multiselect("Select exactly two columns to plot one against the other", numeric_columns)
    
            if selected_two:
                if len(selected_two) == 2:
                    x_col, y_col = selected_two
                    
                    # Sort the dataframe by the x column in ascending order before plotting
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
                            width=600,
                            height=300
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("Please select exactly two columns for this plot.")
            else:
                st.info("Please select at least one column to display the chart.")

    with tab3:
        df = st.session_state["clean_df"]
    
        actions = st.multiselect("Select Actions :", ["NaN Values", "Duplicates", "Outliers"])
    
        # Prepare report before cleaning using current df
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
    
        st.write("### Report Before Cleaning")
        st.dataframe(report_before.fillna('-').astype(str))
    
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
                cleaned = cleaned.loc[keep_mask]
            if "NaN Values" in actions:
                cleaned = cleaned.dropna()
    
            st.session_state["cleaned_df"] = cleaned
        if "cleaned_df" not in st.session_state:
            st.session_state["cleaned_df"] = st.session_state["clean_df"]
        cleaned_latest = st.session_state["cleaned_df"]
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
    
        st.write("### Report After Cleaning")
        st.dataframe(report_after.fillna('-').astype(str))
    
        st.write("### Cleaned Data")
        st.dataframe(cleaned_latest)
    
        csv_string = cleaned_latest.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Data",
            data=csv_string,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    with tab4:
        columns = df.columns
        dataset_choice = st.selectbox(
            "Choose the Type of Data you uploaded",
            ["None","Numeric Type", "Classification Type"]
        )
        if dataset_choice == "Numeric Type":
            model_selection = st.selectbox(
                "Choose the Machine Learning Model you want the prediction from :",
                [
                    "None",
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
                ]
            )
            options = df.select_dtypes(include='number').columns
            target_column = st.selectbox("Select the Target Column:", options)
            if model_selection == 'Linear Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = LinearRegression()
                model.fit(x_train, y_train)
                y_pred=model.predict(x_test)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, x):
                    n = len(y_true)
                    p = x.shape[1]  # number of features
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                
                st.sidebar.header("Linear Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction=model.predict(input_df)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")
            if model_selection == 'Polynomial Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                poly = PolynomialFeatures(degree=2, include_bias=False)
                x_train_poly = poly.fit_transform(x_train)
                x_test_poly = poly.transform(x_test)
                scaler = StandardScaler()
                x_train_poly = scaler.fit_transform(x_train_poly)
                x_test_poly = scaler.transform(x_test_poly)
                model = LinearRegression()
                model.fit(x_train_poly, y_train)
                y_pred = model.predict(x_test_poly)

                st.success("""Model Trained Successfully with Polynomial Regression  
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, X):
                    n = len(y_true)
                    p = x.shape[1]
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test_poly)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                st.sidebar.header("Polynomial Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_poly = poly.transform(input_df)
                input_scaled = scaler.transform(input_poly)
                user_prediction=model.predict(input_scaled)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")

            if model_selection == 'Ridge Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                ridge=Ridge()
                param=np.arange(0.000000001,101,1)
                parameters={'alpha':param}
                ridgecv=GridSearchCV(ridge,parameters,scoring='neg_root_mean_squared_error',cv=10)
                ridgecv.fit(x_train,y_train)
                y_pred=ridgecv.predict(x_test)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, x):
                    n = len(y_true)
                    p = x.shape[1] 
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                
                
                st.sidebar.header("Ridge Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                best_alpha = ridgecv.best_params_['alpha']
                st.sidebar.write(f"Best Parameter for Hyper-Parameter Tuning(HPT): {best_alpha:.2f}")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction=ridgecv.predict(input_df)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")
            if model_selection == 'Lasso Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                ridge=Lasso()
                param=np.arange(0.000000001,101,1)
                parameters={'alpha':param}
                lassocv=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=10)
                lassocv.fit(x_train,y_train)
                y_pred=lassocv.predict(x_test)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, x):
                    n = len(y_true)
                    p = x.shape[1] 
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                
                
                st.sidebar.header("Lasso Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                best_alpha = lassocv.best_params_['alpha']
                st.sidebar.write(f"Best Parameter for Hyper-Parameter Tuning(HPT): {best_alpha:.2f}")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction=lassocv.predict(input_df)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")
            if model_selection == 'Elastic Net Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = ElasticNet()
                model.fit(x_train, y_train)
                y_pred=model.predict(x_test)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, x):
                    n = len(y_true)
                    p = x.shape[1]  # number of features
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                
                st.sidebar.header("Elastic Net Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction=model.predict(input_df)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")
            if model_selection == 'Decision Tree Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = DecisionTreeRegressor(random_state=42)
                model.fit(x_train, y_train)
                y_pred=model.predict(x_test)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, x):
                    n = len(y_true)
                    p = x.shape[1]  # number of features
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                
                st.sidebar.header("Decision Tree Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction=model.predict(input_df)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")
           
            if model_selection == 'Random Forest Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = RandomForestRegressor(random_state=42)
                model.fit(x_train, y_train)
                y_pred=model.predict(x_test)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, x):
                    n = len(y_true)
                    p = x.shape[1]  # number of features
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                
                st.sidebar.header("Random Forest Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction=model.predict(input_df)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")
                
            if model_selection == 'Gradient Boosting Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = GradientBoostingRegressor(random_state=42)
                model.fit(x_train, y_train)
                y_pred=model.predict(x_test)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, x):
                    n = len(y_true)
                    p = x.shape[1]  # number of features
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                
                st.sidebar.header("Gradient Boosting Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction=model.predict(input_df)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")

            if model_selection == 'Support Vector Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = SVR(kernel='rbf')
                model.fit(x_train, y_train)
                y_pred=model.predict(x_test)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, x):
                    n = len(y_true)
                    p = x.shape[1]  # number of features
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                
                st.sidebar.header("Support Vector Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction=model.predict(input_df)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")
            if model_selection == 'K-Nearest Neighbors Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = KNeighborsRegressor(n_neighbors=5)
                model.fit(x_train, y_train)
                y_pred=model.predict(x_test)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, x):
                    n = len(y_true)
                    p = x.shape[1]  # number of features
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                
                st.sidebar.header("K-Nearest Neighbors Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction=model.predict(input_df)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")
            if model_selection == 'AdaBoost Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = AdaBoostRegressor(random_state=42)
                model.fit(x_train, y_train)
                y_pred=model.predict(x_test)
                st.success("""Model Trained Successfully   
                You can now Proceed to Predict the Target column  
                """)
                def adjusted_r2_score(y_true, y_pred, x):
                    n = len(y_true)
                    p = x.shape[1]  # number of features
                    r2 = r2_score(y_true, y_pred)
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_r2_score(y_test, y_pred, x_test)
                mape = mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))
                
                st.sidebar.header("AdaBoost Regression Metrics")
                st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.sidebar.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                st.sidebar.write(f"R-squared (R²): {r2:.4f}")
                st.sidebar.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.sidebar.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')

                st.header("Input feature values for prediction")
                
                input_data = {}
                
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction=model.predict(input_df)
                st.success(f"Predicted Value for the given Target Class is {user_prediction}")
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
            if model_selection == 'Logistic Regression':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = LogisticRegression(max_iter=1000)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("Logistic Regression Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'K-Nearest Neighbors (KNN)':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = KNeighborsClassifier(n_neighbors=5)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("K-Nearest Neighbors Classifier Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'Support Vector Machine (SVM)':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = SVC(probability=True)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("Support Vector Machine Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'Decision Tree Classifier':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = DecisionTreeClassifier()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("Decision Tree Classifier Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'Random Forest Classifier':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = RandomForestClassifier()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("Random Forest Classifier Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'Gradient Boosting Classifier':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = GradientBoostingClassifier()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("Gradient Boosting Classifier Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'AdaBoost Classifier':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = AdaBoostClassifier()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("AdaBoost Classifier Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'Naive Bayes':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = GaussianNB()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("Naive Bayes Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'Linear Discriminant Analysis':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = LinearDiscriminantAnalysis()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("Linear Discriminant Analysis Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'Quadratic Discriminant Analysis':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = QuadraticDiscriminantAnalysis()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("Quadratic Discriminant Analysis Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'XGBoost Classifier':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("XGBoost Classifier Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'LightGBM Classifier':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = LGBMClassifier()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("LightGBM Classifier Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
            
            
            if model_selection == 'Neural Network (MLPClassifier)':
                df_cleaned = df.copy()
                for col in df_cleaned.select_dtypes(include='number'):
                    q1, q3 = df_cleaned[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.drop(target_column, errors='ignore')
                x = df_cleaned[numeric_cols]
                y = df_cleaned[target_column]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                model = MLPClassifier(max_iter=500)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
            
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
            
                st.sidebar.header("Neural Network (MLP) Metrics")
                st.sidebar.write(f"Accuracy: {acc:.4f}")
                st.sidebar.write(f"Precision: {prec:.4f}")
                st.sidebar.write(f"Recall: {rec:.4f}")
                st.sidebar.write(f"F1 Score: {f1:.4f}")
                st.sidebar.write(f"Confusion Matrix:\n{cm}")
            
                totalcolumns = df_cleaned.select_dtypes(include='number').columns.drop(target_column, errors='ignore')
                st.header("Input feature values for prediction")
            
                input_data = {}
            
                for col in totalcolumns:
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
                        max_val = min_val + step if pd.api.types.is_integer_dtype(df[col]) else min_val + 0.01
                        if default_val < min_val or default_val > max_val:
                            default_val = min_val
                    input_data[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
                input_df = pd.DataFrame([input_data])
                input_df = scaler.transform(input_df)
                user_prediction = model.predict(input_df)
                st.success(f"Predicted Class for the given input is {user_prediction[0]}")
    
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



