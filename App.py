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

logo_path = "data:image/jpeg;base64,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"  # Use your image file
col1, col2 = st.columns([1, 6])

with col1:
    st.image(logo_path, width=200)  # Adjust width as needed

with col2:
    st.markdown("<h1 style='margin-bottom:0;'>Dataset Cleaner and Analyser</h1>", unsafe_allow_html=True)

