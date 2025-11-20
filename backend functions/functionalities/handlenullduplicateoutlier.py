from importlibraries import *
def handle_null_and_duplicates_and_outliers(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_working = df.copy()
    for column in numerical_columns:
        df_working[column].fillna(df_working[column].mean(), inplace=True)
    for column in categorical_columns:
        df_working[column].fillna(df_working[column].mode()[0], inplace=True)
    df_working.drop_duplicates(inplace=True)
    for column in numerical_columns:
        Q1 = df_working[column].quantile(0.25)
        Q3 = df_working[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_working = df_working[
            (df_working[column] >= lower_bound) &
            (df_working[column] <= upper_bound)
        ]
    return df_working
