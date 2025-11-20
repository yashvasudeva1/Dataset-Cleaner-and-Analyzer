def handle_null_and_duplicates_and_outliers(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_working = df.copy()

    for column in numerical_columns:
        df_working[column].fillna(df_working[column].mean(), inplace=True)

    for column in categorical_columns:
        df_working[column].fillna(df_working[column].mode()[0], inplace=True)

    df_working.drop_duplicates(inplace=True)

    # Handle outliers using Z-score
    if len(numerical_columns) > 0:
        z_scores = np.abs(stats.zscore(df_working[numerical_columns]))
        df_working = df_working[(z_scores < 3).all(axis=1)]

    return df_working
