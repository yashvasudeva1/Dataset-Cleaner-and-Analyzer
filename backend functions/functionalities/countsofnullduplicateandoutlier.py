def total_null(df):
    total_null=df.isnull().sum().to_frame(name='count')
    return total_null

def total_duplicates(df):
    total_duplicates=df.duplicated().sum()
    return total_duplicates

def total_outliers(df):
    outlier_counts = {}  # Use a dictionary to store outlier counts
    for column in df.select_dtypes(include=np.number).columns:
      Q1 = df[column].quantile(0.25)
      Q3 = df[column].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR

      outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
      # print(f"Outlier count for column {column}: {outlier_count}")

      outlier_counts[column] = outlier_count

    outliers=pd.DataFrame(outlier_counts,index=[0]).transpose()
    return outliers
