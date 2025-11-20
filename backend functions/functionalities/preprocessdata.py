from importlibraries import *
def preprocess_data(x_train, x_test, y_train=None, y_test=None):
    x_train_prep = x_train.copy()
    x_test_prep = x_test.copy()
    encoders = {}
    scaler = StandardScaler()

    #Encode categorical columns
    for column in x_train_prep.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        x_train_prep[column] = le.fit_transform(x_train_prep[column])
        x_test_prep[column] = le.transform(x_test_prep[column])
        encoders[column] = le

    #Encode target column if categorical
    if y_train is not None:
        if y_train.dtype == 'object':
            y_le = LabelEncoder()
            y_train_prep = y_le.fit_transform(y_train)
            y_test_prep = y_le.transform(y_test) if y_test is not None else None
        else:
            y_train_prep, y_test_prep = y_train, y_test
    else:
        y_train_prep, y_test_prep = None, None

    #Scale numeric columns
    numeric_cols = x_train_prep.select_dtypes(include=['int64', 'float64']).columns
    x_train_prep[numeric_cols] = scaler.fit_transform(x_train_prep[numeric_cols])
    x_test_prep[numeric_cols] = scaler.transform(x_test_prep[numeric_cols])

    return x_train_prep, x_test_prep, y_train_prep, y_test_prep, encoders, scaler

