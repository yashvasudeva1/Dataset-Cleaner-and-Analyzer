from importlibraries import *

def preprocess_data(x_train, x_test, y_train=None, y_test=None):
    x_train_prep = x_train.copy()
    x_test_prep = x_test.copy()
    encoders = {}
    scaler = StandardScaler()
    for column in x_train_prep.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        x_train_prep[column] = le.fit_transform(x_train_prep[column])
        test_values = x_test_prep[column].map(
            lambda x: x if x in le.classes_ else "___unknown___"
        )
        if "___unknown___" not in le.classes_:
            le.classes_ = np.append(le.classes_, "___unknown___")
        x_test_prep[column] = le.transform(test_values)
        encoders[column] = le
    if y_train is not None and y_train.dtype == 'object':
        y_le = LabelEncoder()
        all_labels = pd.concat([y_train, y_test], axis=0)
        y_le.fit(all_labels)
        y_train_prep = y_le.transform(y_train)
        y_test_prep = y_le.transform(y_test)
    else:
        y_train_prep, y_test_prep = y_train, y_test
    numeric_cols = x_train_prep.select_dtypes(include=['int64', 'float64']).columns
    x_train_prep[numeric_cols] = x_train_prep[numeric_cols].fillna(0)
    x_test_prep[numeric_cols] = x_test_prep[numeric_cols].fillna(0)
    if len(numeric_cols) > 0:
        x_train_prep[numeric_cols] = scaler.fit_transform(x_train_prep[numeric_cols])
        x_test_prep[numeric_cols] = scaler.transform(x_test_prep[numeric_cols])
    x_train_prep = x_train_prep.fillna(0)
    x_test_prep = x_test_prep.fillna(0)
    if y_train_prep is not None:
        if isinstance(y_train_prep, pd.Series):
            y_train_prep = y_train_prep.fillna(0)
        if isinstance(y_test_prep, pd.Series):
            y_test_prep = y_test_prep.fillna(0)
    return x_train_prep, x_test_prep, y_train_prep, y_test_prep, encoders, scaler
