from importlibraries import *

def preprocess_data(x_train, x_test, y_train=None, y_test=None):
    # -------------------- 1. Remove ID-like columns --------------------
    id_like = [
        col for col in x_train.columns 
        if x_train[col].nunique() == len(x_train)  # unique per row
        or x_train[col].dtype == object and x_train[col].str.len().mean() > 25  # long strings
    ]
    x_train = x_train.drop(columns=id_like, errors="ignore")
    x_test = x_test.drop(columns=id_like, errors="ignore")

    # -------------------- 2. Convert non-numeric columns properly --------------------
    x_train_prep = x_train.copy()
    x_test_prep = x_test.copy()

    # For later decoding
    target_encoder = None

    encoders = {}
    scaler = StandardScaler()

    # Identify categorical columns (object + low unique numeric)
    categorical_cols = []

    for col in x_train_prep.columns:
        if x_train_prep[col].dtype == 'object':
            categorical_cols.append(col)
        elif x_train_prep[col].nunique() < 20:  # numeric but works like categories
            categorical_cols.append(col)

    # Convert ALL categorical columns to strings
    for col in categorical_cols:
        x_train_prep[col] = x_train_prep[col].astype(str)
        x_test_prep[col] = x_test_prep[col].astype(str)

    # -------------------- 3. Label encode categorical features --------------------
    for column in categorical_cols:
        le = LabelEncoder()

        # Fit ONLY on train classes
        x_train_prep[column] = le.fit_transform(x_train_prep[column])

        # Handle unseen test values
        test_vals = x_test_prep[column].map(
            lambda x: x if x in le.classes_ else "___unknown___"
        )

        if "___unknown___" not in le.classes_:
            le.classes_ = np.append(le.classes_, "___unknown___")

        x_test_prep[column] = le.transform(test_vals)

        encoders[column] = le

    # -------------------- 4. Target Encoding --------------------
    if y_train is not None and y_train.dtype == "object":
        target_encoder = LabelEncoder()
        all_labels = pd.concat([y_train, y_test], axis=0).astype(str)
        target_encoder.fit(all_labels)

        y_train_prep = target_encoder.transform(y_train.astype(str))
        y_test_prep = target_encoder.transform(y_test.astype(str))
    else:
        y_train_prep, y_test_prep = y_train, y_test

    # -------------------- 5. Numeric columns scaling --------------------
    numeric_cols = [
        col for col in x_train_prep.columns 
        if col not in categorical_cols
    ]

    x_train_prep[numeric_cols] = x_train_prep[numeric_cols].fillna(0)
    x_test_prep[numeric_cols] = x_test_prep[numeric_cols].fillna(0)

    if len(numeric_cols) > 0:
        x_train_prep[numeric_cols] = scaler.fit_transform(x_train_prep[numeric_cols])
        x_test_prep[numeric_cols] = scaler.transform(x_test_prep[numeric_cols])

    # -------------------- 6. Final cleanup --------------------
    x_train_prep = x_train_prep.fillna(0)
    x_test_prep = x_test_prep.fillna(0)

    return (
        x_train_prep,
        x_test_prep,
        y_train_prep,
        y_test_prep,
        encoders,
        scaler,
        target_encoder,
    )
