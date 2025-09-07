from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
def build_product_dataset(df_productdetails: pd.DataFrame,
                          df_cataloguediscontinuation: pd.DataFrame) -> pd.DataFrame:
    """
    Merge discount data with product details.

    Parameters
    ----------
    df_productdetails : pd.DataFrame
        Product details dataset
    df_cataloguediscontinuation : pd.DataFrame
        Discount dataset

    Returns
    -------
    pd.DataFrame
        Merged dataset
    """

    # Defensive copy
    df_productdetails = df_productdetails.copy()
    df_cataloguediscontinuation = df_cataloguediscontinuation.copy()

    # --- Merge discount into product details ---
    merged = df_productdetails.merge(
        df_cataloguediscontinuation, on="ProductKey", how="right"
    )

    return merged

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def preprocess_product_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess product-week dataset into product+CatEdition level features
    and labels for discontinuation prediction.

    Steps:
    1. Aggregate numeric + categorical features at ProductKey+CatEdition level.
    2. Compute sales slope (trend) from WeeksOut vs ActualsPerWeek (via np.polyfit).
    3. Create Target_Next (discontinued in next CatEdition).
    4. Encode categorical/static columns.
    """

    # --- Define columns ---
    cat_cols = [
        "Supplier", "HierarchyLevel1", "HierarchyLevel2",
        "DIorDOM", "Seasonal", "SpringSummer", "Status"
    ]
    num_cols = ["SalePriceIncVAT", "ForecastPerWeek", "ActualsPerWeek"]

    # --- Aggregation functions ---
    agg_funcs = {col: ['mean', 'min', 'max'] for col in num_cols}
    for col in cat_cols:
        agg_funcs[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]

    # --- Aggregate features ---
    features = (
        df.groupby(['ProductKey', 'CatEdition'])
          .agg(agg_funcs)
    )
    features.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in features.columns
    ]
    features = features.reset_index()

    # --- Compute SalesSlope (trend) with np.polyfit ---
    slopes = (
    df.groupby(['ProductKey', 'CatEdition'])
      .apply(lambda g: np.polyfit(g['WeeksOut'], g['ActualsPerWeek'], 1)[0]
             if g['WeeksOut'].nunique() > 1 else 0.0)
      .reset_index(name="SalesSlope")
    )

    features = features.merge(slopes, on=['ProductKey', 'CatEdition'], how='left')

    # --- Create labels ---
    labels = (
        df.groupby(['ProductKey', 'CatEdition'])['DiscontinuedTF']
          .max()
          .reset_index()
    )
    labels['Target_Next'] = labels.groupby('ProductKey')['DiscontinuedTF'].shift(-1)
    labels = labels.drop(columns=['DiscontinuedTF'])

    # --- Merge features + labels ---
    dataset = features.merge(labels, on=['ProductKey', 'CatEdition'], how='inner')
    dataset = dataset.dropna(subset=['Target_Next'])
    dataset['Target_Next'] = dataset['Target_Next'].astype(int)

    # --- Encoding ---
    # Convert booleans to int
    for col in ['Seasonal_<lambda>', 'SpringSummer_<lambda>']:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype(int)

    # Label encode categorical ID columns
    encode_cols = [
        'Supplier_<lambda>', 'HierarchyLevel1_<lambda>', 'HierarchyLevel2_<lambda>',
        'DIorDOM_<lambda>', 'Status_<lambda>'
    ]
    for col in encode_cols:
        if col in dataset.columns:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])

    # --- Clean column names ---
    dataset = dataset.rename(columns=lambda x: x.replace("<lambda>", "mode"))
    dataset.columns = dataset.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

    return dataset

# TRAIN TEST SPLIT
def walk_forward_split(dataset, feature_drop=['Target_Next', 'ProductKey']):
    cat_editions = sorted(dataset['CatEdition'].unique())

    for cutoff in cat_editions[:-1]:  # leave last edition for unseen future
        train = dataset[dataset['CatEdition'] <= cutoff]
        test  = dataset[dataset['CatEdition'] > cutoff]   # changed here

        X_train = train.drop(columns=feature_drop)
        y_train = train['Target_Next']

        X_test = test.drop(columns=feature_drop)
        y_test = test['Target_Next']

        # Skip if y_test has only 1 class
        if y_test.nunique() < 2:
            print(f"Skipping cutoff {cutoff}: only one class in y_test ({y_test.iloc[0]})")
            continue

        yield cutoff, X_train, y_train, X_test, y_test

# SCALING
from sklearn.preprocessing import StandardScaler
import joblib
def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple:
    """
    Convert datetime columns into numeric features and scale all features.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set
    X_test : pd.DataFrame
        Testing feature set
    date_cols : list, optional
        List of datetime columns to convert (e.g., ['originationDate'])

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        Scaled X_train and X_test as DataFrames
    """
    X_train = X_train.copy()
    X_test = X_test.copy()


    # Scale numeric features
    scaler = StandardScaler()
    scaled_X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    scaled_X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    joblib.dump(scaler, "scaler.pkl")

    return scaled_X_train, scaled_X_test
