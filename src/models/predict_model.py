import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_and_save_model(df: pd.DataFrame,
                                target_column: str = "Target_Next",
                                drop_columns: list = ["Target_Next", "ProductKey"],
                                chosen_threshold: float = 0.3,
                                save_dir: str = "Sainsbury_EndToEnd/test"):
    """
    Train a Logistic Regression model, fit the scaler, and save artifacts.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and target.
    target_column : str, optional
        Target column name. Defaults to 'Target_Next'.
    drop_columns : list, optional
        Columns to drop before training. Defaults to ['Target_Next', 'ProductKey'].
    chosen_threshold : float, optional
        Decision threshold for classification. Defaults to 0.3.
    save_dir : str, optional
        Directory to save artifacts. Defaults to 'Sainsbury_EndToEnd/test'.

    Returns
    -------
    dict
        Dictionary containing model, scaler, columns, and chosen threshold.
    """

    # --- Create save directory ---
    os.makedirs(save_dir, exist_ok=True)

    # --- Split features and target ---
    X = df.drop(columns=drop_columns)
    y = df[target_column]
    feature_columns = list(X.columns)

    # --- Save feature columns ---
    pickle.dump(feature_columns, open(os.path.join(save_dir, "Xcolumns.pkl"), "wb"))

    # --- Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Save scaler ---
    pickle.dump(scaler, open(os.path.join(save_dir, "scaler.pkl"), "wb"))

    # --- Define and train model ---
    logreg_model = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )
    logreg_model.fit(X_scaled, y)

    # --- Package model and threshold ---
    model_package = {
        "model": logreg_model,
        "threshold": chosen_threshold
    }

    # --- Save model package ---
    with open(os.path.join(save_dir, "logreg_model_to_deploy.pkl"), "wb") as file:
        pickle.dump(model_package, file)

    print(f"✅ Model, scaler, feature columns, and threshold saved successfully in '{save_dir}'")

    return {
        "model": logreg_model,
        "scaler": scaler,
        "columns": feature_columns,
        "threshold": chosen_threshold
    }


import os
import pickle
import pandas as pd

def load_and_predict(sample_data: dict,
                     artifacts_dir: str = "Sainsbury_EndToEnd/test"):
    """
    Load saved artifacts, scale new data, and make predictions.

    Parameters
    ----------
    sample_data : dict or list of dict
        Dictionary or list of dictionaries containing feature values for prediction.
    artifacts_dir : str, optional
        Directory where the model, scaler, and columns are saved.
        Defaults to 'Sainsbury_EndToEnd/test'.

    Returns
    -------
    dict
        Dictionary containing probability and predicted class for the input sample(s).
    """

    # --- Paths to artifacts ---
    model_path = os.path.join(artifacts_dir, "logreg_model_to_deploy.pkl")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    columns_path = os.path.join(artifacts_dir, "Xcolumns.pkl")

    # --- Load saved model + threshold ---
    saved_package = pickle.load(open(model_path, "rb"))
    saved_logreg_model = saved_package["model"]
    saved_threshold = saved_package["threshold"]

    # --- Load scaler ---
    saved_scaler = pickle.load(open(scaler_path, "rb"))

    # --- Load training columns ---
    train_cols = pickle.load(open(columns_path, "rb"))

    # --- Convert sample data to DataFrame ---
    sample_df = pd.DataFrame([sample_data]) if isinstance(sample_data, dict) else pd.DataFrame(sample_data)

    # --- Ensure columns match training columns ---
    sample_df = sample_df[train_cols]

    # --- Scale new data ---
    scaled_sample = saved_scaler.transform(sample_df)

    # --- Predict probability ---
    probabilities = saved_logreg_model.predict_proba(scaled_sample)[:, 1]

    # --- Apply saved threshold ---
    predictions = (probabilities >= saved_threshold).astype(int)

    # --- Build results ---
    results = []
    for prob, pred in zip(probabilities, predictions):
        results.append({
            "Predicted_Probability": prob,
            "Predicted_Class": int(pred),
            "Threshold": saved_threshold
        })

    return results
