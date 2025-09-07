import pandas as pd

def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    df_preds = pd.DataFrame(predictions, columns=["prediction"])
    df_preds.to_csv("data/processed/predictions.csv", index=False)
    return df_preds
