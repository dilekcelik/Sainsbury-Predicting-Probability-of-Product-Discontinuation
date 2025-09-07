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
