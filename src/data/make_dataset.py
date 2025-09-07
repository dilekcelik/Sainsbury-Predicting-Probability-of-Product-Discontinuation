import pandas as pd

def load_data(filepath):
    # Read Training Data
    df_productdetails = pd.read_csv(os.path.join(filepath, "ProductDetails.csv"))
    df_cataloguediscontinuation = pd.read_csv(os.path.join(filepath, "CatalogueDiscontinuation.csv"))
    return df_productdetails, df_cataloguediscontinuation
