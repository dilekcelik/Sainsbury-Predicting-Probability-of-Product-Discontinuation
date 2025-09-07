import pandas as pd

def load_data(filepath):
    """Load raw dataset."""
    return pd.read_csv(filepath)
