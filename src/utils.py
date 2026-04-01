import pandas as pd
import numpy as np
import os

def load_and_preprocess_data():
    """
    Loads data, cleans missing values, and prepares matrices for Linear Algebra.
    """
    # Dynamic path to find the data folder
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_path, 'data', 'california_housing.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing CSV file at: {filepath}")

    # 1. Load Data
    df = pd.read_csv(filepath)
    df = df.dropna()

    # 2. Split Features (X) and Target (y)
    # Based on your columns: MedInc is first, we assume house value is last
    X_raw = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 3. Add Bias Column (Column of 1s) - This represents the Intercept
    # In Linear Algebra: Y = w0*1 + w1*X1 + ...
    X_b = np.c_[np.ones((X_raw.shape[0], 1)), X_raw]
    
    return X_b, y