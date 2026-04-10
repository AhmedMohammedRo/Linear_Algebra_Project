import pandas as pd
import numpy as np
import os

def load_and_preprocess_data():
    """
    Loads CO2 Emission data, extracts numeric features, and prepares matrices.
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_path, 'data', 'CO2_Emissions.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing CSV file at: {filepath}")

    # 1. Load Data
    df = pd.read_csv(filepath)
    df = df.dropna()

    # 2. Extract ONLY Numeric Columns
    # This automatically drops text columns like 'Make', 'Model', 'Fuel Type'
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 3. Split Features (X) and Target (y)
    # The Target is CO2 Emissions, which is the LAST numeric column
    X_raw = numeric_df.iloc[:, :-1].values
    y = numeric_df.iloc[:, -1].values
    
    # 4. Add Bias Column (Column of 1s) for the Intercept
    X_b = np.c_[np.ones((X_raw.shape[0], 1)), X_raw]
    
    return X_b, y