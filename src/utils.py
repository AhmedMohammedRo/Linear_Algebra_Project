import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_path, 'data', 'CO2_Emissions.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing CSV file at: {filepath}")

    df = pd.read_csv(filepath)
    df = df.dropna()

    # Explicitly choose meaningful features — NOT fuel consumption
    # because fuel consumption directly predicts CO2 (not interesting)
    feature_cols = ['Engine Size(L)', 'Cylinders']
    target_col   = 'CO2 Emissions(g/km)'

    X_raw = df[feature_cols].values
    y     = df[target_col].values

    # Normalize features so Gradient Descent converges properly
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Add bias column (column of 1s) for the intercept
    X_b = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_b, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler