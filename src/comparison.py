from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

def run_comparison(X, y, manual_weights, alpha=0):
    """
    Compares our manual Linear Algebra results with Scikit-learn.
    """
    print(f"\n--- Model Comparison (Alpha/Lambda = {alpha}) ---")
    
    if alpha == 0:
        sklearn_model = LinearRegression()
    else:
        sklearn_model = Ridge(alpha=alpha)
        
    # Sklearn adds bias automatically, so we remove our first column of 1s
    sklearn_model.fit(X[:, 1:], y)
    
    # Extract Sklearn weights (intercept + coefficients)
    sklearn_weights = np.insert(sklearn_model.coef_, 0, sklearn_model.intercept_)
    
    # Compare with our weights
    diff = np.abs(manual_weights - sklearn_weights)
    max_diff = np.max(diff)
    
    print(f"Manual Weights (first 3): {manual_weights[:3]}")
    print(f"Sklearn Weights (first 3): {sklearn_weights[:3]}")
    print(f"Max Difference: {max_diff:.10e}")
    
    if max_diff < 1e-3:
        print("Result: SUCCESS! Implementation is mathematically accurate.")
    else:
        print("Result: WARNING! Significant numerical difference.")