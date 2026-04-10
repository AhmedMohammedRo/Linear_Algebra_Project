from src.utils import load_and_preprocess_data
from src.model import LinearRegressionManual
from src.comparison import run_comparison

def main():
    print("==============================================")
    print("   Linear Algebra Project: Linear Regression  ")
    print("==============================================\n")
    
    # 1. Load and Prepare Data
    try:
        X, y = load_and_preprocess_data()
        print(f"Dataset Loaded. Matrix X size: {X.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Task 1: Normal Equation Implementation
    print("\n[Executing Task 1: Normal Equation]")
    model_ols = LinearRegressionManual(alpha=0)
    model_ols.fit(X, y)
    print("Manual OLS Weights calculated successfully.")

    # 3. Task 2: Comparison with Industry Standard
    run_comparison(X, y, model_ols.weights, alpha=0)

    # 4. Task 3: Ridge Regression (Regularization)
    print("\n[Executing Task 3: Ridge Regression]")
    alpha_value = 0.5
    model_ridge = LinearRegressionManual(alpha=alpha_value)
    model_ridge.fit(X, y)
    run_comparison(X, y, model_ridge.weights, alpha=alpha_value)

    # 5. Simple Prediction Test
    test_pred = model_ols.predict(X[0:1])
    print(f"\nExample Prediction (First Car): {test_pred[0]:.2f} g/km CO2")
    print(f"Actual CO2 Emission: {y[0]:.2f} g/km")

if __name__ == "__main__":
    main()