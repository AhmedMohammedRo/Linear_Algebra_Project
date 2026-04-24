from src.utils import load_and_preprocess_data
from src.model import LinearRegressionManual, GradientDescentRegression
from src.comparison import run_comparison
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('plots', exist_ok=True)

def main():
    print("=" * 65)
    print("   Linear Algebra Project: Linear Regression")
    print("=" * 65)

    # ── 1. Load & Prepare Data ──────────────────────────────────────
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    print(f"\nData loaded.  Train: {X_train.shape}  Test: {X_test.shape}")

    # ── 2. Task 1 — Normal Equation (lambda = 0) ───────────────────
    print("\n[Task 1] Normal Equation  w = (X^T X)^{-1} X^T y")
    model_ols = LinearRegressionManual(alpha=0)
    model_ols.fit(X_train, y_train)
    print(f"  Weights: {model_ols.weights}")

    # ── 3. Task 2 — Gradient Descent ───────────────────────────────
    print("\n[Task 2] Gradient Descent  (lr=0.1, iterations=1000)")
    model_gd = GradientDescentRegression(learning_rate=0.1, n_iterations=1000)
    model_gd.fit(X_train, y_train)
    print(f"  Final weights: {model_gd.weights}")
    print(f"  Final cost:    {model_gd.cost_history[-1]:.4f}")

    # ── 4. Task 2 — Convergence Plot ───────────────────────────────
    plt.figure(figsize=(7, 4))
    plt.plot(model_gd.cost_history, color='steelblue')
    plt.title('Gradient Descent: Cost vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE / 2)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/gradient_descent_convergence.png', dpi=150)
    plt.show()

    # ── 5. Task 2 — Compare Normal Eq vs Gradient Descent ─────────
    run_comparison(X_train, X_test, y_train, y_test,
                   model_ols, model_gd, alpha=0)

    # ── 6. Task 3 — Ridge Regression (multiple lambda values) ──────
    print("\n[Task 3] Ridge Regression  w = (X^T X + λI)^{-1} X^T y")

    for lam in [0, 0.1, 1, 10, 100]:
        print(f"\n  --- λ = {lam} ---")

        # Normal Equation with Ridge
        model_ridge = LinearRegressionManual(alpha=lam)
        model_ridge.fit(X_train, y_train)
        print(f"  Weights (λ={lam}): {model_ridge.weights}")

        # Gradient Descent with Ridge
        model_gd_ridge = GradientDescentRegression(
            learning_rate=0.1, n_iterations=1000, lambda_=lam
        )
        model_gd_ridge.fit(X_train, y_train)

        run_comparison(X_train, X_test, y_train, y_test,
                       model_ridge, model_gd_ridge, alpha=lam)

    # ── 7. Task 4 — Sample Predictions ────────────────────────────
    pred = model_ols.predict(X_test[:3])
    print("\n[Task 4] Sample Predictions vs Actuals")
    for p, a in zip(pred, y_test[:3]):
        print(f"  Predicted: {p:.1f} g/km   |   Actual: {a:.1f} g/km")

if __name__ == "__main__":
    main()