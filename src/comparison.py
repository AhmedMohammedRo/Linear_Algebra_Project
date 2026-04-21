import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"  {name:<30} | MSE: {mse:>10.2f} | R²: {r2:.4f}")

def run_comparison(X_train, X_test, y_train, y_test,
                   normal_eq_model, gd_model, alpha=0):

    print(f"\n{'='*65}")
    print(f"  METHOD COMPARISON  (Ridge lambda = {alpha})")
    print(f"{'='*65}")
    print(f"  {'Method':<30} | {'MSE':>12} | R²")
    print(f"  {'-'*60}")

    # 1. Our Normal Equation
    evaluate("Normal Equation (ours)",
             y_test, normal_eq_model.predict(X_test))

    # 2. Our Gradient Descent
    evaluate("Gradient Descent (ours)",
             y_test, gd_model.predict(X_test))

    # 3. sklearn as ground truth
    if alpha == 0:
        sk = LinearRegression()
    else:
        sk = Ridge(alpha=alpha)
    sk.fit(X_train[:, 1:], y_train)   # sklearn adds its own intercept
    y_pred_sk = sk.predict(X_test[:, 1:])
    evaluate("scikit-learn (reference)", y_test, y_pred_sk)

    print(f"{'='*65}")

    # Verify weights match sklearn
    sk_w    = np.insert(sk.coef_, 0, sk.intercept_)
    our_w   = normal_eq_model.weights
    max_diff = np.max(np.abs(our_w - sk_w))
    print(f"\n  Weight diff (Normal Eq vs sklearn): {max_diff:.2e}")
    if max_diff < 1e-3:
        print("  ✓ Implementation is mathematically accurate.")
    else:
        print("  ✗ WARNING: Significant difference detected.")