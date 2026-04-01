import numpy as np

class LinearRegressionManual:
    def __init__(self, alpha=0):
        """
        alpha: Regularization strength (lambda).
        alpha=0 means Normal Equation (Task 1).
        alpha>0 means Ridge Regression (Task 3).
        """
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        # Normal Equation Formula: w = (XT * X + alpha * I)^-1 * XT * y
        XT = X.T
        XTX = XT.dot(X)
        
        # Task 3: Apply Ridge Regularization
        if self.alpha > 0:
            I = np.eye(XTX.shape[0])
            I[0, 0] = 0  # We don't regularize the intercept/bias
            XTX = XTX + self.alpha * I
            
        # Task 1: Calculate Weights using the inverse
        # Using np.linalg.inv for the matrix inversion
        self.weights = np.linalg.inv(XTX).dot(XT).dot(y)

    def predict(self, X):
        # Prediction: y_pred = X . w
        return X.dot(self.weights)