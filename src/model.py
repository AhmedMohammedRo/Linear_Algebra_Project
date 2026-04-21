import numpy as np

class LinearRegressionManual:
    """
    Task 1 & 3: Normal Equation with optional Ridge Regularization.
    w = (X^T X + lambda * I)^{-1} X^T y
    """
    def __init__(self, alpha=0):
        self.alpha   = alpha  # lambda (regularization strength)
        self.weights = None

    def fit(self, X, y):
        XT  = X.T
        XTX = XT.dot(X)

        if self.alpha > 0:
            I        = np.eye(XTX.shape[0])
            I[0, 0]  = 0       # Do NOT regularize the bias term
            XTX      = XTX + self.alpha * I

        # np.linalg.solve is more numerically stable than np.linalg.inv
        # Solves: (X^T X) w = X^T y  →  w = (X^T X)^{-1} X^T y
        self.weights = np.linalg.solve(XTX, XT.dot(y))

    def predict(self, X):
        return X.dot(self.weights)


class GradientDescentRegression:
    """
    Task 2: Linear Regression via Gradient Descent.
    Iteratively minimizes the Mean Squared Error cost function:
        J(w) = (1/2m) * ||Xw - y||^2
    Update rule:
        w := w - alpha * (1/m) * X^T (Xw - y)
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr           = learning_rate
        self.n_iterations = n_iterations
        self.weights      = None
        self.cost_history = []   # Track convergence

    def fit(self, X, y):
        m = X.shape[0]                       # number of samples
        self.weights = np.zeros(X.shape[1])  # initialize weights to zero

        for _ in range(self.n_iterations):
            predictions = X.dot(self.weights)
            errors      = predictions - y

            # Gradient of MSE: (1/m) * X^T * errors
            gradient     = (1 / m) * X.T.dot(errors)
            self.weights = self.weights - self.lr * gradient

            # Track cost for convergence plot
            cost = (1 / (2 * m)) * np.sum(errors ** 2)
            self.cost_history.append(cost)

    def predict(self, X):
        return X.dot(self.weights)