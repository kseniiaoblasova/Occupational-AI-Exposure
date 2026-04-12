import numpy as np


class LogisticRegression:
    # initializing the class
    def __init__(self, lr_0=0.01, tol=1e-7, epochs=1000, decay=0.005):
        self.lr_0 = lr_0
        self.tol = tol
        self.epochs = epochs
        self.decay = decay
        self.theta = None
        self.ll_history = []

    # defining the sigmoid (squashing) function
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    # defining the fit method
    def fit(self, X, y):
        n = X.shape[0]
        p = X.shape[1]

        # adding bias term and initializing parameters
        X = np.column_stack([np.ones(n), X])
        self.theta = np.zeros((p+1,), float)

        # initializing previous log likelihood to infinity for the first epoch check
        previous_LL = np.inf

        # updating parameters by moving in direction of steepest ascent (maximizing the log likelihood function)
        for i in range(self.epochs):

            lr_t = self.lr_0 / (1 + self.decay * i)

            # computing predictions with current parameters
            predictions = self.sigmoid(X @ self.theta)
            predictions = np.clip(predictions, 1e-15, 1-1e-15)

            # computing log likelihood function value
            current_LL = np.sum(
                y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
            )

            # appending current log likelihood to track how the function is progressing over time
            self.ll_history.append(current_LL)

            # computing gradient of log likelihood with respect to parameters and updating
            dLL_dtheta = X.T @ (y - predictions)

            self.theta += lr_t * dLL_dtheta

            # checking if the log likelihood function ihas converged
            if abs(current_LL - previous_LL) <= self.tol:
                print(f"The maximum log likelihood was reached at epoch {i}")
                break

            previous_LL = current_LL

    # defining the method to predict continuous probabilities
    def predict_proba(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        return self.sigmoid(X @ self.theta)

    # defining method to predict discrete values of 0 and 1
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    # defining method for model performance evaluation
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
