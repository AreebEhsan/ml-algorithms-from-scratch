import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    return corr**2

# Linear Regression class
class LinearRegression:
    def __init__(self, learning_rate = 0.001, iter = 1000):
        self.learning_rate = learning_rate
        self.iter = iter
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iter):
            y_predicted = np.dot(x, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x):
        y_predicted = np.dot(x, self.weights) + self.bias
        return y_predicted


if __name__ == "__main__":
    x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    from LinearRegression import LinearRegression

    regressor = LinearRegression(learning_rate=0.01)
    regressor.fit(x_train, y_train)
    predicted = regressor.predict(x_test)


    def mse(y_true, y_predicted):
        return np.mean((y_true - y_predicted) ** 2)


    mse_value = mse(y_test, predicted)
    print(mse_value)

    y_pred_line = regressor.predict(x)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
    plt.plot(x, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.show()
