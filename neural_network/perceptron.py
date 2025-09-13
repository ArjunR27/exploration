import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron():
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def weighted_sum(self, X):
        weights = self.w[1:]
        bias = self.w[0]
        return np.dot(X, weights) + bias
    
    def predict_single(self, x):
        weighted_sum = self.weighted_sum(x)
        if weighted_sum > 0.0:
            return 1
        else:
            return -1
        
    def predict_examples(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_single(x))
        return predictions
    
    
    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)
    
    def fit(self, X, Y):
        # bias value + num of features set to 0
        self.w = np.zeros(1 + X.shape[1])
        self.errors = []

        print("Weights: ", self.w)

        for _ in range(self.n_iter):
            error = 0

            for xi, y in zip(X, Y):

                y_pred = self.predict_single(xi)

                update = self.eta * (y - y_pred)

                self.w[1:] = self.w[1:] + update * xi
                print("Updated weights: ", self.w[1:])

                if update != 0:
                    error += 1
                else:
                    error += 0
                
            self.errors.append(error)
        
        return self
    
def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df = shuffle(df)

    X = df.iloc[:, 0:4].values
    y = df.iloc[:, 4].values

    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25)
    train_labels = np.where(train_labels == 'Iris-setosa', 1, -1)
    test_labels = np.where(test_labels == 'Iris-setosa', 1, -1)

    print(train_data)

    perceptron = Perceptron()

    perceptron.fit(train_data, train_labels)

    predictions = perceptron.predict_examples(test_data)

    test_accuracy = accuracy_score(predictions, test_labels)
    print("Accuracy on test data: ", round(test_accuracy, 2) * 100, "%")

    return

if __name__ == "__main__":
    main()