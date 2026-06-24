import numpy as np
from main import *

class LogisticRegression(Alogrothm):
    def __init__(self):
        self.__weights = None
        self.__intercept = None
        self.__coef = None
        self.__classes = None
    def softmax(self,z):
        #make sure the values dont skyrocket to inf and give a prob to all categrories
        exp_z = np.exp(z - np.max(z ,axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    def one_hot(self,z, n_classes):
        #encodes the whole categrories into into diiferent array
        one_hot = np.zeros((len(z),n_classes))
        one_hot[np.arange(len(z)),z] = 1
        return one_hot
    def fit(self,X,z):
        #creates random weights
        X_b = np.hstack((X,np.ones((X.shape[0],1))))
        n_features = X_b.shape[1]
        n_classes = len(np.unique(z))
        self.__weights = np.random.randn(n_features, n_classes)*0.01
        self.__classes = np.unique(z)
        z_encoded = self.one_hot(z,n_classes)
        #tryies to find the correct weight where the loss is minimum using gradient decent
        for i in range(1000):
            z_scores= X_b @ self.__weights
            z_pred = self.softmax(z_scores)
            gradient = X_b.T @ (z_pred - z_encoded)
            learning_rate = 0.01
            self.__weights -= learning_rate * gradient
    def calculate(self,X):
        X_b = np.hstack((X, np.ones((X.shape[0], 1))))
        z_scores = X_b @ self.__weights
        z_pred = self.softmax(z_scores)
        return z_pred
    def predict(self,X):
        z_pred = self.calculate(X)
        indices = np.argmax(z_pred, axis=1)
        return self.__classes[indices]

    def check_accuracy(self, X, z):
        z_pred = self.calculate(X)
        z_encoded = self.one_hot(z,len(self.__classes))
        loss = -np.mean(z_encoded * np.log(z_pred))

        z_pred = self.predict(X)
        accuracy = np.mean(z_pred == z)
        return accuracy, loss
    def get_coef(self):
        self.__coef = self.__weights[:-1]
        return self.__coef
    def get_intercept(self):
        self.__intercept = self.__weights[-1]
        return self.__intercept