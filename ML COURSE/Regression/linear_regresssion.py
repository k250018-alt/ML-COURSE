from main import *
import numpy as np

class LinearRegression(Alogrothm):
    def __init__(self):
        self.__weights = None
        self.__coef = None
        self.__intercept = None
    def fit(self,X,z):
        #created weights and adjusted them using formula INV(X.T@X)@(X.T@Z)
        X_b = np.hstack((X,np.ones((X.shape[0],1))))
        self.__weights = np.linalg.inv(X_b.T @ X_b)@(X_b.T @ z)
    def predict(self,X):
        #gave a prediction using the trained weights
        X_b = np.hstack((X, np.ones((X.shape[0], 1))))
        z_pred = X_b @ self.__weights
        return z_pred
    def check_accuracy(self,X,z):
        #return the average error between the trained and actual data
        z_pred = self.predict(X)
        MSE = 1 / z.shape[0] * np.sum((z_pred - z) * (z_pred - z))
        return MSE
    def get_coef(self):
        self.__coef = self.__weights[:-1]
        return self.__coef
    def get_intercept(self):
        self.__intercept = self.__weights[-1]
        return self.__intercept