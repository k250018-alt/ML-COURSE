from main import *
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

class KNNREgression(Alogrothm):
    def __init__(self):
        # initiate the parameters
        self.__k = 0
        self.__model = 0
        self.__Xtrain = []
        self.__Ztrain = []
        self.__mean = 0.0
        self.__std = 0.0
        self.__Xtest = []
        self.__Ztest = []


    def encludian(self, Xtrain, Xtest):
        # calculate distance using euculidian
        distances = []
        for row in range(len(Xtrain)):
            current_X = Xtrain[row]
            current_distance = 0
            for col in range(len(current_X)):
                current_distance += (current_X[col] - Xtest[col]) ** 2
            current_distance = np.sqrt(current_distance)
            distances.append(current_distance)
        return distances


    def manhattan(self, Xtrain, Xtest):
        # calculate distance using manhattan
        distances = []
        for row in range(len(Xtrain)):
            current_X = Xtrain[row]
            current_distance = 0
            for col in range(len(current_X)):
                current_distance += np.absolute(current_X[col] - Xtest[col])
            distances.append(current_distance)
        return distances


    def hamming(self, Xtrain, Xtest):
        # calculate distance using hamming
        distances = []
        for row in range(len(Xtrain)):
            current_X = Xtrain[row]
            diff = sum(a != b for a, b in zip(current_X, Xtest))
            current_distance = diff / len(current_X)
            distances.append(current_distance)
        return distances


    def cosine(self, Xtrain, Xtest):
        # calculate distance using cosine
        distances = []
        for row in range(len(Xtrain)):
            current_X = Xtrain[row]
            dot_product = np.dot(current_X, Xtest)
            current_magnitude = np.linalg.norm(current_X)
            test_magnitude = np.linalg.norm(Xtest)
            if current_magnitude == 0 or test_magnitude == 0:
                distances.append(1.0)
            else:
                cosine_similarity = dot_product / (current_magnitude * test_magnitude)
                cosine_distance = 1 - cosine_similarity
                distances.append(cosine_distance)
        return distances


    def nearest_neighbours(self, distances):
        # what are the k nearest neighbours
        distances = np.array(distances)
        distances_sorted = np.argsort(distances)
        return distances_sorted[:self.__k]


    def averaging(self, nearest_neighbours, Ztrain):
        # what average of the k nearestneibours
        return np.mean(Ztrain[nearest_neighbours])

    def divide_into_folds(self, foldsize, fold, Xtrain, Ztrain):
        # divide the training data into folds
        test_start = fold * foldsize
        test_end = test_start + foldsize
        Xtest = Xtrain[test_start:test_end]
        Ztest = Ztrain[test_start:test_end]
        X_train = np.concatenate((Xtrain[:test_start], Xtrain[test_end:]))
        z_train = np.concatenate((Ztrain[:test_start], Ztrain[test_end:]))
        return X_train, z_train, Xtest, Ztest


    def test(self, Xtest):
        # caluculated using the Best model and k value
        if self.__model == 1:
            distances = self.encludian(self.__Xtrain, Xtest)
            nearest_neighbours = self.nearest_neighbours(distances)
            return self.averaging(nearest_neighbours, self.__Ztrain)
        if self.__model == 2:
            distances = self.manhattan(self.__Xtrain, Xtest)
            nearest_neighbours = self.nearest_neighbours(distances)
            return self.averaging(nearest_neighbours, self.__Ztrain)
        if self.__model == 3:
            distances = self.hamming(self.__Xtrain, Xtest)
            nearest_neighbours = self.nearest_neighbours(distances)
            return self.averaging(nearest_neighbours, self.__Ztrain)
        if self.__model == 4:
            distances = self.cosine(self.__Xtrain, Xtest)
            nearest_neighbours = self.nearest_neighbours(distances)
            return self.averaging(nearest_neighbours, self.__Ztrain)


    def Best(self, Xtrain, Ztrain):
        # finding the best model and k value
        Xtrain = np.array(Xtrain)
        Ztrain = np.array(Ztrain)
        fold_size = int(len(Xtrain) / 5)
        best_model = 1
        best_k = 1
        best_score = 0

        for i in range(1, 5):
            self.__model = i
            for k in range(1, int(np.sqrt(len(Xtrain))) + 1):
                self.__k = k
                scores = []
                for fold in range(5):
                    X_fold_train, Z_fold_train, X_test_fold, Z_test_fold = self.divide_into_folds(
                        fold_size, fold, Xtrain, Ztrain
                    )
                    self.__Xtrain = X_fold_train
                    self.__Ztrain = Z_fold_train

                    predictions = self.predict(X_test_fold)
                    accuracy = np.mean(predictions == Z_test_fold)
                    scores.append(accuracy)

                mean_accuracy = np.mean(scores)
                if mean_accuracy > best_score:
                    best_score = mean_accuracy
                    best_model = i
                    best_k = k

        self.__model = best_model
        self.__k = best_k


    def predict(self, X):
        X = np.array(X)
        predictions = []
        X = self.standardize(X)
        X = self.normalize(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        for point in X:
            pred = self.test(point)
            predictions.append(pred)
        return np.array(predictions)


    def standardize(self, X):
        # to scale xtrain so number is given higher priority
        return (X - self.__mean) / self.__std


    def normalize(self, X):
        # normalizeing the data to seperate it more and to make clearer prediction
        magnitude = np.linalg.norm(X, axis=1, keepdims=True)
        magnitude = np.where(magnitude == 0, 1, magnitude)
        return X / magnitude


    def fit(self, X, z):
        Xtrain, Xtest, ztrain, ztest = train_test_split(
            X, z, test_size=0.2, random_state=0
        )
        self.__Xtrain = Xtrain
        self.__Xtest = Xtest
        self.__Ztrain = ztrain
        self.__Ztest = ztest
        self.__mean = np.mean(Xtrain, axis=0)
        self.__std = np.std(Xtrain, axis=0)
        X_scaled = self.standardize(self.__Xtrain)
        x_test_scaled = self.standardize(self.__Xtest)
        X_normalized = self.normalize(X_scaled)
        x_test_normalized = self.normalize(x_test_scaled)
        self.Best(X_normalized, ztrain)
        self.__Xtrain = np.array(X_normalized)
        self.__Ztrain = np.array(ztrain)
        self.check_accuracy(x_test_normalized, ztest)


    def check_accuracy(self, X, z):
        pred = self.predict(X)
        return np.mean((pred-z)**2)