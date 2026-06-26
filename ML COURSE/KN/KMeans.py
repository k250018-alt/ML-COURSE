from main import *
import numpy as np
from scipy import stats
from sklearn.metrics import silhouette_score
class KMeans(Alogrothm):
    def __init__(self,model):
        #intialize all the values
        self.__k = 0
        self.__Xtrain = None
        self.__centroid = None
        self.__model = model
        self.__cluster = {}
        self.__mean =0.0
        self.__std =0.0
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
    def createkgroups(self):
        #take random points and with the help of those point divide the data using the proximisity of each point to the random point
        self.__cluster={i:[] for i in range(self.__k)}
        if self.__model == 1:
            for i in range(len(self.__Xtrain)):
                distances = self.encludian(self.__centroid, self.__Xtrain[i])
                closest = int(np.argmin(distances))
                self.__cluster[closest].append(i)
        if self.__model == 2:
            for i in range(len(self.__Xtrain)):
                distances = self.manhattan(self.__centroid, self.__Xtrain[i])
                closest = int(np.argmin(distances))
                self.__cluster[closest].append(i)
        if self.__model == 3:
            for i in range(len(self.__Xtrain)):
                distances = self.hamming(self.__centroid, self.__Xtrain[i])
                closest = int(np.argmin(distances))
                self.__cluster[closest].append(i)
        if self.__model == 4:
            for i in range(len(self.__Xtrain)):
                distances = self.cosine(self.__centroid, self.__Xtrain[i])
                closest = int(np.argmin(distances))
                self.__cluster[closest].append(i)
    def find_actuall_centroid(self):
        #find actuall centroids using the formula
        for i, group in self.__cluster.items():
            if len(group) == 0:
                continue
            points = self.__Xtrain[group]
            if self.__model == 1:
                 self.__centroid[i] = points.mean(axis=0)
            if self.__model == 2:
                self.__centroid[i] = np.median(points,axis=0)
            if self.__model == 3:
                result = np.array(stats.mode(points, axis=0), dtype=float).flatten()
                self.__centroid[i] = result[:points.shape[1]]
            if self.__model == 4:
                 temp = np.mean(points,axis=0)
                 self.__centroid[i] = temp/np.linalg.norm(temp)

    def standardize(self, X):
        # to scale xtrain so number is given higher priority
        return (X - self.__mean) / self.__std

    def normalize(self, X):
        # normalizeing the data to seperate it more and to make clearer prediction
        magnitude = np.linalg.norm(X, axis=1, keepdims=True)
        magnitude = np.where(magnitude == 0, 1, magnitude)
        return X / magnitude

    def fit(self,X,z):
        self.__k = z
        self.__Xtrain = X
        self.__mean = np.mean(self.__Xtrain, axis=0)
        self.__std = np.std(self.__Xtrain, axis=0)
        self.__Xtrain = self.standardize(self.__Xtrain)
        self.__Xtrain = self.normalize(self.__Xtrain)
        indices = np.random.choice(len(self.__Xtrain), self.__k, replace=False)
        self.__centroid = self.__Xtrain[indices].copy()
        for _ in range(100):
            #loop until find the best centroid
            old_centroids = self.__centroid.copy() if self.__centroid is not None else None
            self.createkgroups()
            self.find_actuall_centroid()
            if old_centroids is not None:
                if np.allclose(old_centroids, self.__centroid):
                    break
    def predict(self,X):
        X = self.standardize(X)
        X = self.normalize(X.reshape(1, -1))[0]
        if self.__model == 1:
            distances = self.encludian(self.__centroid, X)
            closest = int(np.argmin(distances))
            return closest
        if self.__model == 2:
            distances = self.manhattan(self.__centroid, X)
            closest = int(np.argmin(distances))
            return closest
        if self.__model == 3:
            distances = self.hamming(self.__centroid, X)
            closest = int(np.argmin(distances))
            return closest
        if self.__model == 4:
            distances = self.cosine(self.__centroid, X)
            closest = int(np.argmin(distances))
            return closest

    def check_accuracy(self, X, z):
        X_scaled = self.standardize(X)
        X_scaled = self.normalize(X_scaled)
        labels =[]
        for i in range(len(X_scaled)):
            if self.__model == 1:
                distances = self.encludian(self.__centroid, X_scaled[i])
            elif self.__model == 2:
                distances = self.manhattan(self.__centroid, X_scaled[i])
            elif self.__model == 3:
                distances = self.hamming(self.__centroid, X_scaled[i])
            elif self.__model == 4:
                distances = self.cosine(self.__centroid, X_scaled[i])
            labels.append(int(np.argmin(distances)))
        labels = np.array(labels)

        # inertia (WCSS) — sum of squared distances to assigned centroid
        inertia = 0
        for cluster_id, indices in self.__cluster.items():
            if len(indices) == 0:
                continue
            points = X_scaled[indices]
            centroid = self.__centroid[cluster_id]
            inertia += np.sum((points - centroid) ** 2)

        # silhouette score — needs at least 2 clusters with points
        if len(set(labels)) > 1:
            sil_score = silhouette_score(X_scaled, labels)
        else:
            sil_score = -1
        return sil_score, inertia