from C45 import *
import numpy as np

class C5(C45):
    def __init__(self,n=10):
        super().__init__()
        self.__tree = []
        self.__ntrees =n
        self.__useful_features = []
        self.__tree_weight =[]
        self.__classes =None
    def Boost(self,X,Y):
        #create n tree and give them a weight
        n =len(Y)
        weights = np.ones(n)/n
        least_error = np.inf
        self.__classes = np.unique(Y)
        k =len(self.__classes)
        for i in range(self.__ntrees):
            indices = np.random.choice(n,size=n,replace=True,p=weights)
            X_test = X[indices].tolist()
            Y_test = Y[indices].tolist()
            model =C45()
            model.fit(X_test,Y_test)
            preds = np.array(model.predict(X))
            wrong = (preds != Y)
            error = np.sum(weights[wrong])
            if error == 0:
                tree_weight = 1
                self.__tree.append(model)
                self.__tree_weight.append(tree_weight)
                break
            elif error >= ((k-1)/k):
                break
            tree_weight = np.log(k-1)+np.log((1-error)/error)
            if tree_weight <= 0:
                break
            weights[wrong] *= np.exp(tree_weight)
            weights[~wrong] *= np.exp(-tree_weight)
            weights /= np.sum(weights)

            self.__tree.append(model)
            self.__tree_weight.append(tree_weight)
    def winnowing(self,X,Y,min_gain_ratio=0.01,keep_fractio =0.5):
        #drop useless features
        self._unique_vals = {
            i: np.unique(X[:, i])
            for i in range(X.shape[1])
        }
        scores =[]
        for i in range(X.shape[1]):
            _,score = self.find_split_val(X,Y,i)
            scores.append((i,score))
        scores.sort(key=lambda s:s[1],reverse=True)
        kept =[idx for idx,score in scores if score >=min_gain_ratio]
        if not kept:
            n_features =X.shape[1]
            top_k =max(1,int(n_features*keep_fractio))
            kept = [idx for idx,score in scores[:top_k]]
        return kept
    def fit(self,X,Y):
        #boost and drop useless features
        X = np.array(X)
        Y = np.array(Y)
        self.__useful_features = self.winnowing(X,Y)
        reduced_X = np.delete(X,[i for i in range(X.shape[1]) if i not in self.__useful_features],axis=1)
        self.Boost(reduced_X,Y)
    def predict(self,X):
        #predict on the baises on which the vote each tree recived
        X = np.array(X)
        reduced_X = np.delete(X, [i for i in range(X.shape[1]) if i not in self.__useful_features], axis=1)
        n =len(reduced_X)
        class_votes ={c: np.zeros(n) for c in self.__classes}
        for model,weight in zip(self.__tree,self.__tree_weight):
            preds = model.predict(reduced_X)
            for idx,p in enumerate(preds):
                class_votes[p][idx] += weight
        final_preds =[]
        for i in range(n):
            best_class =max(self.__classes,key=lambda c:class_votes[c][i])
            final_preds.append(int(best_class))
        return final_preds
    def check_accuracy(self, X, z):
        pred = self.predict(X)
        correct = sum(int(p) == int(actual) for p, actual in zip(pred, z))
        return (correct / len(X)) * 100



