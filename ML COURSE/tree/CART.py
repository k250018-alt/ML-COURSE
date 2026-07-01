import numpy as np
from IB3 import *

class Cart(ID3):
    def __init__(self,alpha,min_split,depth):
        super().__init__(depth,min_split)
        self.__alpha = alpha
    def get_Gini(self,Y):
        #calculate gini
        if len(Y)== 0:
            return 0
        count = np.bincount(Y)
        probs = count/len(Y)
        return 1-np.sum(probs**2)
    def get_weighted_gini(self,Y_left,Y_right):
        #get the weighted gini
        n = len(Y_left)+len(Y_right)
        if n == 0:
            return 0
        return ((len(Y_left)/n)*self.get_Gini(Y_left)+(len(Y_right)/n)*self.get_Gini(Y_right))

    def find_split_val(self, X, Y, split_atr):
        # divide based on the best value where you gain the most info
        parent_gini= self.get_Gini(Y)
        best_gain =0
        col_split = self._unique_vals[split_atr]
        best_split_value = col_split[0]
        for i in col_split:
            if i is None or (isinstance(i, float) and np.isnan(i)):
                continue
            X_right, Y_right, X_left, Y_left = self.divide(X, Y, i, split_atr)
            gini = self.get_weighted_gini(Y_left,Y_right)
            gain = parent_gini - gini
            if gain> best_gain:
                best_gain = gain
                best_split_value = i
        return best_split_value, best_gain
    def find_split_atr(self,X,Y):
        #divide based on the best atr where you gain tthe most info
        best_gain = 0
        best_split_atr = 0
        for i in range(X.shape[1]):
            split_val,gain= self.find_split_val(X,Y,i)
            if gain > best_gain:
                best_gain = gain
                best_split_atr = i
        return best_split_atr,best_gain
    def count_leaves(self,tree):
        #count the leaves in the tree
        if tree["Leaf"] :
            return 1
        return (self.count_leaves(tree["Left"]) + self.count_leaves(tree["Right"]))
    def tree_error(self,tree,X,Y):
        #how much error is in tree
        if len(Y) == 0:
            return 0
        preds =np.array([self.predictone(x,tree) for x in X])
        return np.sum(preds!=Y)
    def prune(self,X,Y,tree):
        #collapse useless bracnches
        if tree["Leaf"] :
            return tree
        X =np.array(X)
        Y =np.array(Y)
        col = X[:,tree["Atr"]]
        if isinstance(tree["Val"],(int,float,np.integer,np.floating)):
            mask = col > tree["Val"]
        else:
            mask = col == tree["Val"]
        X_Right ,Y_Right = X[mask],Y[mask]
        X_Left ,Y_Left = X[~mask],Y[~mask]
        tree["Left"] = self.prune(X_Left,Y_Left,tree["Left"])
        tree["Right"] = self.prune(X_Right,Y_Right,tree["Right"])
        error_subtree =self.tree_error(tree,X,Y)
        leaves_subtree =self.count_leaves(tree)
        cost_subtree = error_subtree + self.__alpha * leaves_subtree
        majority =self.majority_class(Y)
        collapsed ={"Leaf":True,"Class":majority }
        error_leaf =self.tree_error(collapsed,X,Y)
        cost_leaf =error_leaf + self.__alpha * 1
        if cost_leaf <= cost_subtree:
            return collapsed
        return tree
    def fit(self,X,Y):
        #prune the tree after fitting
        super().fit(X,Y)
        self._tree = self.prune(X,Y,self._tree)