import numpy as np
from IB3 import *
class C45(ID3):
    #inherited everything form ID3 only diff that this uses gain ratio
    def split_info(self,y,current_y):
        #how many times was the data split penalize for more splits
        split_info = 0
        for i in current_y:
            ratio = len(i)/len(y)
            if ratio > 0:
                split_info += -ratio*np.log2(ratio)
        return split_info
    def gain_ratio(self,y,current_y):
        #calculate the gain ratio
        split_info = self.split_info(y,current_y)
        gain_info = self.info_gain(y,current_y)
        if split_info == 0:
            return 0
        return gain_info/split_info

    def find_split_val(self,X,Y, split_atr):
        best_info_gain = 0
        col_split = self._unique_vals[split_atr]
        best_split_value = col_split[0]
        for i in col_split:
            if i is None or (isinstance(i, float) and np.isnan(i)):
                continue
            X_right, Y_right, X_left, Y_left = self.divide(X,Y,i, split_atr)
            current_y = []
            current_y.append(Y_left)
            current_y.append(Y_right)
            info_gain = self.gain_ratio(Y, current_y)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_value = i
        return best_split_value, best_info_gain

    def find_split_atr(self,X,Y):
        best_info_gain = 0
        best_split_atr = 0
        for i in range(X.shape[1]):
            split_val, info_gain = self.find_split_val(X,Y,i)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_atr = i
        return best_split_atr, best_info_gain
