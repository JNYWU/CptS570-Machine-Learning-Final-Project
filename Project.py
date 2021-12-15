#%%

import DataPreprocess
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
    
data = DataPreprocess.ReadData()

X_train, y_train, X_test, y_test = DataPreprocess.SplitData(data)

def DecisionTree():
    from sklearn.tree import DecisionTreeClassifier
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)
    
    importance = clf.feature_importances_
    
    return accuracy, importance

if __name__ == "__main__":
    print("Happy Machine Learning")
    
    # decision tree
    decisionTreeAccuracy, decisionTreeImportance = DecisionTree()
    print("Decision Tree Accuracy: ", decisionTreeAccuracy)
    plt.bar([x for x in range(len(decisionTreeImportance))], decisionTreeImportance)
# %%
