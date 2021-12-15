#%%

import DataPreprocess
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

data = DataPreprocess.ReadData()

X_train, y_train, X_test, y_test, header = DataPreprocess.SplitData(data)

def DecisionTree():
    from sklearn.tree import DecisionTreeClassifier
    
    # fit the model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)
    
    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
    
    return accuracy, importance.importances_mean

def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    
    # fit the model
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)

    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
    
    return accuracy, importance.importances_mean

def LogisticRegression():
    from sklearn.linear_model import LogisticRegression
    
    # fit the model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)

    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)

    return accuracy, importance.importances_mean

def RandomForest():
    from sklearn.ensemble import RandomForestClassifier
    
    # fit the model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    # get accuract score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)
    
    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
    
    return accuracy, importance.importances_mean

def FindMinIndex(topN, importance, header):
    tempImportance = importance
    minIndexValues = []
    minHeader = []
    
    for i in range(topN):
        minIndex = np.argmin(tempImportance)
        minHeader.append(header[minIndex])   
        tempImportance[minIndex] = 10
        
    return minHeader
    

if __name__ == "__main__":
    print("Happy Machine Learning")
    
    # decision tree
    decisionTreeAccuracy, decisionTreeImportance = DecisionTree()
    print("Decision Tree Accuracy: ", decisionTreeAccuracy)
    plt.xticks(rotation=90, ha='center')
    plt.bar(header, decisionTreeImportance)
    plt.show()
    
    # random forest
    randomForestAccuracy, randomForestImportance = RandomForest()
    print("Random Forest Accuracy: ", randomForestAccuracy)
    plt.xticks(rotation=90, ha='center')
    plt.bar(header, randomForestImportance)
    plt.show()

    # naive Bayes
    naiveBayesAccuracy, naiveBayesImportance = NaiveBayes()
    print("Naive Bayes Accuracy: ", naiveBayesAccuracy)
    plt.xticks(rotation=90, ha='center')
    plt.bar(header, naiveBayesImportance)
    plt.show()

    # LogisticRegression
    logisticRegressionAccuracy, logisticRegressionImportance = LogisticRegression()
    print("Logistic Regression Accuracy: ", logisticRegressionAccuracy)
    plt.xticks(rotation=90, ha='center')
    plt.bar(header, logisticRegressionImportance)
    plt.show()
    
    # combine the importances
    combinedImportance = decisionTreeImportance + randomForestImportance + naiveBayesImportance + logisticRegressionImportance
    plt.xticks(rotation=90, ha='center')
    plt.bar(header, combinedImportance)
    plt.show()
    
    dropHeader = FindMinIndex(10, combinedImportance, header)
    
    
# %%
