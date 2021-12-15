#%%

import DataPreprocess
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance



data = DataPreprocess.ReadData()

X_train, y_train, X_test, y_test = DataPreprocess.SplitData(data)

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
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)

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

if __name__ == "__main__":
    print("Happy Machine Learning")
    
    # decision tree
    decisionTreeAccuracy, decisionTreeImportance = DecisionTree()
    print("Decision Tree Accuracy: ", decisionTreeAccuracy)
    plt.bar([x for x in range(len(decisionTreeImportance))], decisionTreeImportance)
    plt.show()
    
    # random forest
    randomForestAccuracy, randomForestImportance = RandomForest()
    print("Random Forest Accuracy: ", randomForestAccuracy)
    plt.bar([x for x in range(len(randomForestImportance))], randomForestImportance)
    plt.show()

    # naive Bayes
    naiveBayesAccuracy, naiveBayesImportance = NaiveBayes()
    print("Naive Bayes Accuracy: ", naiveBayesAccuracy)
    plt.bar([x for x in range(len(naiveBayesImportance))], naiveBayesImportance)
    plt.show()

    # LogisticRegression
    logisticRegressionAccuracy, logisticRegressionImportance = LogisticRegression()
    print("Logistic Regression Accuracy: ", logisticRegressionAccuracy)
    plt.bar([x for x in range(len(logisticRegressionImportance))], logisticRegressionImportance)
    plt.show()
# %%
