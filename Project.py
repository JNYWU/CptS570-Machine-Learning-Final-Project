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

def DecisionTree(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    
    # fit the model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)
    
    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
    
    return accuracy, importance.importances_mean

def NaiveBayes(X_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    
    # fit the model
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)

    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
    
    return accuracy, importance.importances_mean

def LogisticRegression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    
    # fit the model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)

    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)

    return accuracy, importance.importances_mean

def RandomForest(X_train, y_train):
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
    minHeader = []
    
    for i in range(topN):
        minIndex = np.argmin(tempImportance)
        minHeader.append(header[minIndex])   
        tempImportance[minIndex] = 10
        
    return minHeader

def Plot(header, importance, title):
    plt.figure()
    plt.xticks(rotation=90, ha='center')
    plt.title(title)
    plt.bar(header, importance)
    plt.savefig(title, bbox_inches='tight')
    
def DropColumns(X_train, y_train, dropCols):
    droppedData = data.drop(dropCols, axis = 1)
    dropped_X_train, dropped_y_train, X_test, y_test, droppedHeader = DataPreprocess.SplitData(droppedData)
    
    return dropped_X_train, dropped_y_train, droppedHeader

if __name__ == "__main__":
    print("Happy Machine Learning")
    
    # decision tree
    decisionTreeAccuracy, decisionTreeImportance = DecisionTree(X_train, y_train)
    print("Decision Tree Accuracy: ", decisionTreeAccuracy)
    Plot(header, decisionTreeImportance, "Decision Tree Importance")
    
    # random forest
    randomForestAccuracy, randomForestImportance = RandomForest(X_train, y_train)
    print("Random Forest Accuracy: ", randomForestAccuracy)
    Plot(header, randomForestImportance, "Random Forest Importance")

    # naive Bayes
    naiveBayesAccuracy, naiveBayesImportance = NaiveBayes(X_train, y_train)
    print("Naive Bayes Accuracy: ", naiveBayesAccuracy)
    Plot(header, naiveBayesImportance, "Naive Bayes")

    # LogisticRegression
    logisticRegressionAccuracy, logisticRegressionImportance = LogisticRegression(X_train, y_train)
    print("Logistic Regression Accuracy: ", logisticRegressionAccuracy)
    Plot(header, logisticRegressionImportance, "Logistic Regression Importance")
    
    # combine the importances
    combinedImportance = decisionTreeImportance + randomForestImportance + naiveBayesImportance + logisticRegressionImportance
    Plot(header, combinedImportance, "Combined Importance")
    
    
    dropHeader = FindMinIndex(10, combinedImportance, header)
    dropped_X_train, dropped_y_train, droppedHeader = DropColumns(X_train, y_train, dropHeader)
    
    
    
# %%
