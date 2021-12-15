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

def DecisionTree(X_train, X_test):
    from sklearn.tree import DecisionTreeClassifier
    
    # fit the model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)
    
    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
    
    return accuracy, importance.importances_mean

def NaiveBayes(X_train, X_test):
    from sklearn.naive_bayes import GaussianNB
    
    # fit the model
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)

    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
    
    return accuracy, importance.importances_mean

def LogisticRegression(X_train, X_test):
    from sklearn.linear_model import LogisticRegression
    
    # fit the model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)

    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)

    return accuracy, importance.importances_mean

def RandomForest(X_train, X_test):
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
    plt.bar(header, importance, sorted(importance, reverse=True))
    plt.savefig(title, bbox_inches='tight')
    
def DropColumns(X_train, y_train, dropCols):
    droppedData = data.drop(dropCols, axis = 1)
    dropped_X_train, y_train, dropped_X_test, y_test, droppedHeader = DataPreprocess.SplitData(droppedData)
    
    return dropped_X_train, dropped_X_test, droppedHeader

if __name__ == "__main__":
    print("Happy Machine Learning")
    
    initialAccuracy = []
    
    # decision tree
    decisionTreeAccuracy, decisionTreeImportance = DecisionTree(X_train, X_test)
    print("Decision Tree Accuracy: ", decisionTreeAccuracy)
    initialAccuracy.append(decisionTreeAccuracy)
    Plot(header, decisionTreeImportance, "Decision Tree Importance")
    
    # random forest
    randomForestAccuracy, randomForestImportance = RandomForest(X_train, X_test)
    print("Random Forest Accuracy: ", randomForestAccuracy)
    initialAccuracy.append(randomForestAccuracy)
    Plot(header, randomForestImportance, "Random Forest Importance")

    # naive Bayes
    naiveBayesAccuracy, naiveBayesImportance = NaiveBayes(X_train, X_test)
    print("Naive Bayes Accuracy: ", naiveBayesAccuracy)
    initialAccuracy.append(naiveBayesAccuracy)
    Plot(header, naiveBayesImportance, "Naive Bayes")

    # LogisticRegression
    logisticRegressionAccuracy, logisticRegressionImportance = LogisticRegression(X_train, X_test)
    print("Logistic Regression Accuracy: ", logisticRegressionAccuracy)
    initialAccuracy.append(logisticRegressionAccuracy)
    Plot(header, logisticRegressionImportance, "Logistic Regression Importance")
    
    # plot accuracy
    xAxis = ["Decision Tree", "Random Forest", "Naive Bayes", "Logistic Regression"]
    plt.figure()
    plt.xticks(rotation=90, ha='center')
    plt.title("Initial Accuracy")
    plt.bar(xAxis, initialAccuracy)
    plt.savefig("Initial Accuracy", bbox_inches='tight')
    
    # combine the importances
    combinedImportance = decisionTreeImportance + randomForestImportance + naiveBayesImportance + logisticRegressionImportance
    Plot(header, combinedImportance, "Combined Importance")
    
    # drop the 10 least important feature
    dropHeader = FindMinIndex(10, combinedImportance, header)
    dropped_X_train, dropped_X_test, droppedHeader = DropColumns(X_train, y_train, dropHeader)
    
    droppedAccuracy = []
    
    # decision tree
    decisionTreeAccuracy, decisionTreeImportance = DecisionTree(dropped_X_train, dropped_X_test)
    print("Dropped Decision Tree Accuracy: ", decisionTreeAccuracy)
    droppedAccuracy.append(decisionTreeAccuracy)
    Plot(droppedHeader, decisionTreeImportance, "Dropped Decision Tree Importance")
    
    # random forest
    randomForestAccuracy, randomForestImportance = RandomForest(dropped_X_train, dropped_X_test)
    print("Dropped Random Forest Accuracy: ", randomForestAccuracy)
    droppedAccuracy.append(randomForestAccuracy)
    Plot(droppedHeader, randomForestImportance, "Dropped Random Forest Importance")

    # naive Bayes
    naiveBayesAccuracy, naiveBayesImportance = NaiveBayes(dropped_X_train, dropped_X_test)
    print("Dropped Naive Bayes Accuracy: ", naiveBayesAccuracy)
    droppedAccuracy.append(naiveBayesAccuracy)
    Plot(droppedHeader, naiveBayesImportance, "Dropped Naive Bayes")

    # LogisticRegression
    logisticRegressionAccuracy, logisticRegressionImportance = LogisticRegression(dropped_X_train, dropped_X_test)
    print("Dropped Logistic Regression Accuracy: ", logisticRegressionAccuracy)
    droppedAccuracy.append(logisticRegressionAccuracy)
    Plot(droppedHeader, logisticRegressionImportance, "Dropped Logistic Regression Importance")
    
    # combine the importances
    combinedImportance = decisionTreeImportance + randomForestImportance + naiveBayesImportance + logisticRegressionImportance
    Plot(droppedHeader, combinedImportance, "Dropped Combined Importance")
    
    
    # plot dropped accuracy
    plt.figure()
    plt.xticks(rotation=90, ha='center')
    plt.title("Dropped Accuracy")
    plt.bar(xAxis, droppedAccuracy)
    plt.savefig("Dropped Accuracy", bbox_inches='tight')
    
    
# %%
