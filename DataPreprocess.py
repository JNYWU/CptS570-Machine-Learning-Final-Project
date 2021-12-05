#%%

def ReadData():
    import csv
    import numpy as np
        
    with open('high_diamond_ranked_10min.csv', 'r') as f:
        data = list(csv.reader(f))
    
    data = np.array(data)
    return data


def SplitData(data):
    header = data[0]
    
    # data length = 9880
    # first 60% as training data
    # next 10% as validation data
    # last 30% as testing data
    
    # column 1 is Blue Wins, set a y
    
    X_train = data[1:5928, 2:]
    y_train = data[1:5928, 1]
    
    X_valid = data[5928:6916, 2:]
    y_valid = data[5928:6916, 1]
    
    X_test = data[6916:, 2:]
    y_test = data[6916:, 1]
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test
    
# %%
