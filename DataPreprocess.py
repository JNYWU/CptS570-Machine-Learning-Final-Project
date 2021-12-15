#%%

def ReadData():
    import pandas as pd
        
    data = pd.read_csv('high_diamond_ranked_10min.csv')
    data.head()    

    return data


def SplitData(data):
    # data length = 9879
    # first 70% as training data
    # last 30% as testing data
    
    # set Blue Wins as y
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    
    dropCols = ['gameId', 'blueWins', 'redGoldPerMin', 'blueGoldPerMin', 'redCSPerMin', 'blueCSPerMin']
    cleanData = data.drop(dropCols, axis = 1)
    
    X = cleanData
    y = data['blueWins']
    
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, y_train, X_test, y_test
    
# %%
