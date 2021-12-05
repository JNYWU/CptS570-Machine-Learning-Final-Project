#%%

import DataPreprocess
import numpy as np
    
data = DataPreprocess.ReadData()

X_train, y_train, X_valid, y_valid, X_test, y_test = DataPreprocess.SplitData(data)


# %%
