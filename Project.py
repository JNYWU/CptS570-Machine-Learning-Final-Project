#%%

import DataPreprocess
import numpy as np
import pandas as pd
    
data = DataPreprocess.ReadData()

X_train, X_test, y_train, y_test = DataPreprocess.SplitData(data)


# %%
