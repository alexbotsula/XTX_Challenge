#%%
import pandas as pd
import numpy as np
import pickle
import copy 
from sklearn.model_selection import StratifiedKFold



#%%
X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test))
