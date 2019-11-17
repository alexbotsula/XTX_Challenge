#%%
from IPython import get_ipython
import pandas as pd
import numpy as np

#%%
model_data = pd.read_csv('/home/alex/Documents/AlgoTrading/XTX Challenge/data_trans.csv')
process_data = pd.read_csv('/home/alex/Documents/AlgoTrading/XTX Challenge/XTXStarterKit-master/python/dct_results.csv')

#%%
model_data.head()
process_data.head()

#%%
for column in list(model_data)[1:]:
    if column == 'y':
        continue
    c_md = model_data[column].values[1:]
    c_pd = process_data[column].values[1:]
    n_diff = sum(np.not_equal(c_md, c_pd))
    if n_diff > 0:
        print('{}: incorrect {}'.format(column, n_diff))

#%%

np.sum(process_data['sizeFlow8'] - model_data['sizeFlow8'])
# print(process_data['sizeFlow7'][:100])
# print(model_data['sizeFlow7'][:100])

#%%
tt = np.array([1, 2, 3, 4])
ind = np.where(tt==2)[0].tolist() + np.where(tt==1)[0].tolist()
tt[ind]