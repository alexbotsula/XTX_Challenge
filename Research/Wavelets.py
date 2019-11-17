#%%
from IPython import get_ipython
import matplotlib.pyplot as plt
import pandas as pd 
from scipy import signal
import numpy as np
from scipy.ndimage.interpolation import shift
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
k = 100 # duration of event in time points
event_edge = np.diff(np.exp(-np.linspace(-2,2,k+1)**2 ))
event_edge = event_edge/np.max(event_edge) # normalize to max=1
plt.plot(event_edge)

#%%
data = pd.read_pickle('~/Documents/AlgoTrading/XTX Challenge/Model scripts/data_train_dct_pca.pkl')
data.head()

#%%
x = data['askRate0'].values
plt.plot(x)
convres = signal.convolve(x, event_edge, 'same')
convres[:50] = 0
convres[-50:] = 0
plt.figure()
plt.plot(convres)

#%%
plt.hist(convres[50:-50], int(convres.size/200))
plt.title('Histogram of convolution result')
plt.show()

#%%
t_cond = np.where(convres > 20)[0]
t_cond2 = np.where(convres < -20)[0]
plt.plot(t_cond, x[t_cond], 'o')
plt.plot(t_cond2, x[t_cond2], 'o')
plt.plot(x)
plt.show()

#%%
t_cond2.shape

#%%
data['askRate0_conv1_max'] = pd.Series(shift(convres,  k//2)).rolling(100, min_periods=1).max()
data['askRate0_conv1_min'] = pd.Series(shift(convres,  k//2)).rolling(100, min_periods=1).min()
plt.plot(data['askRate0_conv1_max'])

print(np.corrcoef(data['askRate0_conv1_max'], data['y']))
print(np.corrcoef(data['askRate0'], data['y']))


#%%
pd.Series([1,2,3,4]).rolling(2, min_periods=1).max()