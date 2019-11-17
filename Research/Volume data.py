# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython
import pandas as pd
import matplotlib.pyplot as plt
import numpy
pd.options.display.max_rows = 4000
pd.options.display.max_seq_items = 2000
get_ipython().run_line_magic('matplotlib', 'inline')



#%%
data = pd.read_pickle('~/Documents/AlgoTrading/XTX Challenge/Model scripts/data_train_dct_pca.pkl')
data.head()


#%%

for i in range(15):
    data['bidAskVolume{}'.format(i)] = data['bidRate{}'.format(i)]*data['bidSize{}'.format(i)] - data['askRate{}'.format(i)]*data['askSize{}'.format(i)]


#%%
volume_cols = ['bidAskVolume{}'.format(i) for i in range(15)]

max_vol = np.argmax(data[volume_cols].values, axis=1)
min_vol = np.argmin(data[volume_cols].values, axis=1)

#%%
plt.plot(max_vol)
plt.plot(min_vol)

#%%
from scipy.ndimage.interpolation import shift
from scipy.signal import correlate
t1 = data['y']
t2 = data['bidAskVolume0']
corr = correlate(t1, t2, mode='full')
plt.plot(corr[corr.size//2:])
lag = np.argmax(np.abs(corr[corr.size//2:]))
print(np.corrcoef(t1, t2))
print(np.corrcoef(t1, shift(t2, lag)))

#%%
from scipy.ndimage.interpolation import shift
shift([1,2,3], 2)


#%%
from scipy.fftpack import fft, ifft
from sklearn.model_selection import train_test_split, TimeSeriesSplit
t1 = data['y']
t2 = data['bidAskVolume0']
t1_fft = np.fft.fft(t1)
t2_fft = np.fft.fft(t2)
t1_fft.shape
conv = np.fft.ifft(t1_fft * t2_fft)
plt.plot(np.real(conv))
print(np.correlate(t1,t2))
print(np.correlate(np.real(conv), t2))

#%%
from sklearn.decomposition import PCA
rates_PCA = PCA(n_components=2)
tt = rates_PCA.fit_transform(data)

#%%
3//2