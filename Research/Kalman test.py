# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_rows = 4000
pd.options.display.max_seq_items = 2000
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('/home/alex/Documents/AlgoTrading/XTX Challenge/XTXStarterKit-master/data.csv')
data.head()

#%%
#process_data = np.array(data[['bidSize1', 'askSize1']])
process_data = np.transpose(np.array([data['bidSize1'] * data['bidRate1'], data['askSize1'] * data['askRate1']]))
process_data.shape

#%%
# Kalman test for volumes
import sys
np.set_printoptions(threshold=sys.maxsize)
data['bidSize1_14'] = np.sum(data[['bidSize{}'.format(i) for i in range(1,15)]], axis=1)
data['askSize1_14'] = np.sum(data[['askSize{}'.format(i) for i in range(1,15)]], axis=1)


#%%

process_data = np.array(data[['bidSize1_14', 'bidSize0', 'askSize0', 'askSize1_14']])
N_states = 11                                                                       # number of states
xhat = np.zeros((process_data.shape[0], N_states))                                  # a posteri estimate of x
P = np.identity(N_states)                                                           # a posteriori error estimate
xhatminus = np.zeros((process_data.shape[0], N_states))                             # a priori estimate of x
Pminus = np.identity(N_states)                                                      # a priori error estimate
K = np.zeros((N_states, process_data.shape[1]))                                     # gain or blending factor
Q = np.identity(N_states) * 1e-3                                                    # process variance
R = 1                                                                               # estimate of measurement variance

A = np.identity(N_states)
A[0:4, 4:11] = [
    [1, 0, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 0, 1, -1],
    [0, 0, 1, 0, 1, 0,  1],
    [0, 0, 0, 1, -1, 0, 0]
]

H = np.zeros((process_data.shape[1], N_states))
H[0:4, 0:4] = np.identity(4)

xhat[0, 0:4] = process_data[0, :]

for k in range(1, process_data.shape[0]):
    # time update
    xhatminus[k, :] = A @ xhat[k-1, :]
    Pminus = A @ P @ np.transpose(A) + Q

    # measurement update
    K = Pminus @ np.transpose(H) @ np.linalg.inv(H @ Pminus @ np.transpose(H) + R)
    xhat[k, :] = xhatminus[k, :] + (K @ (process_data[k, :] - H @ xhatminus[k, :]))
    P = (np.identity(N_states) - K @ H) @ Pminus



#%%
plt.figure()
plt.plot(xhat[:, 1][9900:10000])
plt.plot(process_data[:, 1][9900:10000])

plt.figure()
plt.plot(xhat[:, 2][9900:10000])
plt.plot(process_data[:, 2][9900:10000])

plt.figure()
plt.plot(xhat[:, 10][9900:10000])

plt.figure()
plt.plot(xhat[:, 5][9900:10000])
plt.plot(xhat[:, 6][9900:10000])

#%%
plt.figure()
plt.plot(xhat[:, 10])

#%%
print(np.corrcoef(data['bidSize0'], data['y']))

for i in range(xhat.shape[1]):
    print('{}********'.format(i))
    print(np.corrcoef(xhat[:,i], data['y']))

#%%
