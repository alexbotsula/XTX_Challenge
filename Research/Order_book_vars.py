

#%%
from IPython import get_ipython

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial, stats
pd.options.display.max_rows = 4000
pd.options.display.max_seq_items = 2000
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('/home/alex/Documents/AlgoTrading/XTX Challenge/XTXStarterKit-master/data.csv')
data.head()

#%%

ask_rate = data[['askRate{}'.format(i) for i in range(15)]].values
bid_rate = data[['bidRate{}'.format(i) for i in range(15)]].values
ask_size = data[['askSize{}'.format(i) for i in range(15)]].values
bid_size = data[['bidSize{}'.format(i) for i in range(15)]].values
ask_vol = ask_rate * ask_size
bid_vol = bid_rate * bid_size
ask_vol_cumm = np.cumsum(ask_vol, axis=1)
bid_vol_cumm = np.cumsum(bid_vol, axis=1)
ask_size_cumm = np.cumsum(ask_size, axis=1)
bid_size_cumm = np.cumsum(bid_size, axis=1)

plt.plot(np.mean(bid_vol, axis=0)/np.mean(bid_size, axis=0), np.mean(bid_vol_cumm, axis=0))
plt.plot(np.mean(ask_vol, axis=0)/np.mean(ask_size, axis=0), np.mean(ask_vol_cumm, axis=0))

#%%
# Cosine distance between bid and ask volume vectors
bid_ask_cosine = [spatial.distance.cosine(ask_vol_cumm[i,:], bid_vol_cumm[i, :]) for i in range(ask_vol_cumm.shape[0])]
plt.plot(bid_ask_cosine)

#%%
# % comulative sum +/- X$ rate - bid and ask
rate_thresh = 2
bid_rate_diff = bid_rate[:, 0, None] - bid_rate
ask_rate_diff = ask_rate - ask_rate[:, 0, None]

ask_vol_cumm_flt = ask_vol_cumm * (ask_rate_diff <= rate_thresh) 
bid_vol_cumm_flt = bid_vol_cumm * (bid_rate_diff <= rate_thresh) 

ask_cumm_vol_perc = np.max(ask_vol_cumm_flt, axis=1) / ask_vol_cumm[:,14]
bid_cumm_vol_perc = np.max(bid_vol_cumm_flt, axis=1) / bid_vol_cumm[:,14]
total_cumm_vol_perc = (np.max(ask_vol_cumm_flt, axis=1) + np.max(bid_vol_cumm_flt, axis=1)) / (bid_vol_cumm[:,14] + ask_vol_cumm[:,14])
delta_cumm_volume = np.max(ask_vol_cumm_flt, axis=1) - np.max(bid_vol_cumm_flt, axis=1)


plt.plot(ask_cumm_vol_perc)
plt.plot(bid_cumm_vol_perc)
plt.figure()
plt.plot(total_cumm_vol_perc) 

plt.figure()
plt.plot(delta_cumm_volume)

#%%
# Spread of rates under 20% of cumm volume - bid and ask
perc_thresh = 0.2
ask_vol_cumm_perc = ask_vol_cumm / ask_vol_cumm[:, 14, None]
bid_vol_cumm_perc = bid_vol_cumm / bid_vol_cumm[:, 14, None]

ask_rate_max = np.min(ask_rate * (ask_vol_cumm_perc > perc_thresh) + 1e6 * (ask_vol_cumm_perc <= perc_thresh), axis=1)
bid_rate_min = np.max(bid_rate * (bid_vol_cumm_perc > perc_thresh), axis=1)
bid_ask_spread = ask_rate_max - bid_rate_min

plt.plot(ask_rate_max)
plt.figure()
plt.plot(bid_rate_min)
plt.figure()
plt.plot(bid_ask_spread)


#%%
# Slope of cumm Size and cumm Volume vs. Bid and Ask Rate
ask_vol_cumm_perc  = ask_vol_cumm / ask_vol_cumm[:, 14, None]
bid_vol_cumm_perc = bid_vol_cumm / bid_vol_cumm[:, 14, None] 
ask_size_cumm_perc = ask_size_cumm / ask_size_cumm[:, 14, None]
bid_size_cumm_perc = bid_size_cumm / bid_size_cumm[:, 14, None]


ask_vol_slope = [stats.linregress(ask_rate[i, :], ask_vol_cumm_perc[i, :]).slope for i in range(ask_rate.shape[0])]
bid_vol_slope = [stats.linregress(bid_rate[i, :], bid_vol_cumm_perc[i, :]).slope for i in range(bid_rate.shape[0])]
 

plt.figure()
plt.plot(ask_vol_slope)
plt.figure()
plt.plot(bid_vol_slope)
plt.figure()
plt.plot(ask_size_slope)
plt.figure()
plt.plot(bid_size_slope)
plt.figure()
plt.plot(vol_slope_diff)
plt.figure()
plt.plot(size_slope_diff)


#%%
stats.pearsonr(ask_vol_slope, ask_size_slope)

#%%
