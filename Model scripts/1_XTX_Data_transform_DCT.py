import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy import spatial, stats

from sklearn.impute import SimpleImputer
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from multiprocessing import set_start_method
import multiprocessing as mp
import os

try:
    set_start_method("forkserver")
except:
    pass

os.environ["OMP_NUM_THREADS"] = "8"

class DCTAttributeAdder:
    def __init__(self, window_size, n_coef_perc, colnames):
        self._colnames = colnames
        self._window_size = window_size
        self._n_coef = int(window_size * n_coef_perc)
        self._colnames_dct = None
      
    def dct_rolling(self, x):
        return x.rolling(self._window_size, min_periods=1).apply(lambda x: self.dct_window(x, self._n_coef), raw=True)

    @staticmethod
    def dct_window(x, n_coef):
        y = dct(x, norm='ortho')
        window = np.zeros(x.shape[0])
        window[:min(n_coef, len(x))] = 1
        yr = idct(y * window, norm='ortho')
        return yr[-1] 
    
    def transform(self, X):
        X_df = pd.DataFrame()#X
        p = mp.Pool(mp.cpu_count())
        res = p.map(self.dct_rolling, [X[col] for col in self._colnames])
        p.close()

        for i, col in enumerate(np.array(self._colnames)):
            X_df['{}_dct_mw_{}_{}'.format(col, self._window_size, self._n_coef)] = res[i]
            X_df['{}_dct_mw_{}_{}_delta'.format(col, self._window_size, self._n_coef)] = X_df['{}_dct_mw_{}_{}'.format(col, self._window_size, self._n_coef)].values - X[col].values
    
        return X_df      

if __name__ == '__main__':
    model_variables = ['askRate0', 'bidRate0', 'askSize0', 'bidSize0', 'askSize1', 'bidSize1']
    print('Reading source data...')
    data_training = pd.read_csv('/mnt/disks/disk1/data-training.csv')#.iloc[:20000,:]
    #data_training = pd.read_csv('/home/alex/Documents/AlgoTrading/XTX Challenge/XTXStarterKit-master/data.csv')
    pickle.dump(list(data_training), open('./data_transform_options/data_columns.pkl', 'wb'))

    # Impute
    print('Variables...')
    # Impute all sizes (xxxSizeX) with zeros
    imp_zero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    size_cols = data_training.columns.str.contains('Size')
    rate_cols = data_training.columns.str.contains('Rate')
    rate_colnames = data_training.columns[rate_cols]

    data_training.loc[:, size_cols] = imp_zero.fit_transform(data_training.loc[:, size_cols])

    # Impute all rates (xxxRateX) with mean value
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_training.loc[:, rate_cols] = imp_mean.fit_transform(data_training.loc[:, rate_cols])
    joblib.dump(imp_mean, './data_transform_options/mean_imputer.pkl')
    
    # PCA for rates
    rates_PCA = PCA(n_components=2)
    rates_PCA.fit(data_training.loc[:, rate_cols])
    pcas = rates_PCA.transform(data_training.loc[:, rate_cols])
    data_training['ratePCA1'] = pcas[:,0]
    data_training['ratePCA2'] = pcas[:,1]
    model_variables += ['ratePCA1', 'ratePCA2']
    joblib.dump(rates_PCA, './data_transform_options/rates_PCA.pkl') 

    # Create additional order book variables
    ask_rate = data_training[['askRate{}'.format(i) for i in range(15)]].values
    bid_rate = data_training[['bidRate{}'.format(i) for i in range(15)]].values
    ask_size = data_training[['askSize{}'.format(i) for i in range(15)]].values
    bid_size = data_training[['bidSize{}'.format(i) for i in range(15)]].values
    ask_vol = ask_rate * ask_size
    bid_vol = bid_rate * bid_size
    ask_vol_cumm = np.cumsum(ask_vol, axis=1)
    bid_vol_cumm = np.cumsum(bid_vol, axis=1)
    ask_size_cumm = np.cumsum(ask_size, axis=1)
    bid_size_cumm = np.cumsum(bid_size, axis=1)
    
    # % comulative sum +/- X$ rate - bid and ask
    rate_thresh_list = [1, 2, 3, 4]
    bid_rate_diff = bid_rate[:, 0, None] - bid_rate
    ask_rate_diff = ask_rate - ask_rate[:, 0, None]
    for rate_thresh in rate_thresh_list:
        ask_vol_cumm_flt = ask_vol_cumm * (ask_rate_diff <= rate_thresh) 
        bid_vol_cumm_flt = bid_vol_cumm * (bid_rate_diff <= rate_thresh) 

        data_training['askCummVolPerc{}'.format(rate_thresh)] = np.max(ask_vol_cumm_flt, axis=1) / ask_vol_cumm[:,14]
        data_training['bidCummVolPerc{}'.format(rate_thresh)] = np.max(bid_vol_cumm_flt, axis=1) / bid_vol_cumm[:,14]
        data_training['totalCummVolPerc{}'.format(rate_thresh)] = (np.max(ask_vol_cumm_flt, axis=1) + np.max(bid_vol_cumm_flt, axis=1)) / (bid_vol_cumm[:,14] + ask_vol_cumm[:,14])
        data_training['deltaCummVol{}'.format(rate_thresh)] = np.max(ask_vol_cumm_flt, axis=1) - np.max(bid_vol_cumm_flt, axis=1)
        model_variables += ['askCummVolPerc{}'.format(rate_thresh), 'bidCummVolPerc{}'.format(rate_thresh), 'totalCummVolPerc{}'.format(rate_thresh), 'deltaCummVol{}'.format(rate_thresh)]

    # Spread of rates under 20% of cumm volume - bid and ask
    perc_thresh_list = [0.1, 0.2, 0.4]
    ask_vol_cumm_perc = ask_vol_cumm / ask_vol_cumm[:, 14, None]
    bid_vol_cumm_perc = bid_vol_cumm / bid_vol_cumm[:, 14, None]
    for perc_thresh in perc_thresh_list: 
        data_training['askRateMax{}'.format(perc_thresh)] = np.min(ask_rate * (ask_vol_cumm_perc > perc_thresh) + 1e6 * (ask_vol_cumm_perc <= perc_thresh), axis=1)
        data_training['bidRateMin{}'.format(perc_thresh)] = np.max(bid_rate * (bid_vol_cumm_perc > perc_thresh), axis=1)
        data_training['bidAskSpread{}'.format(perc_thresh)] = data_training['askRateMax{}'.format(perc_thresh)] - data_training['bidRateMin{}'.format(perc_thresh)]
        model_variables += ['askRateMax{}'.format(perc_thresh), 'bidRateMin{}'.format(perc_thresh), 'bidAskSpread{}'.format(perc_thresh)]
        
    # Slope of cumm Size vs. Bid and Ask Rate
    ask_size_cumm_perc = ask_size_cumm / ask_size_cumm[:, 14, None]
    bid_size_cumm_perc = bid_size_cumm / bid_size_cumm[:, 14, None]
    data_training['askSizeSlope'] = [stats.linregress(ask_rate[i, :], ask_size_cumm_perc[i, :]).slope for i in range(ask_rate.shape[0])]
    data_training['bidSizeSlope'] = [stats.linregress(bid_rate[i, :], bid_size_cumm_perc[i, :]).slope for i in range(bid_rate.shape[0])]
    data_training['bidAskSlopeDiff'] = data_training['askSizeSlope'] + data_training['bidSizeSlope']
    model_variables += ['askSizeSlope', 'bidSizeSlope', 'bidAskSlopeDiff']
    
    # Create Bid-Ask Volume columns
    for i in range(6):
        data_training['bidAskVolume{}'.format(i)] = data_training['bidRate{}'.format(i)] * data_training['bidSize{}'.format(i)] - data_training['askRate{}'.format(i)] * data_training['askSize{}'.format(i)]
        model_variables += ['bidAskVolume{}'.format(i)]

    # Order imbalance
    df1 = pd.DataFrame((ask_size_cumm - bid_size_cumm) / (ask_size_cumm + bid_size_cumm + 1), 
                columns=['imbalance{}'.format(i) for i in range(15)])
    data_training = pd.concat([data_training, df1], axis=1)
    model_variables += ['imbalance{}'.format(i) for i in range(15)]
    
    # Delete intermediary data
    del ask_size_cumm_perc, bid_size_cumm_perc, ask_vol_cumm_perc, bid_vol_cumm_perc, ask_rate, bid_rate, ask_size, bid_size, ask_vol, bid_vol, ask_vol_cumm, bid_vol_cumm, ask_size_cumm, bid_size_cumm
  
    # Run DCT
    dct_rates_100_02 =  DCTAttributeAdder(window_size=100, n_coef_perc=0.02, colnames=model_variables)
    dct_rates_1e3_01 =  DCTAttributeAdder(window_size=int(1e3), n_coef_perc=0.01, colnames=model_variables)
    dct_rates_1e3_005 = DCTAttributeAdder(window_size=int(1e3), n_coef_perc=0.005, colnames=model_variables)
    
    columns = np.array(list(data_training))
    rate_cols = ['askRate{}'.format(i) for i in range(2, 15)] + \
        ['bidRate{}'.format(i) for i in range(2, 15)]
    columns = columns[[col not in rate_cols for col in columns]]
    data_training = data_training[columns]

    print('Running DCT...')
    print('\tStep1...')
    data_training1 = dct_rates_100_02.transform(data_training)
    print('\tStep2...')
    data_training2 = dct_rates_1e3_01.transform(data_training)
    print('\tStep3...')
    data_training3 = dct_rates_1e3_005.transform(data_training)
    
    
    print('\tMerging...')
    data_training = pd.concat([data_training, 
                               data_training1], axis=1)
    del data_training1
    data_training = pd.concat([data_training, 
                               data_training2], axis=1)
    del data_training2
    data_training = pd.concat([data_training, 
                               data_training3], axis=1)
    del data_training3
    
    # Kalman-based flows
    print('Flows...')
    data_training['bidSize1_14'] = np.sum(data_training[['bidSize{}'.format(i) for i in range(1,15)]], axis=1)
    data_training['askSize1_14'] = np.sum(data_training[['askSize{}'.format(i) for i in range(1,15)]], axis=1)
    
    process_data = np.array(data_training[['bidSize1_14', 'bidSize0', 'askSize0', 'askSize1_14']])
    N_states = 12                                                                       # number of states
    xhat = np.zeros((process_data.shape[0], N_states))                                  # a posteri estimate of x
    P = np.identity(N_states)                                                           # a posteriori error estimate
    xhatminus = np.zeros((process_data.shape[0], N_states))                             # a priori estimate of x
    Pminus = np.identity(N_states)                                                      # a priori error estimate
    K = np.zeros((N_states, process_data.shape[1]))                                     # gain or blending factor
    Q = np.identity(N_states) * 1e-3                                                    # process variance
    R = 1                                                                               # estimate of measurement variance

    A = np.identity(N_states)

    A[0:4, 4:12] = [
        [1, 0, 0, 0, 0, -1, 0, -1],
        [0, 1, 0, 0, 0, 1, -1, 0],
        [0, 0, 1, 0, 1, 0,  1, 0],
        [0, 0, 0, 1, -1, 0, 0, 1]
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
    
    for i in range(1,9):
        data_training['sizeFlow{}'.format(i)] = xhat[:,i+3]
    
    # Delta 
    print('Lagged delta...')
    data_training[['{}_delta1'.format(pred) for pred in model_variables]] = data_training[model_variables] - data_training[model_variables].shift()
    
    print('Saving results...')

    data_training.to_pickle('/mnt/disks/disk1/data_train_1709.pkl')
    # data_training.to_csv('/home/alex/Documents/AlgoTrading/XTX Challenge/data_trans.csv')
    print('Results saved successfuly!')

