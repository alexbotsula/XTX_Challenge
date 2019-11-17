import pandas as pd
import numpy as np
from scipy.fftpack import dct, idct


class DCTAttributeAdder:
    def __init__(self, window_size, n_coef_perc, columns):
        self._window_size = window_size
        self._n_coef = int(window_size * n_coef_perc)
        #Pre-populate column names of the results
        self.columns_ = columns
        self._res_columns = ['{}_dct_mw_{}_{}'.format(col, self._window_size, self._n_coef) for col in self.columns_] + \
                            ['{}_dct_mw_{}_{}_delta'.format(col, self._window_size, self._n_coef) for col in self.columns_]

    def dct_window(self, x):
        y = dct(x, norm='ortho')
        window = np.zeros(y.shape)
        window[:,:min(self._n_coef, x.shape[1])] = 1
        yr = idct(y * window, norm='ortho')
        return yr[:,-1]

    def dct(self, X):
        vals = X[-1, :]
        res = self.dct_window(np.transpose(X[-self._window_size:, :]))
        deltas = [res[i] - vals[i] for i in range(len(res))]

        X_df_res = pd.DataFrame(data=np.array([list(res) + deltas]), columns=self._res_columns)

        return X_df_res