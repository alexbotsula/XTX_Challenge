#%%
import pandas as pd
import numpy as np

#%%
data = pd.read_csv('/home/alex/Documents/AlgoTrading/XTX Challenge/data_trans.csv')

#%%
ask_size_cols = ['askSize{}'.format(i) for i in range(15)]
bid_size_cols = ['bidSize{}'.format(i) for i in range(15)]

ask_size_change = data[ask_size_cols] - data[ask_size_cols].shift()
bid_size_change = data[bid_size_cols] - data[bid_size_cols].shift()

ask_size_change_pos = ask_size_change > 0
ask_size_change_pos.insert(15, 'dummy', True)
ask_size_change_neg = ask_size_change < 0
ask_size_change_neg.insert(15, 'dummy', True)

bid_size_change_pos = bid_size_change > 0
bid_size_change_pos.insert(15, 'dummy', True)
bid_size_change_neg = bid_size_change < 0
bid_size_change_neg.insert(15, 'dummy', True)

def min_elem(mask):
    mask_num = mask.values * list(range(16)) + \
        np.logical_not(mask.values) * 100
    return np.min(mask_num, axis=1)

data['askChangeFirstPos'] = min_elem(ask_size_change_pos)
data['askChangeFirstNeg'] = min_elem(ask_size_change_neg)
data['bidChangeFirstPos'] = min_elem(bid_size_change_pos)
data['bidChangeFirstNeg'] = min_elem(bid_size_change_neg)
data['bidChangePosBeforeNeg'] = data['bidChangeFirstPos'] < data['bidChangeFirstNeg']
data['askChangePosBeforeNeg'] = data['askChangeFirstPos'] < data['askChangeFirstNeg']



#%%
ttt = ask_size_change_pos.values * list(range(16)) + \
    np.logical_not(ask_size_change_pos.values) * 100