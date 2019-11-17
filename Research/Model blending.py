#%%
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.svm import SVR

#%%
# Read model files
xgb1 = pickle.load(open("/home/alex/Documents/AlgoTrading/XTX Challenge/XTXStarterKit-master/python/core_models/xgb1_small.pkl", "rb"))
regr1 = pickle.load(open("/home/alex/Documents/AlgoTrading/XTX Challenge/XTXStarterKit-master/python/core_models/regr1.pkl", "rb"))
xgb1_columns = pickle.load(open("/home/alex/Documents/AlgoTrading/XTX Challenge/XTXStarterKit-master/python/core_models/xgb1_columns.pkl", "rb"))
regr1_columns = pickle.load(open("/home/alex/Documents/AlgoTrading/XTX Challenge/XTXStarterKit-master/python/core_models/regr1_columns.pkl", "rb"))
models = [xgb1, regr1]


#%%
# Read source data
data = pd.read_csv('/home/alex/Documents/AlgoTrading/XTX Challenge/data_trans.csv').iloc[1:,:]

#%%
# Produce forecast of each model
n_splits = 3

def model_fcast_kfold(dtrain, model, predictors, n_splits):
    train_pred = np.empty((0,1), float)
    kf_split = KFold(n_splits=n_splits)

    for (train_index, test_index) in kf_split.split(dtrain):
        model.fit(dtrain.iloc[train_index][predictors], dtrain.iloc[train_index]['y'])
        test_predictions = model.predict(dtrain.iloc[test_index][predictors])
        train_pred = np.append(train_pred, test_predictions)
    
    return train_pred

stack_df = pd.DataFrame()
stack_df['y'] = data['y']
stack_df['xgb_pred'] = model_fcast_kfold(data, xgb1, xgb1_columns, n_splits)
stack_df['regr_pred'] = model_fcast_kfold(data, regr1, regr1_columns, n_splits)

#%%
def model_wf_cv(alg, dtrain, predictors, target, n_splits):
    tscv = KFold(n_splits=n_splits) #TimeSeriesSplit
    cv_scores_test = np.zeros((n_splits, 1))
    cv_scores_train = np.zeros((n_splits, 1))
    cv_scores_r2_test = np.zeros((n_splits, 1))
    
    for i, (train_index, test_index) in enumerate(tscv.split(dtrain)):
        
        alg.fit(dtrain.iloc[train_index][predictors], dtrain.iloc[train_index][target])
        
        test_predictions = alg.predict(dtrain.iloc[test_index][predictors])
        train_predictions = alg.predict(dtrain.iloc[train_index][predictors])
        
        cv_scores_test[i, 0] = np.sqrt(metrics.mean_squared_error(dtrain.iloc[test_index][target].values, 
                                                             test_predictions))
        cv_scores_train[i, 0] = np.sqrt(metrics.mean_squared_error(dtrain.iloc[train_index][target].values, 
                                                             train_predictions))
        cv_scores_r2_test[i, 0] = metrics.r2_score(dtrain.iloc[test_index][target].values, 
                                                             test_predictions)
        
    return np.mean(cv_scores_train), np.mean(cv_scores_test), np.mean(cv_scores_r2_test)

def grid_search_rmse(alg, dtrain, predictors, target, n_splits, parameters):
    best_score = np.inf
    for g in ParameterGrid(parameters):
        print(g)
        alg.set_params(**g)
        wf_cv_score_train, wf_cv_score_test, wf_cv_score_test_r2 = model_wf_cv(alg, dtrain, predictors, target, n_splits)
        if(wf_cv_score_test < best_score):
            best_score = wf_cv_score_test
            best_grid = g
        print("\tCV score test: %f (R2 %f)\tCV score train: %f"%(wf_cv_score_test, wf_cv_score_test_r2, wf_cv_score_train))
    return best_score, best_grid

#%% 
# Build blending model - SVR
predictors = ['xgb_pred', 'regr_pred']

svr = SVR(
    kernel=’rbf’
)

params_svr = {
    'gamma': [0.1, 1, 10], 
    'C': [0.1, 1, 10]
}

print(grid_search_rmse(copy.deepcopy(svr), stack_df, predictors, 'y', 3, params_svr))
