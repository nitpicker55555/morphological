import json

import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,classification_report,f1_score
from sklearn.metrics import ConfusionMatrixDisplay
df_100m=pd.read_csv("plus_time.csv")
X = df_100m.copy().drop(columns=['Function',	'label_func',	'id_block4b','Unnamed: 0','Unnamed: 0.1','Unnamed: 0.3','Unnamed: 0.2'])


y = df_100m[['label_func']]

df_100m2=pd.read_csv("plus_time_without_weight.csv")
X2 = df_100m2.copy().drop(columns=['Function',	'label_func',	'id_block4b','Unnamed: 0','Unnamed: 0.1','Unnamed: 0.3','Unnamed: 0.2','nid'])


y2 = df_100m2[['label_func']]
def group_data(shap_values):

    result_list=[]
    result_list.append( shap_values.iloc[:, 0:45])
    result_list.append( shap_values.iloc[:, 45:47])
    result_list.append( shap_values.iloc[:, 47:51])
    result_list.append( shap_values.iloc[:, 51:63])
    result_list.append( shap_values.iloc[:, 63:65])
    result_list.append( shap_values.iloc[:, 65:89])
    result_list.append( shap_values.iloc[:, 89:113])
    return result_list
# Choose hyperparameter domain to search over
space = {
        'max_depth':hp.choice('max_depth', np.arange(1, 31, 1, dtype=int)),
        #'n_estimators':hp.choice('n_estimators', np.arange(40, 2001, 10, dtype=int)),
        'colsample_bytree':hp.quniform('colsample_bytree', 0.3, 1.01, 0.1),
        'min_child_weight':hp.choice('min_child_weight', np.arange(10, 51, 10, dtype=int)),
        'subsample':        hp.quniform('subsample', 0.4, 1.01, 0.1),
        'learning_rate':    hp.choice('learning_rate',    np.arange(0.02, 0.3, 0.02)),
        'gamma': hp.quniform('gamma', 0, 1, 0.1),

        'objective':'multi:softmax',
        'tree_method' : 'hist',
        'device' : 'gpu',
        'eval_metric': 'mlogloss',
        'num_class':8
    }


def inner_cv(X_temp, y_temp,n_evals=1000):

    def score(params, n_folds=5):

              #Cross-validation
        # d_train = xgboost.DMatrix(X_temp,y_temp)

        # cv_results = xgboost.cv(params, d_train, nfold = n_folds,num_boost_round=500,
        #                 early_stopping_rounds = 10, metrics = 'rmse', seed = 0)

        # loss = min(cv_results['test-rmse-mean'])

        params['objective'] = 'multi:softmax'
        params['eval_metric'] = 'mlogloss'
        params['num_class'] = 8


        d_train = xgboost.DMatrix(X_temp, y_temp)

        cv_results = xgboost.cv(params, d_train, nfold=n_folds, num_boost_round=500,
                                early_stopping_rounds=10, metrics='mlogloss', seed=0)


        loss = min(cv_results['test-mlogloss-mean'])
        return loss


    def optimize(trials, space):

        best = fmin(score, space, algo=tpe.suggest, max_evals=n_evals,
                    rstate=np.random.default_rng(42))#Add seed to fmin function
        return best

    trials = Trials()
    best_params = optimize(trials, space)

    # Return the best parameters
    best_params = space_eval(space, best_params)

    print(best_params)
    return best_params
def each_iter(X,y,col_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    d_train = xgboost.DMatrix(X_train, label=y_train.label_func)
    d_test = xgboost.DMatrix(X_test, label=y_test.label_func)

    best_params = inner_cv(X_train, y_train, n_evals=50)
    # best_params = joblib.load('best_params_file.joblib')

    evals_result = {}

    model = xgboost.train(best_params, d_train, num_boost_round=5000, evals=[(d_train, "train"), (d_test, "test")],
                          verbose_eval=50, early_stopping_rounds=10, evals_result=evals_result)

    y_pred = model.predict(d_test)

    mae = mean_absolute_error(y_test, y_pred)


    r2 = r2_score(y_test, y_pred)


    rmse = np.sqrt(mean_squared_error(y_test, y_pred))


    accuracy = accuracy_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred, average='macro')  # 二分类问题
    # 对于多分类问题，可以设置 average 参数为 'macro', 'micro', 'weighted' 或者 'samples'

    result_dict={'col_name':col_name,'MAE':mae,'R²':r2,'RMSE':rmse,'accuracy':accuracy,'recall':recall}

    return result_dict


from sklearn.model_selection import train_test_split

X_merged=group_data(X)
X_merged2=group_data(X2)
grouped_label=['Morphology','OD','Human mobility','Social economy','Latitude & longitude','Road speed holiday','Road speed weekday']

score=each_iter(X_merged[0],y,grouped_label[0])

with open('result_score_with_weight2.jsonl','a',encoding='utf-8') as file:
    file.write(json.dumps(score)+'\n')
print(grouped_label[0],score)
def combine_iloc(X_merged,part1,part2):
    return pd.concat([X_merged[part1], X_merged[part2]], axis=1)
for i in range(6):
    score=each_iter(combine_iloc(X_merged,0,i+1),y,grouped_label[i+1])

    with open('result_score_with_weight2.jsonl','a',encoding='utf-8') as file:
        file.write(json.dumps(score)+'\n')
    print(i,score)


for i in range(2):
    score=each_iter(combine_iloc(X_merged2,0,i+5),y,grouped_label[i+5])

    with open('result_score_with_weight3.jsonl','a',encoding='utf-8') as file:
        file.write(json.dumps(score)+'\n')
    print(i,score)
