
import os
import pandas as pd
import numpy as np
import joblib
import xgboost
import shap
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,classification_report,f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
df_100m=pd.read_csv("plus_time.csv")
X = df_100m.copy().drop(columns=['Function',	'label_func',	'id_block4b','Unnamed: 0','Unnamed: 0.1','Unnamed: 0.3','Unnamed: 0.2'])

X_names=X.columns.tolist()
y = df_100m[['label_func']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

d_train = xgboost.DMatrix(X_train, label=y_train.label_func)
d_test = xgboost.DMatrix(X_test, label=y_test.label_func)


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

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
        'tree_method' : 'gpu_hist',
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
                    rstate=np.random.default_rng(42) )#Add seed to fmin function
        return best

    trials = Trials()
    best_params = optimize(trials, space)

    # Return the best parameters
    best_params = space_eval(space, best_params)

    # print(best_params)
    return best_params
def accuracy(best_params):


    evals_result = {}

    model = xgboost.train(best_params, d_train, num_boost_round=5000, evals=[(d_train, "train"), (d_test, "test")],
                          verbose_eval=50, early_stopping_rounds=20, evals_result=evals_result)
    y_pred = model.predict(d_test)
    # y_true = [...]  # 真实的标签
    # y_pred = [...]  # 模型预测的标签

    # 计算总体精度

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    # recall = recall_score(y_test, y_pred, average='macro')  # 二分类问题
    # 对于多分类问题，可以设置 average 参数为 'macro', 'micro', 'weighted' 或者 'samples'

    return accuracy

accuracy_value=0
for i in range(50):
    print(i,'====')
    best_params = inner_cv(X_train, y_train, n_evals=100)
    new_acc=accuracy(best_params)
    if new_acc>accuracy_value:
        joblib.dump(best_params, f'best_params_file{str(new_acc).replace(".","")[:5]}.joblib')
        accuracy_value=new_acc
        print('new acc',accuracy_value)



# 保存参数到文件


# best_params = joblib.load('best_params_file.joblib')