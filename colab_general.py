import h5py
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
import xgboost
from reverse_shap_values import reload_shap
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,classification_report,f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from colab_csv import generate_csv
from colab_pdf import draw_summary
from colab_single_pdf import draw_single
def generate_shap_values(seed):
    df_100m = pd.read_csv("plus_time_without_weight.csv")
    X = df_100m.copy().drop(
        columns=['Function', 'label_func', 'id_block4b', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.3', 'Unnamed: 0.2',
                 'nid'])
    X_names = X.columns.tolist()
    y = df_100m[['label_func']]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    d_train = xgboost.DMatrix(X_train, label=y_train.label_func)
    d_test = xgboost.DMatrix(X_test, label=y_test.label_func)

    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

    # Choose hyperparameter domain to search over
    space = {
        'max_depth': hp.choice('max_depth', np.arange(1, 31, 1, dtype=int)),
        # 'n_estimators':hp.choice('n_estimators', np.arange(40, 2001, 10, dtype=int)),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 1.01, 0.1),
        'min_child_weight': hp.choice('min_child_weight', np.arange(10, 51, 10, dtype=int)),
        'subsample': hp.quniform('subsample', 0.4, 1.01, 0.1),
        'learning_rate': hp.choice('learning_rate', np.arange(0.02, 0.3, 0.02)),
        'gamma': hp.quniform('gamma', 0, 1, 0.1),

        'objective': 'multi:softmax',
        'tree_method': 'hist',
        'eval_metric': 'mlogloss',
        'num_class': 8
    }

    def inner_cv(X_temp, y_temp, n_evals=1000):
        def score(params, n_folds=5):
            # Cross-validation
            # d_train = xgboost.DMatrix(X_temp,y_temp)

            # cv_results = xgboost.cv(params, d_train, nfold = n_folds,num_boost_round=500,
            #                 early_stopping_rounds = 10, metrics = 'rmse', seed = 0)

            # loss = min(cv_results['test-rmse-mean'])

            params['objective'] = 'multi:softmax'
            params['eval_metric'] = 'mlogloss'
            params['num_class'] = 8

            d_train = xgboost.DMatrix(X_temp, y_temp)

            cv_results = xgboost.cv(params, d_train, nfold=n_folds, num_boost_round=500,
                                    early_stopping_rounds=10, metrics='mlogloss', seed=seed)

            loss = min(cv_results['test-mlogloss-mean'])
            return loss

        def optimize(trials, space, seed_value=seed):
            best = fmin(score, space, algo=tpe.suggest, max_evals=n_evals,
                        trials=trials, rstate=np.random.RandomState(seed_value))
            return best

        trials = Trials()
        best_params = optimize(trials, space)

        # Return the best parameters
        best_params = space_eval(space, best_params)

        print(best_params)
        return best_params

    import joblib

    # 保存参数到文件
    # joblib.dump(best_params, 'best_params_file.joblib')
    # best_params = joblib.load('best_params_file09052.joblib')
    best_params = joblib.load('best_params_file09052.joblib')
    import xgboost as xgb
    import shap
    from sklearn.preprocessing import LabelEncoder

    # 初始化标签编码器
    label_encoder = LabelEncoder()

    # 对原始类标签进行编码，使其从0开始且连续
    y_encoded = label_encoder.fit_transform(y)
    model = xgb.XGBClassifier(**best_params)
    # 现在使用编码后的标签来拟合模型
    model.fit(X, y_encoded)

    # 计算 SHAP 值
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X, check_additivity=False)
    return shap_values,df_100m,X_names,X
shap_values_value=reload_shap()
shap_values, df_100m, X_names,X=generate_shap_values(0)
shap_values.values=shap_values_value
draw_single(shap_values,X)
draw_summary(shap_values,X)
