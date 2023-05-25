#!/usr/bin/env python
# coding: utf-8
from re import X
from numpy.core.fromnumeric import diagonal
from numpy.ma.core import where
from sklearn import metrics
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
np.random.seed(203)
import pandas as pd
#import featuretools as ft
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from optsprime.datasets.data_preprocess import custom_preprocessing

import lightgbm as lgbm
from sklearn import metrics
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from tqdm import tqdm
from optsprime.utils.utils import Evaluater
import random

data_path="data/wlx/CVPA_preprocess_drop_branch_code.csv"
X_all,Y_all = custom_preprocessing(data_path)
feature_names = X_all.columns.to_list()
X_all=X_all.values.astype(np.float32)
Y_all=Y_all.values.astype(np.int64)
num_train=len(Y_all)
print(num_train)
num_test=int(0.1*num_train)
num_val=int(0.1*num_train)
idx=list(range(num_train))
num_train=int(num_train-num_test-num_val)
random.seed(1234)
random.shuffle(idx)
# x_train,x_testval,y_train,y_testval=train_test_split(X_all,Y_all,test_size=0.2,random_state=6)
# x_val,x_test,y_val,y_test=train_test_split(x_testval,y_testval,test_size=0.5,random_state=6)
x_train=X_all[idx[:num_train]]
y_train=Y_all[idx[:num_train]]
x_val=X_all[idx[num_train:num_train+num_val]]
y_val=Y_all[idx[num_train:num_train+num_val]]
print(np.sum(y_val))
x_test=X_all[idx[num_train+num_val:]]
y_test=Y_all[idx[num_train+num_val:]]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def log_loss(y_true,y_pred):
    y_sigmoid=sigmoid(y_pred)
    grad=y_sigmoid-y_true
    hess=y_sigmoid*(1-y_sigmoid)
    return grad,hess

def log_eval(y_true,y_pred):
    y_sigmoid=sigmoid(y_pred)
    #p = np.clip(y_sigmoid, eps, 1-eps)
    loss=-y_true * np.log(y_sigmoid) - (1 - y_true) * np.log(1-y_sigmoid)
    return "metric_eval", np.mean(loss), False

def objective_cross(space):
    global X_all,Y_all
    X=X_all.values.astype(np.int64)
    Y=Y_all.values.astype(np.int64)
    clf= lgbm.LGBMClassifier(num_leaves=10)
    kf=StratifiedKFold(n_splits=10)
    dataloader=tqdm(kf.split(X,Y),total=10)
    eval=Evaluater("lgbm_tiaozheng")
    
    for k,(train_index, test_index) in enumerate(dataloader):
        evaluation = [(X[train_index], Y[train_index]), ( X[test_index], Y[test_index])]
        clf.fit(evaluation[0][0], evaluation[0][1], eval_set=evaluation, eval_metric="acc", early_stopping_rounds=10,verbose=False)
        pred_prob = clf.predict_proba(evaluation[1][0])[:,1]
        pred_res= clf.predict(evaluation[1][0])
        eval.clc_update(evaluation[1][1],pred_prob,pred_res)
    res=eval.mean()
    return {'loss': -res["auc_mean"],'status': STATUS_OK }

def objective(space):
    clf= lgbm.LGBMClassifier(num_leaves=20)
    eval=Evaluater("lgbm")
    clf.fit(x_train, y_train, eval_set=[(x_val,y_val)],verbose=False)
    pred_prob = clf.predict_proba(x_test)[:,1]
    pred_res= (pred_prob>0.25)*1
    eval.clc_update(y_test,pred_prob,pred_res)
    res=eval.mean()
    feature_importance = clf.feature_importances_

    # 获取前10个特征的索引和重要性得分
    top10_indices = np.argsort(feature_importance)[-10:]
    top10_scores = feature_importance[top10_indices]
        
    top10_features = [feature_names[i] for i in top10_indices]
    print(top10_features)
    plt.figure()
    plt.barh(range(len(top10_scores)), top10_scores, align='center')
    plt.yticks(range(len(top10_scores)),top10_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('lgbm Feature Importance')
    plt.savefig("lgbm_feature_importance.png")
    return {'loss': -res["auc_mean"],"space":space,'status': STATUS_OK }
objective(None)
# space={'max_depth': hp.quniform("max_depth", 0, 2, 1)}

# trials = Trials()
# best_hyperparams = fmin(fn = objective,
#                         space = space,
#                         algo = tpe.suggest,
#                         max_evals = 5,
#                         trials = trials)
# print("The best hyperparameters are : ","\n")
# print(best_hyperparams)