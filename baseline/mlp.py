#!/usr/bin/env python
# coding: utf-8
from re import X
from tkinter import Scale
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
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold
from optsprime.datasets.data_preprocess import custom_preprocessing

import lightgbm as lgbm
from sklearn import metrics
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from tqdm import tqdm
from optsprime.utils.utils import Evaluater
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_path="data/wlx/CVPA_preprocess_drop_branch_code.csv"
X_all,Y_all = custom_preprocessing(data_path)

X_all=X_all.values.astype(np.float32)
X_all=scaler.fit_transform(X_all)
Y_all=Y_all.values.astype(np.int64)

x_train,x_testval,y_train,y_testval=train_test_split(X_all,Y_all,test_size=0.2,random_state=6)
x_val,x_test,y_val,y_test=train_test_split(x_testval,y_testval,test_size=0.5,random_state=6)

def objective(space):
    clf= MLPClassifier()
    eval=Evaluater("mlp")
    clf.fit(x_train, y_train)
    pred_prob = clf.predict_proba(x_test)[:,1]
    pred_res= clf.predict(x_test)
    eval.clc_update(y_test,pred_prob,pred_res)
    res=eval.mean()
    return {'loss': -res["auc_mean"],"space":space,'status': STATUS_OK }
objective(None)
