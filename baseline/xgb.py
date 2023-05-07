#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing 
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

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from optsprime.datasets.data_preprocess import custom_preprocessing
import xgboost as xgb

data_path="data/wlx/CVPA_preprocess_drop_branch_code.csv"
X_all,Y_all = custom_preprocessing(data_path)
X_all=X_all.values.astype(np.int64)
Y_all=Y_all.values.astype(np.int64)
x_train,x_testval,y_train,y_testval=train_test_split(X_all,Y_all,test_size=0.2,random_state=6)
x_val,x_test,y_val,y_test=train_test_split(x_testval,y_testval,test_size=0.5,random_state=6)

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
from optsprime.utils.utils import Evaluater

def objective(space):
    clf=xgb.XGBClassifier(max_depth=2)
    eval=Evaluater("xgb_tiaozheng")
    clf.fit(x_train,y_train)
    pred_prob = clf.predict_proba(x_test)[:,1]
    pred_res= (pred_prob>0.24)*1
    eval.clc_update(y_test,pred_prob,pred_res)
    res=eval.mean()
    return {'loss': -res["auc_mean"],'status': STATUS_OK }
objective(None)