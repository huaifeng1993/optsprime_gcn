import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import normalize

def custom_preprocessing(data_path):
    train_df=pd.read_csv(data_path)
    label_df=train_df["overdue7"]
    feature_df=train_df.drop("overdue7",axis=1)
    # 对数据归一化
    feature_df=(feature_df-feature_df.min())/(feature_df.max()-feature_df.min())
    #feature_df=normalize(feature_df.to_numpy())
    return feature_df,label_df