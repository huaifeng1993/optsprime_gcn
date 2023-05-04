import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

data_path="../data/wlx/CFPA数据.csv"
data=pd.read_csv(data_path,index_col="index")