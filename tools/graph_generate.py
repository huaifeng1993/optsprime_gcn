import numpy as np
from numpy.linalg import norm
import pandas as pd
# define two lists or array
A = np.array([[2,1,2],
              [3,2,9], 
              [-1,2,-3],
              [2,2,2]
              ])

#根据输入的数据计算余弦相似度并生成index
def graph_generage(A,node_num=10,threshold=0.95):
    #cosine_sim_matrix = np.dot(A, A.T) / (norm(A, axis=1)[:, np.newaxis] * norm(A, axis=1)[np.newaxis, :])
    cosine_sim_matrix = np.zeros((len(A), len(A)))
    edges = []
    # 循环计算余弦相似度,并选择与A[i]相似最大的10个节点生成边
    for i in range(len(A)):
        cosine_sim_matrix[i] = np.dot(A[i], A.T) / (norm(A[i])* norm(A.T, axis=0))
        tmp_cosing_sim = cosine_sim_matrix[i]
        #求tmp_cosing_sim最大的10个值的index
        max_sim_index = np.argsort(tmp_cosing_sim)[-(node_num+1):][:-1]
        # 索引i与索引max_sim_index之间生成边
        tmp_edges=[(i,index) for index in max_sim_index if tmp_cosing_sim[index]>threshold]
        edges.extend(tmp_edges)
    return edges

def create_graph_from_csv(data_path):
    data=pd.read_csv(data_path)
    data=data.drop("branchcode",axis=1)
    data=data.to_numpy()
    edge_index=graph_generage(data,node_num=10,threshold=0.95)
    #保存边的index的csv文件，特征特征名称为dst和src
    pd.DataFrame(edge_index).to_csv("data/wlx/CVPA_preprocess_edge_index.csv",index=False,header=["src","dst"])
    return edge_index

edge_index=create_graph_from_csv("data/wlx/CVPA_preprocess.csv")
print(len(edge_index))
