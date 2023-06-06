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
def graph_generage(A,node_num=8,threshold=0.95):
    #cosine_sim_matrix = np.dot(A, A.T) / (norm(A, axis=1)[:, np.newaxis] * norm(A, axis=1)[np.newaxis, :])
    
    edges = []
    edges_weight=[]
    edges_attr=[]
    #计算每个节点与其他节点特征相同的个数，如果大于10个则生成边，并把相同特征个数作为边的权重
    
    for i in range(len(A)):
        #属性相同的feature 置为1，不同的置为0
        tmp_edge_attr=A[i]==A
        sim_matrix=np.sum(A[i]==A[i+1:],axis=1)
        tmp_edge=[[(i,index),sim,tmp_edge_attr[index]] for index,sim in enumerate(sim_matrix) if sim>=node_num]
        edges.extend([eg[0] for eg in tmp_edge])
        edges_weight.extend([eg[1] for eg in tmp_edge])
        edges_attr.extend([eg[2]*1 for eg in tmp_edge])
    return edges,edges_weight,edges_attr

def create_graph_from_csv(data_path):
    data=pd.read_csv(data_path)#[:2000]
    print("data shape:",data.shape)
    data_new=data.drop("branchcode",axis=1)
    data_new.to_csv("data/wlx/CVPA_preprocess_drop_branch_code.csv",index=False)
    int_col=["branchcode","education","job","familysize","labor","industry","incomesource","house","education2","education3","use7","relation2"]
    data=data[int_col]
    data_group=data.groupby("branchcode")

    edge_index_all=[]
    edges_weight_all=[]
    edges_attr_all=[]
    num_edge=[]
    num_node=[]
    for name,data in data_group:
       
        data=data.drop("branchcode",axis=1)
        #重新设置index
        data.index=range(len(data))
        data=data.to_numpy()
        edge_index,edges_weight,edges_attr=graph_generage(data,node_num=11,threshold=0.95)
        edge_index_all.extend(edge_index)
        edges_weight_all.extend(edges_weight)
        edges_attr_all.extend(edges_attr)
        num_edge.append(len(edge_index))
        num_node.append(len(data))
        print(name,"num_node:",len(data),"num_edge:",len(edge_index))
        #保存边的index的csv文件，特征特征名称为dst和src
    pd.DataFrame(edge_index_all).to_csv("data/wlx/CVPA_preprocess_edge_index.csv",index=False,header=["src","dst"])
    pd.DataFrame(edges_weight_all).to_csv("data/wlx/CVPA_preprocess_edge_weight.csv",index=False,header=["weight"])
    pd.DataFrame(edges_attr_all).to_csv("data/wlx/CVPA_preprocess_edge_attr.csv",index=False)
    pd.DataFrame(num_edge).to_csv("data/wlx/CVPA_preprocess_num_edge.csv",index=False,header=["num_edge"])
    pd.DataFrame(num_node).to_csv("data/wlx/CVPA_preprocess_num_node.csv",index=False,header=["num_node"])
    return len(edge_index_all)

edge_index_num=create_graph_from_csv("data/wlx/CVPA_preprocess.csv")
print(edge_index_num)
