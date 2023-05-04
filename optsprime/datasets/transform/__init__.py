from .init_graph import initialize_edge_weight,initialize_node_features
from torch_geometric.transforms import Compose

tf_config=dict(initialize_edge_weight=initialize_edge_weight,
            initialize_node_features=initialize_node_features)
pretf_config=dict()

def build_transforms(cfg):
    tf_compose=[]
    for key in cfg:
        tf_compose.append(tf_config[key])
    transforms=Compose([tf_compose])
    return transforms
    
def build_pretransforms(cfg):
    pretf_compose=[]
    for key in cfg:
        pretf_compose.append(pretf_config[key])
    transforms=Compose([pretf_compose])
    return transforms