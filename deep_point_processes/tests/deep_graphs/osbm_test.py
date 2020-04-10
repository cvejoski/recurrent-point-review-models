from deep_graphs.models import OverlappingStochasticBlockModels
from deep_graphs.data.loaders import EdgesDataLoader
from deep_graphs.processes import node2vec_walk

import torch
import numpy as np
from tyche.utils.helper import load_params

from scipy.sparse import coo_matrix
from deep_graphs.models import GCNModelVAE
from tyche.utils.helper import create_instance
from deep_graphs.data.loaders import GAEDataLoader

data_dir = "/home/an/Desktop/Projects/DeepPointProcesses/data/deep_graphs/cora/"

epochs = 1
dropout = 0.2

data_dir = "/home/an/Desktop/Projects/DeepPointProcesses/deep_point_processes/experiments/graphs/osbm.yaml"
full_params = load_params(data_dir)
model_params = full_params.get('model')
encoder_params = model_params.get('args').get('encoder')
decoder_params = model_params.get('args').get('decoder')
data_loader_params = full_params.get('data_loader')
device = torch.device("cpu")

if __name__=="__main__":
    #==================================
    # SAMPLE
    #==================================
    OSBM = OverlappingStochasticBlockModels(device,**model_params.get("args"))
    graph_adjacency = OSBM.generator().numpy()
    np.fill_diagonal(graph_adjacency, 0)
    sparse_adjacency = coo_matrix(graph_adjacency)

#    save_graph(graph_adjacency,data_dir,"osbm_edges",sparse=False)

