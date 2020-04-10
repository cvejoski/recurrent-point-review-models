
import os
from gentext.graphs.processes import node2vec_walk
from gentext.graphs.models import InfiniteEdgePartition
from gentext.graphs.utils import save_graph, read_graph

from gentext.graphs.utils import community_edge_matrix_stick_breaking, stick_breaking

kwargs = {"number_of_nodes":100,
        "number_of_communities":3,
        "finite_difference_epsilon":0.001,
        "beta_alpha":1.,
        "gamma_alpha":1.,
        "gamma_beta":1.,
        "dirichlet_alpha":[1.,1.],
        "sparse_matrix":True}

training_arguments = {"number_of_epochs":1,
                     "batch_size":10,
                     "finite_difference_epsilon":0.001,
                     "learning_rate":0.001}

data_dir = "../../sample_data/graphs/"
file_name = "small_iep.txt"

if __name__=="__main__":

    #IEP = InfiniteEdgePartition(**kwargs)
    #sparse_adjacency = IEP.sample()
    #save_graph(sparse_adjacency,data_dir,file_name)
    sparse_adjacency  = read_graph(data_dir, file_name)
    walk = node2vec_walk(sparse_adjacency,3,current_node=None,p=1.,q=1.)
