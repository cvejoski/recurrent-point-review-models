import torch
import numpy as np
from torch import optim

from deep_graphs.models import InfiniteEdgePartition
from deep_graphs.trainer import IEP_batch_training_step, IEP_training_step

data_dir = "/home/an/Desktop/Doctorado/Projects/GENTEXT/data/graphs/"
file_name = "infinite_partition_model.txt"

kwargs = {"number_of_nodes":100,
         "number_of_communities":3,
         "finite_difference_epsilon":0.001,
         "beta_alpha":1.,
         "gamma_alpha":1.,
         "gamma_beta":1.,
         "dirichlet_alpha":[1.,1.]}

training_arguments = {"number_of_epochs":1,
                      "batch_size":10,
                      "finite_difference_epsilon":0.001,
                      "learning_rate":0.001}

if __name__=="__main__":
    number_of_nodes = kwargs.get('number_of_nodes', 10)
    number_of_communities = kwargs.get('number_of_communities', 2)

    batch_size = training_arguments.get("batch_size", 10)
    number_of_epochs = training_arguments.get("number_of_epochs", 10)
    finite_difference_epsilon = training_arguments.get("finite_difference_epsilon", 0.001)
    learning_rate = training_arguments.get("learning_rate",0.001)

    graph_data = np.int_(np.loadtxt(data_dir + file_name).T)
    IEP = InfiniteEdgePartition(**kwargs)
    graph_optimizer = optim.Adam(IEP.parameters(),lr=learning_rate)

    for epoch in range(number_of_epochs):
        for edge_bath in data_loader(graph_data, batch_size):
            IEP_batch_training_step(IEP, graph_optimizer, edge_bath)
            break
        IEP_training_step(IEP, graph_optimizer)