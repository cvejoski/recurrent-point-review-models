from deep_graphs.utils.functions import positive_parameter, positive_parameter_inverse, construct_inverse_gamma

from deep_graphs.data.loaders import EdgesDataLoader, old_data_loader
from deep_graphs.utils.graphs_plots import from_adjacency_numpy_plot
from deep_graphs.data.datasets import save_graph
from deep_graphs.models import IEPM
from scipy.stats import gamma as scipy_gamma

from torch import optim
import torch
from matplotlib import pyplot as plt
import networkx as nx

data_dir = "/home/an/Desktop/Projects/PointProcesses/Data/deep_graphs/iepm/"
batch_size = 32
device = torch.device("cpu")

iepm_args = {
    "number_of_nodes":2000,
    "number_of_communities":3,
    "finite_difference_epsilon":1e-3,
    "gamma_alpha":0.2,
    "gamma_beta":0.2,
    "sparse_matrix":True}

basic_event_data_loader =  {"data_path": data_dir,
                            "data_name":"iepm",
                            "number_of_nodes": 2000,
                            "batch_size": batch_size,
                            "shuffle": True,
                            "num_workers": 1}

trainer_args = {
    "module": "genimage.trainer",
    "name": "TrainingVAE",
    "args":None,
    "epochs": 100,
    "save_dir": "./results/saved/",
    "logging": {
        "tensorboard_dir": "./results/logging/tensorboard/",
        "logging_dir": "./results/logging/raw/",
        "formatters": {
            "verbose": "%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s",
            "simple": "%(levelname)s %(asctime)s %(message)s"
        }
    }
}

if __name__=="__main__":
    #====================================================
    # SAMPLE
    #====================================================
    test_a = 250.
    test_b = 40.

    iepm = IEPM(None,False,**iepm_args)
    optimizer = optim.Adam(iepm.parameters(),lr=0.01)

    alpha = positive_parameter_inverse(torch.full(iepm.alpha.size(), test_a).double())
    beta = positive_parameter_inverse(torch.full(iepm.beta.size(), test_b).double())

    print(alpha)
    print(beta)

    test_a_back = positive_parameter(alpha)
    test_b_back = positive_parameter(beta)
    print(test_a_back[0,0])
    print(test_b_back[0,0])

    iepm.alpha = torch.nn.Parameter(alpha)
    iepm.beta = torch.nn.Parameter(beta)

    mygamma = torch.distributions.Gamma(torch.full(iepm.alpha.size(), test_a).double(),
                                        torch.full(iepm.beta.size(), test_b).double())
    mygamma_sample = mygamma.sample().flatten().numpy()
    print("from scipy")
    print((scipy_gamma.fit(mygamma_sample, loc=0)[0], 1. / scipy_gamma.fit(mygamma_sample, loc=0)[2]))

    reparametrization_sample = iepm((test_a,test_b))[1]
    reparametrization_sample = reparametrization_sample.flatten().detach().numpy()

    print("from reparametrization")
    print((scipy_gamma.fit(reparametrization_sample, loc=0)[0], 1. / scipy_gamma.fit(reparametrization_sample, loc=0)[2]))

    #edges = iepm.sample() # matrix is symmetric and probably unconnected
    #from_adjacency_numpy_plot(edges)
    #save_graph(edges,data_dir,"iepm",iepm_args=iepm_args)
    #====================================================
    # LOAD
    #====================================================
    EDL = EdgesDataLoader(device,**basic_event_data_loader)
    print(len(EDL.train))
    #====================================================
    # TRAINING
    #====================================================
    for minibatch in EDL.train:
        iepm.train_step(minibatch,{"loss_optimizer":optimizer},None)
        break