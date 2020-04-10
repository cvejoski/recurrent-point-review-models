from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import torch

from deep_graphs.utils.random_walks_metrics import score_matrix_from_random_walks, graph_from_scores
from deep_graphs.data.datasets import read_all_graphs
from deep_graphs.processes import node2vec_walk

number_of_nodes = 20

if __name__=="__main__":
    #BARABASI GRAPH
    barabasi_graph = nx.barabasi_albert_graph(number_of_nodes,3)
    sparse_adjacency = nx.adjacency_matrix(barabasi_graph).tocoo()

    #GENERATE GRAPHS
    node2vec_walk(sparse_adjacency, 5)
    walks  = []

    #READ FROM MODELS
    for i in range(100):
        walks.append(node2vec_walk(sparse_adjacency,5))

    walks = torch.Tensor(walks)
    score_matrix = score_matrix_from_random_walks(walks,number_of_nodes,symmetric=True)
    graph_adjacency_from_scores = graph_from_scores(score_matrix , barabasi_graph.number_of_edges())

    graph_from_scores = nx.from_numpy_matrix(graph_adjacency_from_scores)
    nx.draw(graph_from_scores)
    plt.show()

