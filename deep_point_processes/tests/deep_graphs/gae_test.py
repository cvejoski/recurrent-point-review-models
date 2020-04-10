
import torch
from tyche.utils.helper import create_instance
from tyche.utils.helper import load_params
from deep_graphs.models import GCNModelVAE
from deep_graphs.data.loaders import GAEDataLoader

data_dir = "/home/an/Desktop/Projects/DeepPointProcesses/data/deep_graphs/cora/"

epochs = 1
dropout = 0.2

data_dir = "/home/an/Desktop/Projects/DeepPointProcesses/deep_point_processes/experiments/graphs/gae.yaml"
full_params = load_params(data_dir)
model_params = full_params.get('model')
encoder_params = model_params.get('args').get('encoder')
decoder_params = model_params.get('args').get('decoder')
data_loader_params = full_params.get('data_loader')
device = torch.device("cpu")

if __name__=="__main__":
    data_loader = GAEDataLoader(device,**data_loader_params.get("args"))
    model = GCNModelVAE(data_loader,**model_params.get("args"))

    optimizers = dict()
    optimizer = create_instance('optimizer', full_params, model.parameters())
    optimizers['loss_optimizer'] = optimizer

    model.train_step(data_loader, optimizers, 0., None)