
import torch
import numpy as np
from dpp.data.loaders import BasicEventDataLoader
from dpp.data.loaders import InteractingPointsDataLoader
from dpp.utils.datahandling import interacting_systems_load_data


import torch.optim as optim
from dpp.blocks import RNNDecoder
from dpp.blocks import MLPEncoder
from dpp.models import InteractingPointProcess
from dpp.trainer import TrainingInteractingPointProcess
from dpp.utils.gumbel import gumbel_softmax, my_softmax
from dpp.utils.datahandling import encode_onehot
from tyche.loss import IPP_ELBO

basic_event_data_loader =  {"data_path": "/home/an/Desktop/Projects/PointProcesses/Results/NonlinearHawkes/InteractingPointsData/",
                            "batch_size": 10,
                             "bptt_size": 20,
                             "shuffle": True,
                             "num_workers": 1}
dims = 4
timesteps = 49
encoder_hidden = 256
edge_types = 2
decoder_hidden = 256
number_of_interacting_atoms = 5

model_args = {
    "module": "dpp_test.models",
    "name": "InteractingPointProcess",
    "args": {
        "number_of_interacting_atoms":number_of_interacting_atoms,
        "timesteps_per_sample":timesteps,
        "prediction_steps":10,
        "input_dimensions":dims,
        "edge_types":edge_types,
        "gumbel_tau":0.01,
        "gumbel_hard":False,
        "encoder": {
            "module": "dpp_test.blocks",
            "name": "MLPEncoder",
            "args": {
                "n_hid": encoder_hidden,
                "do_prob": 0.0,
                "factor": True
            }
        },
        "decoder": {
            "module": "dpp_test.blocks",
            "name": "RNNDecoder",
            "args": {
                "n_hid": decoder_hidden,
                "do_prob": 0.0,
                "skip_first": False,
            }
        }
    }
}

LOSS_ARGS={
    "module": "tyche.loss",
    "name": "IPP_ELBO",
    "args": {
        "reduction": "sum",
        "prior":True,
        "output_variance":5e-5,
        "number_of_interacting_atoms":number_of_interacting_atoms,
        "edge_types":edge_types
    }
}

TRAINER_ARGS={
    "module": "tyche.trainer",
    "name": "TrainingRnnHawkes",
    "args": {
        "bm_metric": "RMSELoss"
    },
    "epochs": 250,
    "save_dir": "/tmp/dpp_test/saved/",
    "logging": {
        "tensorboard_dir": "/tmp/dpp_test/logging/tensorboard/",
        "logging_dir": "/tmp/dpp_test/logging/raw/",
        "formatters": {
            "verbose": "%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s",
            "simple": "%(levelname)s %(asctime)s %(message)s"
        }
    }
}

#"hidden_size": 32,
#"embedding_size": 16,
#"cell_type": "LSTM",
#"n_layers": 1,
#"n_markers": 2,
#"dropout": 0

loss_args = LOSS_ARGS.get("args")
model_args = model_args.get("args")
number_of_interacting_atoms = model_args.get("number_of_interacting_atoms",None)
gumbel_tau = model_args.get("gumbel_tau",0.01)
gumbel_hard = model_args.get("gumbel_hard",0.001)
timesteps = model_args.get("timesteps_per_sample",10)
prediction_steps = model_args.get("prediction_steps",2)

MLPEncoder_ARGS = model_args.get("encoder",None)
MLPEncoder_args = MLPEncoder_ARGS.get("args",None)

RNNDecoder_ARGS =  model_args.get("decoder",None)
RNNDecoder_args = RNNDecoder_ARGS.get("args",None)

Trainer_args = TRAINER_ARGS.get("args",None)

# Generate off-diagonal interaction graph
off_diag = np.ones([number_of_interacting_atoms, number_of_interacting_atoms]) - np.eye(number_of_interacting_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if __name__=="__main__":
    #POINT PROCESSES
    print("Point Processes")
    data_loader = InteractingPointsDataLoader(**basic_event_data_loader)
    for batch_idx, (data,relations) in enumerate(data_loader.train):
        print(data.shape)
        break

    #INTERACTING SYSTEMS
    print(" Interacting Systems")
    data_dir = "/home/an/Desktop/Projects/General/interacting-systems/data/"
    suffix = '_springs5'
    train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min = interacting_systems_load_data(data_dir,10,suffix)
    for batch_idx, (data, relations) in enumerate(train_data_loader):
        print("data")
        print(data.shape)
        print("relations")
        print(relations.shape)
        break

    # ENCODERS AND DECODERS
    mlp_encoder = MLPEncoder(timesteps*dims,edge_types, **MLPEncoder_args)
    rnn_decoder = RNNDecoder(dims,edge_types,**RNNDecoder_args)
    #MODEL
    IPP = InteractingPointProcess(**model_args)
    IPP.train()
    #LOSS
    elbo = IPP_ELBO(**loss_args)
    #OPTIMIZER
    optimizer = optim.Adam(list(mlp_encoder.parameters()) + list(rnn_decoder.parameters()),lr=0.001)
    #TRAINER
    for batch_idx, (data, relations) in enumerate(train_data_loader):
        target = data[:, :, 1:, :]
        logits = mlp_encoder(data, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=gumbel_tau, hard=gumbel_hard)
        prob = my_softmax(logits, -1)
        output = rnn_decoder(data,
                             edges,
                             rel_rec,
                             rel_send,
                             100,
                             burn_in=True,
                             burn_in_steps=timesteps - prediction_steps)

        logits, edges, prob, output = IPP(data)
        elbo(output, target,prob, 0)
        optimizer.zero_grad()

        break