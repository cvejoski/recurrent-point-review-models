

from dpp.data.loaders import ConditionalWasssersteingPointDataLoader
from dpp.blocks import Encoder_PPW, Decoder_PPW
import torch

conditional_wasserstein = {"data_path": "/home/an/Desktop/Projects/PointProcesses/Results/NonlinearHawkes/InteractingPointsData/",
                          "batch_size": 32,
                          "past_of_sequence": .7,
                          "shuffle": True,
                          "num_workers": 1}

number_of_interacting_atoms = 5
timesteps = 10
dims = 2
edge_types = 2

model_args = {
    "module": "dpp_test.models",
    "name": "ConditionalWasserstein",
    "args": {
        "number_of_interacting_atoms":number_of_interacting_atoms,
        "timesteps_per_sample":timesteps,
        "prediction_steps":10,
        "input_dimensions":dims,
        "edge_types":edge_types,
        "gumbel_tau":0.01,
        "gumbel_hard":False,
        "encoder": {
            "module": "dpp.blocks",
            "name": "Encoder_PPW",
            "args": {
                "type_of_rnn":"gru",
                "number_of_layers":1,
                "past_time_steps":37,
                "hidden_size":256,
                "input_dimension":2
            }
        },
        "decoder": {
            "module": "dpp.blocks",
            "name": "Decoder_PPW",
            "args": {
                "type_of_rnn":"gru",
                "number_of_layers":1,
                "future_time_steps":17,
                "hidden_size":256,
                "input_dimension":2
            }
        }
    }
}


#=======================================================
# DEFINING THE MODELS
#=======================================================

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

encoder_args = model_args.get("args").get("encoder").get("args")
decoder_args = model_args.get("args").get("decoder").get("args")

encoder_ppw = Encoder_PPW(**encoder_args)
decoder_ppw = Decoder_PPW(**decoder_args)

batch_size = conditional_wasserstein.get("batch_size")
hidden_state = encoder_ppw.initialize_hidden_state(batch_size,device)

if __name__=="__main__":
    #CONDITIONAL WASSERSTEIN
    print("Conditional Wasserstein Point Processes")
    data_loader = ConditionalWasssersteingPointDataLoader(**conditional_wasserstein)

    for batch_idx, (past,future) in enumerate(data_loader.train):
        print(past.shape)
        print(future.shape)

        output, hidden_state = encoder_ppw(past, hidden_state)
        output, hidden_state = decoder_ppw(future,hidden_state)


        break