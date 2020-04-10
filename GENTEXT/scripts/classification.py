import argparse
from pathlib import Path
from torch.nn import functional as F
from tyche.utils.helper import create_instance, load_params, expand_params, get_device, get_class_nonlinearity


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

#try to make everyhting the same
#optimizer = optim.SGD(model.parameters(), lr=config.lr)
#channels ...
class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, vocab, device):
        super(DPCNN, self).__init__()
        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, 300), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(self.channel_size, 5)
        self.voc_dim, self.emb_dim = vocab.vectors.size()
        self.embedding = nn.Embedding(self.voc_dim, self.emb_dim)
        emb_matrix = vocab.vectors.to(device)
        self.embedding.weight.data.copy_(emb_matrix)


    def forward(self, x):
        x, _ = x
        x = self.embedding(x)  # [B, T, D]
        x = torch.unsqueeze(x, 1)  # [B, 1, T, D]
        batch = x.shape[0]

        # Region embedding
        x = self.conv_region_embedding(x)        # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)                      # pad
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        x = x.view(batch, self.channel_size)
        x = self.linear_out(x)

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels


def train(args):
    params = load_params(args.parameters)
    gs_params = expand_params(params)

    for search in gs_params:
        train_params(search)


def train_params(params):

    print(("Name of the Experiment: " + params['name']))
    device = get_device(params)




    # Data loader
    data_loader = create_instance('data_loader', params, device)

    # Model
    #model = create_instance('network', params, data_loader.vocab, data_loader.fix_len, 0)
    #model.to(device)

    model2 = DPCNN(data_loader.vocab, device)
    model2.to(device)

    critic_optimizer = create_instance('critic_optimizer', params, model2.parameters())


    # Optimizers
    optimizers = dict()
    optimizers['critic_optimizer'] = critic_optimizer
    epochs = 20
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for batch in data_loader.train:
            x = batch.text
            y = batch.label
            optimizers['critic_optimizer'].zero_grad()
            logits = model2.forward(x)
            class_loss = F.cross_entropy(logits, y)
            class_loss.backward()
            optimizers['critic_optimizer'].step()
            epoch_loss += class_loss

            prediction = logits.argmax(dim=1)
            result = [i for i, j in zip(prediction, y) if i == j]
            accuracy = len(result)/x[0].size(0)
            epoch_accuracy += accuracy

        print("Training:")
        print("epoch_loss {}".format(epoch_loss))
        print("epoch_accuracy {}".format(epoch_accuracy/len(data_loader.train)))
        with torch.no_grad():
            epoch_loss = 0
            epoch_accuracy = 0
            for batch in data_loader.validate:
                x = batch.text
                y = batch.label
                logits = model2.forward(x)
                class_loss = F.cross_entropy(logits, y)
                epoch_loss += class_loss

                prediction = logits.argmax(dim=1)
                result = [i for i, j in zip(prediction, y) if i == j]
                accuracy = len(result)/x[0].size(0)
                epoch_accuracy += accuracy

        print("Validation:")
        print("epoch_loss {}".format(epoch_loss))
        print("epoch_accuracy {}".format(epoch_accuracy/len(data_loader.validate)))


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "parameters",
            type=Path,
            help="path to the file containing the parameters \
                                for the experiment")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    train(args)


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])