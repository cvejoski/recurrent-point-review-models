import numpy as np
import torch as torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from tyche.utils.helper import get_class_nonlinearity


def get_cnn_params(out_channels, kernel, stride, padding, dilation, n_blocks, n_layers_per_block, T, cnn_block):
    """
    Returns list of kernel, padding, dilation and stride sizes,
    as well as input dimensions and channel numbers,
    for CNN blocks constructed using EncoderCNN or GaussianEncoderCNN, and corresponding decoders.
    To be used to compute the receptive field of the blocks.
    """
    n_levels = len(kernel)
    dim = [T]
    K = []
    P = []
    D = []
    S = []
    DIM = []
    N_CH = []
    for i in range(n_levels):
        # Dimension along time direction
        t = (dim[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1
        dim.append(t)
        # Number of input channels
        in_channels = 1 if i == 0 else out_channels[i - 1]
        if cnn_block == "residual_net":
            # Padding to keep time dimension constant
            p_ = int(dilation[i] * (kernel[i] - 1) / 2)
            for j in range(n_blocks):
                dim_in, dim_out = dim[i], dim[i + 1]
                d = [dilation[i]] * n_layers_per_block
                k = [kernel[i]] * n_layers_per_block
                if j == 0:
                    # Only the first block modifies the n_channels and T of the input tensor
                    ch_in = in_channels
                    p = [padding[i]] + [p_] * (n_layers_per_block - 1)
                    s = [stride[i]] + [1] * (n_layers_per_block - 1)
                    dim_ = [dim_in] + [dim_out] * (n_layers_per_block - 1)
                else:
                    ch_in = out_channels[i]
                    p = [p_] * n_layers_per_block
                    s = [1] * n_layers_per_block
                    dim_ = [dim_out] * n_layers_per_block
                for l in range(n_layers_per_block):
                    K.append(k[l])
                    P.append(p[l])
                    S.append(s[l])
                    D.append(d[l])
                    DIM.append(dim_[l])
                    N_CH.append((ch_in, out_channels[i]))
        else:
            in_channels = 1 if i == 0 else out_channels[i - 1]
            K.append(kernel[i])
            P.append(padding[i])
            S.append(stride[i])
            D.append(dilation[i])
            DIM.append(dim[i])
            N_CH.append((in_channels, out_channels[i]))

    return K, P, S, D, DIM, N_CH


def print_receptive_field(out_channels, kernel, stride, padding,
                          dilation, n_blocks, n_layers_per_block, T, cnn_block):
    """
    Computes and prints the receptive field of Encoder/Decoder CNN
    """

    K, P, S, D, DIM, N_CH = get_cnn_params(out_channels, kernel, stride, padding,
                                           dilation, n_blocks, n_layers_per_block, T, cnn_block)
    N = len(K)
    r = 1
    S_ = 1
    for i in range(N):
        S_ *= S[i]
        r += (K[i] - 1) * D[i] * S_
        print("k =", K[i], "p =", P[i], "s =", S[i], "d =", D[i], "in_ch =", N_CH[i][0], "out_ch =", N_CH[i][1],
              "t(input) =", DIM[i], "Receptive field =", r)


class Block(nn.Module):
    def __init__(self, vocab, fix_len, latent_dim, recurrent=False, **kwargs):
        super(Block, self).__init__()
        self.voc_dim, self.emb_dim = vocab.vectors.size()
        self.fix_len = fix_len
        self.latent_dim = latent_dim
        self.SOS = vocab.stoi['<sos>']
        self.EOS = vocab.stoi['<eos>']
        self.PAD = vocab.stoi['<pad>']
        self.UNK = vocab.unk_index
        self.embedding = nn.Embedding(self.voc_dim, self.emb_dim)
        emb_matrix = vocab.vectors.to(self.device)
        self.embedding.weight.data.copy_(emb_matrix)
        self.embedding.weight.requires_grad = kwargs.get('train_word_embeddings', True)
        self.recurrent = recurrent

    @property
    def is_recurrent(self):
        return self.recurrent

    @property
    def device(self):
        return next(self.parameters()).device


class AttentionLayer(nn.Module):
    """
    Attention mechanism from
    "Hierarchical Attention Networks for Document Classification" (Z Yang et al)
    """

    def __init__(self, in_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.m_context = nn.Parameter(torch.Tensor(attention_dim, in_dim))
        self.b_context = nn.Parameter(torch.Tensor(attention_dim))
        self.v_context = nn.Parameter(torch.Tensor(attention_dim))
        self.param_init()

    def forward(self, x):
        """
        input (x) shape: [B, T, D]
        """
        u = torch.tensordot(x, self.m_context, dims=[[2], [1]]) + self.b_context
        u = torch.tanh(u)  # [B, T, Z]
        u = torch.tensordot(u, self.v_context, dims=[[2], [0]])  # [B, T]
        alpha = F.softmax(u, dim=1)
        alpha = torch.unsqueeze(alpha, 2)  # [B, T, 1]
        out = torch.sum(alpha * x, dim=1)  # [B, D]
        return out

    def param_init(self):
        """
        Parameters initialization.
        """
        torch.nn.init.normal_(self.m_context)
        torch.nn.init.normal_(self.v_context)
        torch.nn.init.zeros_(self.b_context)


class LMDecoderRNN(Block):
    """
    Language model for text generation
    parametrized by Recurrent Neural Networks
    """

    def __init__(self, vocab, fix_len, latent_dim, **kwargs):
        super(LMDecoderRNN, self).__init__(vocab, fix_len, latent_dim, True, **kwargs)

        self.hidden_dim = kwargs.get('hidden_dim')
        self.dropout = kwargs.get('dropout', 0)
        self.hidden_state = None
        self.rnn_cell = nn.LSTM(input_size=self.emb_dim + self.latent_dim,
                                hidden_size=self.hidden_dim,
                                batch_first=True)

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.get_logits = nn.Linear(self.hidden_dim, self.voc_dim)

    def forward(self, input, z=None):
        """
        input: input data (x): [B, T], latent code (z): [B, Z]
        returns: logits over Voc. Shape: [B, T, V]
        """
        x, l = input
        output = torch.zeros(x.size(0), x.size(1), self.hidden_dim, device=self.device)
        x = self.embedding(x)  # [B, T, D]
        _ix = l.nonzero().view(-1)
        t_len = x.size(1)

        if z is not None:
            z = z.unsqueeze(1).expand(-1, self.fix_len, -1)
            x = torch.cat((x, z), dim=-1)  # [B, T, D + D']

        x = torch.nn.utils.rnn.pack_padded_sequence(x[_ix], l[_ix], True, False)

        hidden_state = self._get_hidden_states(_ix)
        x, hidden_state = self.rnn_cell(x, hidden_state)  # [B, T, D]
        self._update_hidden_state(hidden_state, _ix)

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, True, total_length=t_len)

        # apply dropout
        x = self.dropout_layer(x)

        output[_ix] = output[_ix] + x
        logits = self.get_logits(output)  # [B, T, V]
        return logits, output

    def _get_hidden_states(self, _ix):
        return tuple(x[:, _ix] for x in self.hidden_state)

    def _update_hidden_state(self, hidden_state, _ix):
        self.hidden_state[0][:, _ix] = hidden_state[0]
        self.hidden_state[1][:, _ix] = hidden_state[1]

    def initialize_hidden_state(self, batch_size: int, device: any):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)
        self.hidden_state = (h.to(device), c.to(device))

    def reset_history(self):
        self.hidden_state = tuple(x.detach() for x in self.hidden_state)


class AuxiliaryNoiseCNN(nn.Module):
    """
    Adds auxiliary noise layer to input of CNNs
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation, in_dim):
        super(AuxiliaryNoiseCNN, self).__init__()

        self.h, self.w = in_dim
        self.in_channels = in_channels
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding, stride=stride, dilation=dilation)

    def forward(self, x):
        """
        input: input data (x) [B, channels, T, D]
        """
        b, _, _, _ = x.shape

        device = next(self.parameters()).device
        z = torch.randn(b, self.in_channels, self.h, self.w, device=device)
        z = self.layer(z)

        return z + x


class Mask1D(nn.Module):
    """
    Given a padded sequence [pad, ..., pad, X, pad, ..., pad]
    (in the time dimension), cuts short the right padding for
    causal convolution, i.e. output: [pad, ..., pad, X]

    """

    def __init__(self, padding):
        super(Mask1D, self).__init__()
        self.padding = padding

    def forward(self, x):
        """
        input (x) shape [B, channels, T, D]
        output shape [B, channels, T-p, D]
        """
        return x[:, :, :-self.padding[0]].contiguous()


class ResidualBlockBottleneck(nn.Module):
    """
    Residual Block Bottleneck (from Bytenet)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple, padding: int, stride: int, dilation: int,
                 nonlinearity, dim: tuple, normalization: str,
                 causal: bool = False, auxiliary_noise: bool = False, dropout: int = 0):

        super(ResidualBlockBottleneck, self).__init__()

        # Input and output dimensions (H, W):
        dim_in, dim_out = dim

        layers = nn.ModuleList([])

        if causal:
            padding = 2 * padding

        if in_channels == 1:
            inner_number_channels = int(out_channels / 2)
        else:
            inner_number_channels = int(in_channels / 2)

        if normalization == "weight_norm":
            # 1 x 1
            layers.append(weight_norm(nn.Conv2d(in_channels, inner_number_channels, (1, kernel_size[1]))))

            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

            # noise
            if auxiliary_noise:
                layers.append(AuxiliaryNoiseCNN(1, inner_number_channels,
                                                (1, 1), (0, 0), (1, 1), (1, 1), (dim_in[0], 1)))
            # non-linearity
            layers.append(nonlinearity())

            # k x 1
            layers.append(weight_norm(nn.Conv2d(inner_number_channels, inner_number_channels, (kernel_size[0], 1),
                                                padding=(padding, 0), stride=(stride, 1), dilation=(dilation, 1))))
            # noise
            if auxiliary_noise:
                layers.append(AuxiliaryNoiseCNN(1, inner_number_channels,
                                                (1, 1), (0, 0), (1, 1), (1, 1), (dim_out[0], 1)))
            # causality
            if causal:
                layers.append(Mask1D((padding, 0)))
            # non-linearity
            layers.append(nonlinearity())

            # 1 x 1
            layers.append(weight_norm(nn.Conv2d(inner_number_channels, out_channels, 1)))
            # noise
            if auxiliary_noise:
                layers.append(AuxiliaryNoiseCNN(1, out_channels, (1, 1), (0, 0), (1, 1), (1, 1), (dim_out[0], 1)))
            # non-linearity
            layers.append(nonlinearity())

        elif normalization == "instance_norm":
            # normalization
            layers.append(nn.InstanceNorm2d(in_channels, affine=True))
            # non-linearity
            layers.append(nonlinearity())
            # 1 x 1
            layers.append(nn.Conv2d(in_channels, inner_number_channels, (1, kernel_size[1])))

            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

            # noise
            if auxiliary_noise:
                layers.append(AuxiliaryNoiseCNN(1, inner_number_channels,
                                                (1, 1), (0, 0), (1, 1), (1, 1), (dim_in[0], 1)))

            # normalization
            layers.append(nn.InstanceNorm2d(inner_number_channels, affine=True))
            # non-linearity
            layers.append(nonlinearity())
            # k x 1
            layers.append(nn.Conv2d(inner_number_channels, inner_number_channels, (kernel_size[0], 1),
                                    padding=(padding, 0), stride=(stride, 1), dilation=(dilation, 1)))
            # noise
            if auxiliary_noise:
                layers.append(AuxiliaryNoiseCNN(1, inner_number_channels,
                                                (1, 1), (0, 0), (1, 1), (1, 1), (dim_out[0], 1)))
            # causality
            if causal:
                layers.append(Mask1D((padding, 0)))

            # normalization
            layers.append(nn.InstanceNorm2d(inner_number_channels, affine=True))
            # non-linearity
            layers.append(nonlinearity())
            # 1 x 1
            layers.append(nn.Conv2d(inner_number_channels, out_channels, 1))
            # noise
            if auxiliary_noise:
                layers.append(AuxiliaryNoiseCNN(1, out_channels, (1, 1), (0, 0), (1, 1), (1, 1), (dim_out[0], 1)))
        else:
            raise Exception("Undefined Normalization! It should be set to either weight_norm or instance_norm.")

        self.layers = nn.Sequential(*layers)

        # Residual layer conditions on time dimension:
        if not causal:
            # (i) halves input height dim:
            cond_1 = stride != 1 and \
                     padding == np.ceil((dilation * (kernel_size[0] - 1) + 1) / stride) - 1

            # (ii) reduces input height dim by x with k = (x/d)+1:
            cond_2 = stride == 1 and padding == 0

            # (iii) keeps input height dim fixed, with k odd or d even
            cond_3 = stride == 1 and padding == int(dilation * (kernel_size[0] - 1) / 2)
        else:
            cond_1 = stride != 1 and \
                     padding == 2 * np.ceil((dilation * (kernel_size[0] - 1) + 1) / stride) - 2
            cond_2 = stride == 1 and padding == 0
            cond_3 = stride == 1 and padding == dilation * (kernel_size[0] - 1)

        # if both n_channels and H/W changed, or if only H/W changed
        if any([cond_1, cond_2]):
            res_layer = nn.ModuleList([])
            res_layer.append(nn.Conv2d(in_channels, out_channels, kernel_size[0],
                                       padding=padding, stride=stride, dilation=dilation))
            if causal:
                res_layer.append(Mask1D((padding, 0)))

            self.res_layer = nn.Sequential(*res_layer)

        # if only n_channels changed (or only W changed)
        elif in_channels != out_channels and cond_3 or kernel_size[1] != 1:
            self.res_layer = nn.Conv2d(in_channels, out_channels, (1, kernel_size[1]))

        else:
            self.res_layer = None

        self.param_init()

    def param_init(self):
        """
        Parameters initialization.
        """
        for layer in self.modules():
            if hasattr(layer, 'weight'):
                if isinstance(layer, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)

    def forward(self, x):
        out = self.layers(x)
        res = x if self.res_layer is None else self.res_layer(x)
        return out + res


class LMDecoderCNN(Block):
    """
    Decoder for text generation
    parametrized by (causal) Convolutional Neural Networks
    """

    def __init__(self, vocab, fix_len, latent_dim, **kwargs):
        super(LMDecoderCNN, self).__init__(vocab, fix_len, latent_dim, **kwargs)

        # Parameters:
        kernel = kwargs.get('kernel_sizes')
        stride = kwargs.get('stride')
        padding = kwargs.get('padding')
        dilation = kwargs.get('dilation')
        out_channels = kwargs.get('output_channels')
        aux_noise_1 = False

        cnn_block = kwargs.get('cnn_block')

        # Non-linearity:
        nonlinearity_ = kwargs.get('nonlinearity')
        nonlinearity = get_class_nonlinearity(nonlinearity_)

        # Network size:
        n_layers_per_block = kwargs.get('n_layers_per_residual_block')
        n_blocks = kwargs.get('n_residual_blocks')

        normalization = kwargs.get('normalization')

        # tokens for generation (out-of-sampling)
        self.max_len = kwargs.get('max_len', 32)

        # residual blocks
        n_levels = len(out_channels)
        dim = [(self.fix_len, self.emb_dim + self.latent_dim)]
        for i in range(n_levels):
            t = (dim[i][0] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1
            dim.append((t, 1))

        layers = nn.ModuleList([])
        print("----- Decoder (resnet) -----")

        print_receptive_field(out_channels, kernel, stride, padding,
                              dilation, n_blocks, n_layers_per_block, self.fix_len, cnn_block)
        for i in range(n_levels):
            in_channels = 1 if i == 0 else out_channels[i - 1]
            kernel_size = (kernel[i], self.emb_dim + self.latent_dim) if i == 0 else (kernel[i], 1)

            if cnn_block == "residual_bottleneck":
                layers.append(ResidualBlockBottleneck(in_channels,
                                                      out_channels[i],
                                                      kernel_size,
                                                      padding[i],
                                                      stride[i],
                                                      dilation[i],
                                                      nonlinearity,
                                                      (dim[i], dim[i + 1]),
                                                      normalization=normalization,
                                                      causal=True,
                                                      auxiliary_noise=aux_noise_1))
            else:
                print("cnn-block not specified")
                raise Exception

        self.layers = nn.Sequential(*layers)
        self.get_logits = nn.Linear(out_channels[-1], self.voc_dim)

    def forward(self, input, z=None):
        """
        Perform a forward step of the model.

        Parameters
        ----------
        input (Tensor) of shape [B, T, D]
        g (Tensor) of shape [B, T, D']

        Returns
        -------

        """
        x, _ = input
        x = self.embedding(x)  # [B, T, D]
        if z is not None:
            z = z.unsqueeze(1).expand(-1, self.fix_len, -1)
            x = torch.cat((x, z), dim=-1)  # [B, T, D + D']
        x = torch.unsqueeze(x, 1)  # [B, 1, T, D]

        # Convolution
        h = self.layers(x)  # [B, D, T, 1]
        h = torch.squeeze(h, 3).permute(0, 2, 1)  # [B, T, D]
        logits = self.get_logits(h)  # [B, T, V]

        return logits, h
