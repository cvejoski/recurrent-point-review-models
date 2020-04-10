from itertools import product
from typing import Any, Dict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from tyche.loss import kullback_leibler
from tyche.models import AModel
from tyche.utils import param_scheduler as p_scheduler
from tyche.utils.helper import create_instance, gumbel_softmax, \
    free_params, frozen_params


class BasicLM(AModel):
    """
    Basic Language model (no latent dimension)
    """

    def __init__(self, vocab, fix_len, latent_dim=0, **kwargs):
        super(BasicLM, self).__init__()
        self.voc_dim = vocab.vectors.size(0)
        self.fix_len = fix_len
        self.ignore_index = vocab.stoi['<pad>']
        if self.metrics is not None:
            for m in self.metrics:
                m.ignore_index = self.ignore_index
                m.reduce = self.reduce

        self.decoder = create_instance('decoder', kwargs, *(vocab, fix_len, latent_dim))

    def forward(self, input, z=None):
        """
        input  target shape: [B, T]
        d_embedding (Tensor) of shape [B, D'] representing global dynamic state
        Notation. B: batch size; T: seq len (== fix_len)
        returns: loss and KL divergences of VAE
        """

        logits, h = self.decoder(input, z)  # [B, T, V]
        logits = logits[:, :-1].contiguous().view(-1, self.voc_dim)  # [T * B, V]

        return logits, h

    def loss(self, y, y_target, seq_len=None):
        """
        returns the cross_entropy for the language model
        Notation. B: batch size; T: seq len (== fix_len)
        """
        if seq_len is not None:
            # Mask loss for batch-dependent seq_len normalization
            batch_size = seq_len.size(0)
            loss = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, reduction='none')
            mask = (y_target.view(-1, self.fix_len - 1) != self.ignore_index).float()
            loss = loss.view(-1, self.fix_len - 1) * (mask.float() / (seq_len.view(-1, 1).float() - 1.0))
            loss = loss.sum() / batch_size
        else:
            loss = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index)

        stats = self.new_stats()
        stats['loss'] = loss
        stats['cross_entropy'] = loss
        return stats

    def metric(self, y: Any, y_target: Any, seq_len=None):
        """
        returns a dictionary with metrics
        """
        with torch.no_grad():
            batch_size = y_target.size(0) / (self.fix_len - 1)

            stats = self.new_metric_stats()
            # Cross entropy
            if seq_len is not None:
                # Mask loss for batch-dependent seq_len normalization
                cost = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, reduction='none')
                mask = (y_target.view(-1, self.fix_len - 1) != self.ignore_index).float()
                cost = cost.view(-1, self.fix_len - 1) * (mask.float() / (seq_len.view(-1, 1).float() - 1.0))
                cost = cost.sum() / float(batch_size)
            else:
                cost = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index)

            stats['cross_entropy'] = cost
            stats['perplexity'] = torch.exp(cost)

            if self.metrics is not None:
                for m in self.metrics:
                    stats[type(m).__name__] = m(y, y_target)

            return stats

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        x = minibatch.text
        seq_len = x[1]
        target = x[0][:, 1:].contiguous().view(-1)

        # Optimizers initialization:
        optimizer['loss_optimizer'].zero_grad()

        # Initialize hidden state for rnn models
        B = x[0].size(0)
        self.initialize_hidden_state(B, self.device)

        # Train loss
        logits, h = self.forward(x)
        loss_stats = self.loss(logits, target, seq_len)
        metric_stats = self.metric(logits, target, seq_len)
        loss_stats['loss'].backward()
        optimizer['loss_optimizer'].step()

        # Detach history from rnn models
        self.detach_history()

        prediction = logits.argmax(dim=1).view(B, -1)
        target = target.view(B, -1)
        return {**loss_stats, **metric_stats, **{'reconstruction': (prediction, target)}}

    def validate_step(self, minibatch: Any):
        x = minibatch.text

        target = x[0][:, 1:].contiguous().view(-1)
        seq_len = x[1]
        # Initialize hidden state for rnn models
        B = x[0].size(0)
        self.initialize_hidden_state(B, self.device)

        # Evaluate model
        logits, h = self.forward(x)
        loss_stats = self.loss(logits, target, seq_len)
        metric_stats = self.metric(logits, target, seq_len)

        prediction = logits.argmax(dim=1).view(B, -1)
        target = target.view(B, -1)
        return {**loss_stats, **metric_stats, **{'reconstruction': (prediction, target)}}

    def new_metric_stats(self) -> Dict:
        stats = dict()
        stats['perplexity'] = 0
        stats['cross_entropy'] = 0
        return stats

    def new_stats(self) -> Dict:
        stats = dict()
        stats['loss'] = 0
        stats['cross_entropy'] = 0
        return stats

    def initialize_hidden_state(self, batch_size, device, enc=True, dec=True):
        if dec and self.decoder.is_recurrent:
            self.decoder.initialize_hidden_state(batch_size, device)

    def detach_history(self, enc=True, dec=True):
        if self.decoder.is_recurrent and dec:
            self.decoder.reset_history()


class DynamicLM(BasicLM):
    def __init__(self, vocab, fix_len, latent_dim=0, **kwargs):
        super(DynamicLM, self).__init__(vocab, fix_len, latent_dim, **kwargs)

    def forward(self, input, d_embedding):
        """
        Forward step of the dynamic language model.

        Parameters
        ----------
        input (Tensor) of shape [B, T, D]
        d_embedding (Tensor) of shape [B, D'] representing global dynamic state

        Returns
        -------
        (logits, hidden_state)
        Notation. B: batch size; T: seq len (== fix_len); D: hidden dimension
        """

        logits, h = self.decoder(input, d_embedding)  # [B, T, V]
        logits = logits[:, :-1].contiguous().view(-1, self.voc_dim)  # [T * B, V]

        return logits, h

    def initialize_hidden_state(self, batch_size, dec=True):
        if dec and self.decoder.is_recurrent:
            self.decoder.initialize_hidden_state(batch_size, self.device)


class AE(AModel):
    """
    Standard Deterministic Autoencoder
    """

    def __init__(self, vocab, fix_len, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.voc_dim = vocab.vectors.size(0)
        self.fix_len = fix_len
        self.ignore_index = vocab.stoi['<pad>']
        if self.metrics is not None:
            for m in self.metrics:
                m.ignore_index = self.ignore_index
                m.reduce = self.reduce
        self.latent_dim = kwargs.get('latent_dim')
        self.encoder = create_instance('encoder', kwargs, vocab, fix_len, self.latent_dim)
        self.decoder = create_instance('decoder', kwargs, vocab, fix_len, self.latent_dim)

    def forward(self, input):
        """
        returns: loss and KL divergences of VAE
        input & target shape: [B, T]
        Notation. B: batch size; T: seq len (== fix_len); V: voc size
        """

        h, _, _ = self.encoder(input)

        logits, _ = self.decoder(input, h)  # [B, T, V]
        logits = logits[:, :-1].contiguous().view(-1, self.voc_dim)  # [T * B, V]

        return logits, h

    def loss(self, y, y_target, seq_len=None):
        """
        returns the NORMALIZED [1/(B*T)] loss function of the variational autoencoder
        """

        if seq_len is not None:
            # Mask loss for batch-dependent seq_len normalization
            batch_size = seq_len.size(0)
            loss = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, reduction='none')
            mask = (y_target.view(-1, self.fix_len - 1) != self.ignore_index).float()
            loss = loss.view(-1, self.fix_len - 1) * (mask.float() / (seq_len.view(-1, 1).float() - 1.0))
            loss = loss.sum() / batch_size
        else:
            loss = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index)

        stats = self.new_stats()
        stats['loss'] = loss
        stats['cross_entropy'] = loss
        stats['perplexity'] = torch.exp(loss)
        return stats

    def metric(self, y: Any, y_target: Any, seq_len=None):
        with torch.no_grad():
            stats = dict()

            if seq_len is not None:
                # Mask loss for batch-dependent seq_len normalization
                batch_size = seq_len.size(0)
                loss = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, reduction='none')
                mask = (y_target.view(-1, self.fix_len - 1) != self.ignore_index).float()
                loss = loss.view(-1, self.fix_len - 1) * (mask.float() / (seq_len.view(-1, 1).float() - 1.0))
                loss = loss.sum() / batch_size
            else:
                loss = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index)

            stats['perplexity'] = torch.exp(loss)

            if self.metrics is not None:
                for m in self.metrics:
                    stats[type(m).__name__] = m(y, y_target)
            return stats

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        x = minibatch.text
        seq_len = x[1]
        target = x[0][:, 1:].contiguous().view(-1)

        # Optimizers initialization:
        optimizer['loss_optimizer'].zero_grad()

        # Initialize hidden state for rnn models
        B = x[0].size(0)
        self.initialize_hidden_state(B, self.device)

        # Train loss
        logits, _ = self.forward(x)
        loss_stats = self.loss(logits, target, seq_len)
        metric_stats = self.metric(logits, target, seq_len)
        loss_stats['loss'].backward()
        optimizer['loss_optimizer'].step()

        # Detach history from rnn models
        self.detach_history()

        prediction = logits.argmax(dim=1).view(B, -1)
        target = target.view(B, -1)
        return {**loss_stats, **metric_stats, **{'reconstruction': (prediction, target)}}

    def validate_step(self, minibatch: Any):
        x = minibatch.text

        target = x[0][:, 1:].contiguous().view(-1)
        seq_len = x[1]
        # Initialize hidden state for rnn models
        B = x[0].size(0)
        self.initialize_hidden_state(B, self.device)

        # Evaluate model
        logits, _ = self.forward(x)
        loss_stats = self.loss(logits, target, seq_len)
        metric_stats = self.metric(logits, target, seq_len)

        prediction = logits.argmax(dim=1).view(B, -1)
        target = target.view(B, -1)
        return {**loss_stats, **metric_stats, **{'reconstruction': (prediction, target)}}

    def new_stats(self) -> Dict:
        stats = dict()
        stats['loss'] = 0
        stats['cross_entropy'] = 0
        return stats

    def initialize_hidden_state(self, batch_size, device, enc=True, dec=True):
        if enc and self.encoder.is_recurrent:
            self.encoder.initialize_hidden_state(batch_size, device)
        if dec and self.decoder.is_recurrent:
            self.decoder.initialize_hidden_state(batch_size, device)

    def detach_history(self, enc=True, dec=True):
        if self.encoder.is_recurrent and enc:
            self.encoder.reset_history()
        if self.decoder.is_recurrent and dec:
            self.decoder.reset_history()
