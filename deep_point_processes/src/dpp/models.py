from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Exponential
from tyche.models import AModel
from tyche.utils.helper import to_one_hot, quadratures, create_instance, sum_dictionares, free_params, frozen_params

from .blocks import RNN


class ARTPP(AModel):
    def __init__(self, data_loader, ignore_index, **kwargs):
        super(ARTPP, self).__init__(**kwargs)
        input_dim = kwargs.pop("input_dim")
        self.lang_e_size = kwargs.get('language_embedding_size', 0)
        self.e_size = kwargs.pop('embedding_size')
        self.output_dim = kwargs.pop('output_dim')
        self.t_max = data_loader.t_max
        self.ignore_index = ignore_index
        for m in self.metrics:
            m.ignore_index = self.ignore_index
        input_dim = self.__init_mark_layers_if_is_marks(input_dim)

        self.__rnn = RNN(input_dim, **kwargs)
        self.__output_layer = nn.Sequential(nn.Linear(self.__rnn.hidden_size, 100),
                                            nn.ReLU(),
                                            nn.Linear(100, self.output_dim),
                                            nn.ReLU())
        self.__bow_layer = nn.Linear(self.__rnn.hidden_size, self.lang_e_size)
        self.__forward = self.__forward_with_marks if self.is_marks else self.__forwad_without_marks

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None) -> Dict:
        N = 0.0
        for ix, data in enumerate(minibatch):
            optimizer['loss_optimizer'].zero_grad()
            _, seq_len = data.time
            if seq_len.sum() == 0:
                continue
            B = seq_len.size(0)
            if ix == 0:
                self.initialize_hidden_state(B)
            t, bow_logits = self.forward(data)
            loss_stats = self.loss(t, bow_logits, data)
            metrics_stats = self.metric(1. / (t + 1e-6), bow_logits, data)
            loss_stats['loss'].backward()
            optimizer['loss_optimizer'].step()
            self.detach_history()

            N += seq_len.sum().item()

        loss_stats['loss'] /= N
        loss_stats['time_likelihood'] /= N
        loss_stats['reconstruction_likelihood'] /= N
        metrics_stats['MSELoss'] /= N
        metrics_stats['MSELoss'] *= self.t_max ** 2
        metrics_stats['perplexity'] = torch.exp(loss_stats['reconstruction_likelihood'])
        return {**loss_stats, **metrics_stats}

    def validate_step(self, minibatch: Any) -> Dict:
        _, seq_len = minibatch.time
        N = float(seq_len.sum())
        B = seq_len.size(0)
        self.initialize_hidden_state(B)
        t, y = self.forward(minibatch)
        loss_stats = self.loss(t, y, minibatch)
        metrics_stats = self.metric(1. / (t + 1e-6), y, minibatch)
        loss_stats['loss'] /= N
        loss_stats['time_likelihood'] /= N
        loss_stats['reconstruction_likelihood'] /= N
        metrics_stats['MSELoss'] /= N
        metrics_stats['MSELoss'] *= self.t_max ** 2
        metrics_stats['perplexity'] = torch.exp(loss_stats['reconstruction_likelihood'])
        return {**loss_stats, **metrics_stats}

    def forward(self, data, h=None, step=None):
        return self.__forward(data, h, step)

    def loss_t(self, data, t_predicted):
        _, seq_len = data.time
        _ix = seq_len.nonzero().view(-1)
        t_target = data.target_time[_ix]
        mask = t_target != self.ignore_index
        d1 = Exponential(t_predicted.squeeze(-1))
        time_likelihood = -d1.log_prob(t_target[_ix])[mask].sum()

        stats = self.new_stats()
        stats['loss'] = time_likelihood
        stats['time_likelihood'] = time_likelihood

        return stats

    def loss_t_val(self, data, t_predicted, step: int):
        _, seq_len = data.time
        seq_len = torch.clamp(seq_len / (step + 1), 0, 1)
        _ix = seq_len.nonzero().view(-1)
        t_target = data.target_time[_ix, step, None]
        mask = t_target != self.ignore_index
        d1 = Exponential(t_predicted.squeeze(-1))
        time_likelihood = -d1.log_prob(t_target[_ix])[mask].sum()

        stats = self.new_stats()
        stats['loss'] = time_likelihood
        stats['time_likelihood'] = time_likelihood

        return stats

    def metric_t(self, predicted_t: Any, data: Any) -> Dict:
        seq_len = data.time[1]
        _ix = seq_len.nonzero().view(-1)
        with torch.no_grad():
            stats = dict()
            for m in self.metrics:
                stats[type(m).__name__] = m(predicted_t[_ix].squeeze(), data.target_time[_ix].squeeze())
            return stats

    def metric_t_val(self, predicted_t: Any, data: Any, step: int) -> Dict:
        seq_len = torch.clamp(data.time[1] / (step + 1), 0, 1)
        _ix = seq_len.nonzero().view(-1)
        with torch.no_grad():
            stats = dict()
            for m in self.metrics:
                stats[type(m).__name__] = m(predicted_t[_ix].squeeze(), data.target_time[_ix, step].squeeze())
            return stats

    def loss(self, t_predicted, bow_logits, data):
        _ix = data.time[1].nonzero().view(-1)
        t_target = data.target_time[_ix]
        mask = t_target != -1
        d1 = Exponential(t_predicted.squeeze(-1))
        time_likelihood = -d1.log_prob(t_target[_ix])[mask].sum()
        if self.is_marks:
            reconstruction_loss = -torch.mul(bow_logits, data.target_bow[_ix])
            reconstruction_loss = reconstruction_loss.sum(-1) / (data.target_bow[_ix].sum(-1) + 1)
            reconstruction_loss = reconstruction_loss[mask].sum()
        else:
            reconstruction_loss = torch.Tensor([0])

        loss = time_likelihood + reconstruction_loss

        stats = self.new_stats()
        stats['loss'] = loss
        stats['time_likelihood'] = time_likelihood
        stats['reconstruction_likelihood'] = reconstruction_loss
        return stats

    def metric(self, t: Any, bow_logits: Any, data: Any) -> Dict:
        _ix = data.time[1].nonzero().view(-1)
        with torch.no_grad():
            stats = dict()
            for m in self.metrics:
                stats[type(m).__name__] = m(t.squeeze(), data.target_time[_ix].squeeze())
            return stats

    def __init_mark_layers_if_is_marks(self, input_dim):
        if self.is_marks:
            self.__t2e = nn.Linear(self.lang_e_size, self.e_size)
            input_dim += self.e_size
        return input_dim

    def __forwad_without_marks(self, data, h):
        h = self.__rnn(data.time)
        y = self.__output_layer(h)
        return y, None

    def __forward_with_marks(self, data, h, step: int):
        t, seq_len = data.time
        if h is None:
            text_embedding = data.bow
        else:
            if step is None:
                text_embedding = h
                T = t.size(1)
                E = h.size(1)
                if T != E:
                    text_embedding = F.pad(h, (0, 0, 0, T - E), 'constant', 0)
            else:
                seq_len = torch.clamp(seq_len / (step + 1), 0, 1)
                text_embedding = h.unsqueeze(1)
                t = t[:, step, None]

        m_emb = self.__t2e(text_embedding)
        x = torch.cat((t, m_emb), 2)
        h = self.__rnn((x, seq_len))
        y = self.__output_layer(h)
        if h is None:
            bow_logits = torch.nn.functional.log_softmax(self.__bow_layer(h), dim=2)
            return y, bow_logits
        else:
            return y

    @property
    def get_hidden_states(self):
        return self.__rnn.hidden_state

    @property
    def temp_embedding_size(self):
        return self.__rnn.hidden_size

    def new_stats(self) -> Dict:
        statistics = dict()
        statistics['loss'] = 0.0
        statistics['time_likelihood'] = 0.0

        return statistics

    @property
    def is_marks(self):
        return self.lang_e_size != 0

    def initialize_hidden_state(self, batch_size):
        self.__rnn.initialize_hidden_state(batch_size, self.device)

    def detach_history(self):
        self.__rnn.reset_history()


class RMTPP(AModel):
    def __init__(self, data_loader, ignore_index, **kwargs):
        super(RMTPP, self).__init__(**kwargs)
        input_dim = kwargs.pop("input_dim")
        self.K = kwargs.pop('n_markers')
        self.e_size = kwargs.pop('embedding_size')
        self.inv_sampling_size = kwargs.pop('inv_sampling_size')
        self.t_max = data_loader.t_max
        self.ignore_index = ignore_index
        for m in self.metrics:
            m.ignore_index = self.ignore_index
        input_dim = self.__init_mark_layers_if_is_marks(input_dim)

        self.__rnn = RNN(input_dim, **kwargs)
        self.__past_influence = nn.Linear(self.__rnn.hidden_size, 1)
        self.__w = nn.Parameter(torch.randn(1, requires_grad=True))
        self.__loss = self.__loss_time_mark if self.is_marks else self.__loss_time

    def intensity(self, t, h):
        return torch.exp(torch.add(self.__past_influence(h), torch.mul(self.__w, t)))

    def __inverse_sampling(self, t, h, size):
        with torch.no_grad():
            a = self.__past_influence(h)
            B, L = t.size(0), t.size(1)
            y = torch.rand((B, L, size), device=self.device)
            s = torch.log(-1. / (y - 1.))
            dt = (-a + torch.log(torch.abs(self.__w * (s + torch.exp(a) / self.__w)))) / self.__w
            dt = dt.mean(dim=-1)
        return dt

    def __prediction(self, t, h):
        def f(x):
            return 1. / (x * x) * torch.exp(-self.__nll_arrivals(1. / x - t, h))

        a = torch.ones_like(t) / 1e64
        r = quadratures(f, a, 1. / t, 40)
        return r

    def forward(self, data):
        return None, None

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        N = 0.0
        for ix, data in enumerate(minibatch):
            optimizer['loss_optimizer'].zero_grad()
            _, seq_len = data.time
            if seq_len.sum() == 0:
                continue
            B = seq_len.size(0)
            if ix == 0:
                self.initialize_hidden_state(B)
            # t, y = self.forward(data)
            loss_stats, t, y = self.loss(data)
            metrics_stats = self.metric(t, y, data)
            loss_stats['loss'].backward()
            optimizer['loss_optimizer'].step()
            self.detach_history()

            N += seq_len.sum().item()

        loss_stats['loss'] /= N
        # loss_stats['time_likelihood'] /= N
        metrics_stats['MSELoss'] /= N
        metrics_stats['MSELoss'] *= self.t_max ** 2
        return {**loss_stats, **metrics_stats}

    def validate_step(self, minibatch: Any) -> Dict:
        _, seq_len = minibatch.time
        N = float(seq_len.sum())
        B = seq_len.size(0)
        self.initialize_hidden_state(B)
        loss_stats, t, y = self.loss(minibatch)
        metrics_stats = self.metric(t, y, minibatch)

        loss_stats['loss'] /= N
        # loss_stats['time_likelihood'] /= N
        metrics_stats['MSELoss'] /= N
        metrics_stats['MSELoss'] *= self.t_max ** 2
        return {**loss_stats, **metrics_stats}

    def loss(self, data):
        h, loss, marks_logits = self.__loss(data)
        x, seq_len = data.time
        _ix = seq_len.nonzero().view(-1)
        dy = self.__inverse_sampling(x[_ix, :, 0], h, self.inv_sampling_size)
        stats = self.new_stats()
        stats['loss'] = loss
        stats['time_likelihood'] = loss
        return stats, dy, marks_logits

    def metric(self, t: Any, y: Any, data: Any) -> Dict:
        _ix = data.time[1].nonzero().view(-1)
        with torch.no_grad():
            stats = dict()
            for m in self.metrics:
                stats[type(m).__name__] = m(t[_ix].squeeze(), data.target_time[_ix].squeeze())
            return stats

    def __loss_time(self, data):
        x, seq_len = data.time
        _ix = seq_len.nonzero().view(-1)
        y_t = data.target_time[_ix]
        x_t = x[_ix]

        h = self.__rnn((x_t, seq_len))

        nll_arrivals = self.__nll_arrivals(y_t.unsqueeze(-1), h)

        mask = y_t != self.ignore_index
        nll_arrivals = nll_arrivals[mask].sum()

        return h, nll_arrivals, None

    def __loss_time_mark(self, data):
        x, seq_len = data.time
        _ix = seq_len.nonzero().view(-1)
        x_t = x[_ix]
        y_t = data.target_time[_ix, None]
        x_m = data.mark[_ix]
        y_m = data.target_mark[_ix]
        one_h = to_one_hot(x_m, self.K)
        m_emb = self.__m2e(one_h)
        x = torch.cat((x_t, m_emb), 2)
        h = self.__rnn((x, seq_len))
        marks_logits = self.__marks_logit(h)
        nll_marks = self.__nll_marks(y_m, marks_logits)
        nll_arrivals = self.__nll_arrivals(y_t.unsqueeze(-1), h)
        loss = nll_marks + nll_arrivals
        mask = y_t != self.ignore_index
        return h, loss[mask].sum(), marks_logits

    def __nll_marks(self, targets, logits):
        mark_logits = logits.view(-1, self.K)
        targets = targets.contiguous().view(-1)
        nll_marks = F.cross_entropy(mark_logits, targets)

        return nll_marks

    def __nll_arrivals(self, x, h):
        past_influence = self.__past_influence(h)
        inten = past_influence + self.__w * x
        log_likelihood = inten + torch.div(torch.exp(past_influence), self.__w) - torch.div(torch.exp(inten), self.__w)
        return -log_likelihood

    def __init_mark_layers_if_is_marks(self, input_dim):
        if self.is_marks:
            self.__m2e = nn.Linear(self.K, self.e_size)
            input_dim += self.e_size
            self.__marks_layer = nn.Linear(self.__rnn.hidden_size, self.K)
        return input_dim

    def new_stats(self) -> Dict:
        statistics = dict()
        statistics['loss'] = torch.Tensor([0.0])
        statistics['time_likelihood'] = torch.Tensor([0.0])
        statistics['mark_likelihood'] = torch.Tensor([0.0])

        return statistics

    @property
    def is_marks(self):
        return self.K != 0

    def initialize_hidden_state(self, batch_size):
        self.__rnn.initialize_hidden_state(batch_size, self.device)

    def detach_history(self):
        self.__rnn.reset_history()


class BoWTPP(AModel):
    """
    This is the models used in the AAAI workshop
    """

    def __init__(self, data_loader, ignore_index, **kwargs):
        super().__init__(**kwargs)
        self.lang_e_size = kwargs.get('language_embedding_size')
        input_dim = kwargs.pop("input_dim")
        self.e_size = kwargs.get('embedding_size')
        self.inv_sampling_size = kwargs.pop('inv_sampling_size')

        self.t_max = data_loader.t_max
        self.ignore_index = ignore_index
        for m in self.metrics:
            m.ignore_index = self.ignore_index

        self.__rnn = RNN(input_dim + self.e_size, **kwargs)
        self.__w = nn.Parameter(torch.randn(1, requires_grad=True))
        self.__t2e = nn.Linear(self.lang_e_size, self.e_size)  # text to embeding layer
        self.__l_lambda = nn.Linear(self.__rnn.hidden_size, 1)
        self.__bow_linear = nn.Linear(self.__rnn.hidden_size, self.lang_e_size)

    def past_influence(self, h):
        return self.__l_lambda(h)

    def intensity(self, t, h):
        return torch.exp(torch.add(self.past_influence(h), torch.mul(self.__w, t)))

    def __inverse_sampling(self, t, h):
        with torch.no_grad():
            a = self.past_influence(h)
            B, L = t.size(0), t.size(1)
            y = torch.rand((B, L, self.inv_sampling_size), device=self.device)
            s = torch.log(-1. / (y - 1.))
            dt = (-a + torch.log(torch.abs(self.__w * (s + torch.exp(a) / self.__w)))) / self.__w
            dt = dt.mean(dim=-1)
        return dt

    def __prediction(self, t, h):
        def f(x):
            return 1. / (x * x) * torch.exp(-self.__nll_arrivals(1. / x - t, h))

        a = torch.ones_like(t) / 1e32
        r = quadratures(f, a, 1. / t, 20)
        return r

    def forward(self, data):
        return None, None

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        N = 0.0
        for ix, data in enumerate(minibatch):
            optimizer['loss_optimizer'].zero_grad()
            _, seq_len = data.time
            if seq_len.sum() == 0:
                continue
            B = seq_len.size(0)
            if ix == 0:
                self.initialize_hidden_state(B)
            # t, y = self.forward(data)
            loss_stats, t, y = self.loss(data)
            metrics_stats = self.metric(t, y, data)
            loss_stats['loss'].backward()
            optimizer['loss_optimizer'].step()
            self.detach_history()

            N += seq_len.sum().item()

        loss_stats['loss'] /= N
        loss_stats['time_likelihood'] /= N
        loss_stats['reconstruction_likelihood'] /= N
        metrics_stats['MSELoss'] /= N
        metrics_stats['MSELoss'] *= self.t_max ** 2
        metrics_stats['perplexity'] = torch.exp(loss_stats['reconstruction_likelihood'])
        return {**loss_stats, **metrics_stats}

    def validate_step(self, minibatch: Any) -> Dict:
        _, seq_len = minibatch.time
        N = float(seq_len.sum())
        B = seq_len.size(0)
        self.initialize_hidden_state(B)
        loss_stats, t, y = self.loss(minibatch)
        metrics_stats = self.metric(t, y, minibatch)

        loss_stats['loss'] /= N
        loss_stats['time_likelihood'] /= N
        loss_stats['reconstruction_likelihood'] /= N
        metrics_stats['MSELoss'] /= N
        metrics_stats['MSELoss'] *= self.t_max ** 2
        metrics_stats['perplexity'] = torch.exp(loss_stats['reconstruction_likelihood'])
        return {**loss_stats, **metrics_stats}

    def loss_t(self, data, text_emb):
        t, seq_len = data.time
        _ix = seq_len.nonzero().view(-1)
        t_target = data.target_time[_ix]
        mask = t_target != self.ignore_index
        T = t.size(1)
        E = text_emb.size(1)
        if T != E:
            text_emb = F.pad(text_emb, (0, 0, 0, T - E), 'constant', 0)

        text_emb = self.__t2e(text_emb)  # B, T, E_SIZE
        x = torch.cat((t, text_emb), 2)  # B, T, D + E_SIZE
        h = self.__rnn((x, seq_len))

        nll_arrivals = self.__nll_arrivals(t_target.unsqueeze(-1), h)
        nll_arrivals = nll_arrivals[mask].sum()
        dy = self.__inverse_sampling(t[_ix, 0].unsqueeze(1), h)

        stats = self.new_stats()
        stats['loss'] = nll_arrivals
        stats['time_likelihood'] = nll_arrivals

        return stats, dy

    def loss_t_val(self, data, text_emb, step: int):
        t, seq_len = data.time
        seq_len = torch.clamp(seq_len / (step + 1), 0, 1)
        _ix = seq_len.nonzero().view(-1)
        t_target = data.target_time[_ix, step, None]
        mask = t_target != self.ignore_index

        t = t[:, step]
        text_emb = self.__t2e(text_emb)  # B, T, E_SIZE
        x = torch.cat((t, text_emb), 1).unsqueeze(1)  # B, 1, D + E_SIZE
        h = self.__rnn((x, seq_len))

        nll_arrivals = self.__nll_arrivals(t_target.unsqueeze(-1), h)
        nll_arrivals = nll_arrivals[mask].sum()
        dy = self.__inverse_sampling(t[_ix, 0].unsqueeze(1), h)

        stats = self.new_stats()
        stats['loss'] = nll_arrivals
        stats['time_likelihood'] = nll_arrivals

        return stats, dy

    def metric_t(self, t: Any, data: Any) -> Dict:
        seq_len = data.time[1]
        _ix = seq_len.nonzero().view(-1)
        with torch.no_grad():
            stats = dict()
            for m in self.metrics:
                stats[type(m).__name__] = m(t[_ix].squeeze(), data.target_time[_ix].squeeze())
            return stats

    def metric_t_val(self, t: Any, data: Any, step: int) -> Dict:
        seq_len = torch.clamp(data.time[1] / (step + 1), 0, 1)
        _ix = seq_len.nonzero().view(-1)
        with torch.no_grad():
            stats = dict()
            for m in self.metrics:
                stats[type(m).__name__] = m(t[_ix].squeeze(), data.target_time[_ix, step].squeeze())
            return stats

    def loss(self, data):
        t, seq_len = data.time
        _ix = seq_len.nonzero().view(-1)
        y_t = data.target_time[_ix]
        mask = y_t != self.ignore_index

        text_emb = self.__t2e(data.bow)  # B, T, E_SIZE
        x = torch.cat((t, text_emb), 2)  # B, T, D + E_SIZE
        h = self.__rnn((x, seq_len))

        bow_logits = torch.nn.functional.log_softmax(self.__bow_linear(h), dim=2)

        nll_arrivals = self.__nll_arrivals(y_t.unsqueeze(-1), h)
        nll_arrivals = nll_arrivals[mask].sum()
        dy = self.__inverse_sampling(t[_ix, :, 0], h)

        reconstruction_loss = -torch.mul(bow_logits, data.target_bow[_ix])
        reconstruction_loss = reconstruction_loss.sum(-1) / (data.target_bow[_ix].sum(-1) + 1)
        reconstruction_loss = reconstruction_loss[mask].sum()

        loss = nll_arrivals + reconstruction_loss

        stats = self.new_stats()
        stats['loss'] = loss
        stats['time_likelihood'] = nll_arrivals
        stats['reconstruction_likelihood'] = reconstruction_loss

        return stats, dy, bow_logits

    def metric(self, t: Any, y: Any, data: Any) -> Dict:
        _ix = data.time[1].nonzero().view(-1)
        with torch.no_grad():
            stats = dict()
            for m in self.metrics:
                stats[type(m).__name__] = m(t[_ix].squeeze(), data.target_time[_ix].squeeze())
            return stats

    def __nll_marks(self, marks, logits):
        mark_logits = logits.view(-1, self.K)
        targets = marks[:, :, -1].contiguous().view(-1)
        nll_marks = F.cross_entropy(mark_logits, targets, reduction='sum')
        return nll_marks

    def __nll_arrivals(self, x, h):
        past_influence = self.past_influence(h)
        inten = past_influence + self.__w * x
        log_likelihood = inten + torch.div(torch.exp(past_influence), self.__w) - torch.div(torch.exp(inten), self.__w)
        return -log_likelihood

    @property
    def get_hidden_states(self):
        return self.__rnn.hidden_state

    @property
    def temp_embedding_size(self):
        return self.__rnn.hidden_size

    def initialize_hidden_state(self, batch_size: int) -> None:
        self.__rnn.initialize_hidden_state(batch_size, self.device)

    def detach_history(self) -> None:
        self.__rnn.reset_history()

    def new_stats(self) -> Dict:
        statistics = dict()
        statistics['loss'] = torch.Tensor([0.0])
        statistics['time_likelihood'] = torch.Tensor([0.0])

        return statistics


class TextTPP(AModel):
    def __init__(self, data_loader, **kwargs):
        super().__init__(**kwargs)

        self.tpp_model = create_instance('tpp_model', kwargs, data_loader)
        self.language_model = create_instance('language_model', kwargs, data_loader.vocab, data_loader.text_fix_length,
                                              self.tpp_model.temp_embedding_size)
        self.attention = kwargs.get('attention')

        if self.attention:
            self.attention_layer = create_instance('attention_layer', kwargs,
                                                   self.language_model.hidden_dim,
                                                   self.tpp_model.temp_embedding_size)

    def loss(self, y: Any, y_target: Any) -> Dict:
        pass

    def metric(self, y: Any, y_target: Any) -> Dict:
        pass

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        N = 0.0
        all_loss_stats = []
        all_metrics_stats = []
        reconstruction_prediction = []
        reconstruction_target = []
        for ix, data in enumerate(minibatch):
            S = []
            _, seq_len = data.time
            if seq_len.sum() == 0:
                continue
            B = seq_len.size(0)
            T = data.time[0].size(1)
            free_params(self.language_model)
            frozen_params(self.tpp_model)
            if ix == 0:
                self.initialize_hidden_state(B)
            loss_stats = []
            metrics_stats = []
            for i in range(T):
                if torch.max(seq_len) <= i:
                    break
                text = data.text[0][:, i]  # [B, SEQ_LEN]
                text_length = data.text[1][:, i]
                target_text = text[:, 1:].contiguous().view(-1)
                h = self.tpp_model.get_hidden_states[0].squeeze(0)
                self.language_model.initialize_hidden_state(B)
                logits_text, s = self.language_model((text, text_length), h)

                mask = torch.arange(text.size(1), device=self.device)[None, :] < text_length[:, None]
                mask = mask.float()
                if self.attention:
                    s = self.attention_layer(s, h, mask)
                else:
                    s = (s * mask.unsqueeze(-1)).mean(1)

                S.append(s)

                lang_loss = self.language_model.loss(logits_text, target_text, text_length)
                lang_loss['cross_entropy'] *= text_length.size(0)
                lang_metrics = self.language_model.metric(logits_text, target_text, text_length)

                loss_stats.append(lang_loss)
                metrics_stats.append(lang_metrics)

                nozero_ix = text_length.nonzero().view(-1)
                prediction_text = logits_text.argmax(dim=1).view(B, -1)[nozero_ix]
                reconstruction_target.append(target_text.view(B, -1)[nozero_ix])
                reconstruction_prediction.append(prediction_text)

                optimizer['language_optimizer'].zero_grad()  # optimizer for the language model
                lang_loss['cross_entropy'].backward()
                optimizer['language_optimizer'].step()
                self.detach_history()

            optimizer['language_optimizer'].zero_grad()
            loss_stats = sum_dictionares(loss_stats)
            metrics_stats = sum_dictionares(metrics_stats)
            all_loss_stats.append(loss_stats)
            all_metrics_stats.append(metrics_stats)

            free_params(self.tpp_model)
            frozen_params(self.language_model)
            if ix == 0:
                self.initialize_hidden_state(B)

            S = torch.stack(S, dim=1).detach()
            time_loss, t = self.tpp_model.loss_t(data, S)
            time_metrics = self.tpp_model.metric_t(t, data)

            all_loss_stats.append(time_loss)
            all_metrics_stats.append(time_metrics)

            optimizer['loss_optimizer'].zero_grad()  # optimizer for the temporal model
            time_loss['time_likelihood'].backward()
            optimizer['loss_optimizer'].step()
            optimizer['loss_optimizer'].zero_grad()
            self.detach_history()

            N += seq_len.sum().item()

        all_loss_stats = sum_dictionares(all_loss_stats)
        all_metrics_stats = sum_dictionares(all_metrics_stats)
        all_loss_stats['loss'] /= N
        all_loss_stats['time_likelihood'] /= N
        all_loss_stats['cross_entropy'] /= N
        all_metrics_stats['cross_entropy'] = all_loss_stats['cross_entropy']
        all_metrics_stats['MSELoss'] /= N
        all_metrics_stats['MSELoss'] *= self.tpp_model.t_max ** 2
        all_metrics_stats['perplexity'] = torch.exp(all_loss_stats['cross_entropy'])
        reconstruction_prediction = torch.cat(reconstruction_prediction)
        reconstruction_target = torch.cat(reconstruction_target)
        return {**all_loss_stats, **all_metrics_stats,
                **{'reconstruction': (reconstruction_prediction, reconstruction_target)}}

    def validate_step(self, data: Any) -> Dict:
        _, seq_len = data.time
        N = float(seq_len.sum())
        B = seq_len.size(0)
        T = data.time[0].size(1)
        self.initialize_hidden_state(B)
        loss_stats = []
        metrics_stats = []
        reconstruction_prediction = []
        reconstruction_target = []
        max_seq_len = torch.max(seq_len)
        for i in range(T):
            if max_seq_len <= i:
                break
            text = data.text[0][:, i]
            text_length = data.text[1][:, i]
            target_text = text[:, 1:].contiguous().view(-1)
            self.language_model.initialize_hidden_state(B)
            h = self.tpp_model.get_hidden_states[0].squeeze(0)
            logits_text, s = self.language_model((text, text_length), h)

            mask = torch.arange(text.size(1), device=self.device)[None, :] < text_length[:, None]
            mask = mask.float()
            if self.attention:
                s = self.attention_layer(s, h, mask)
            else:
                s = (s * mask.unsqueeze(-1)).mean(1)

            lang_loss = self.language_model.loss(logits_text, target_text, text_length)
            lang_metrics = self.language_model.metric(logits_text, target_text, text_length)
            lang_loss['cross_entropy'] *= text_length.size(0)
            time_loss, t = self.tpp_model.loss_t_val(data, s, i)
            time_metrics = self.tpp_model.metric_t_val(t, data, i)

            loss_stats.append(time_loss)
            loss_stats.append(lang_loss)
            metrics_stats.append(time_metrics)
            metrics_stats.append(lang_metrics)

            nozero_ix = text_length.nonzero().view(-1)
            prediction_text = logits_text.argmax(dim=1).view(B, -1)[nozero_ix]
            reconstruction_target.append(target_text.view(B, -1)[nozero_ix])
            reconstruction_prediction.append(prediction_text)

        loss_stats = sum_dictionares(loss_stats)
        metrics_stats = sum_dictionares(metrics_stats)

        loss_stats['loss'] /= N
        loss_stats['time_likelihood'] /= N
        loss_stats['cross_entropy'] /= N
        metrics_stats['cross_entropy'] = loss_stats['cross_entropy']
        metrics_stats['MSELoss'] /= N
        metrics_stats['MSELoss'] *= self.tpp_model.t_max ** 2
        metrics_stats['perplexity'] = torch.exp(loss_stats['cross_entropy'])
        reconstruction_prediction = torch.cat(reconstruction_prediction)
        reconstruction_target = torch.cat(reconstruction_target)
        return {**loss_stats, **metrics_stats,
                **{'reconstruction': (reconstruction_prediction, reconstruction_target)}}

    def initialize_hidden_state(self, batch_size: int) -> None:
        self.tpp_model.initialize_hidden_state(batch_size)

    def detach_history(self) -> None:
        self.tpp_model.detach_history()

    def new_stats(self) -> Dict:
        statistics = dict()
        statistics['loss'] = torch.Tensor([0.0])
        statistics['time_likelihood'] = torch.Tensor([0.0])
        return statistics


class TextARPP(TextTPP):
    def __init__(self, data_loader, **kwargs):
        super().__init__(data_loader, **kwargs)

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        N = 0.0
        all_loss_stats = []
        all_metrics_stats = []
        reconstruction_prediction = []
        reconstruction_target = []
        for ix, data in enumerate(minibatch):
            S = []
            _, seq_len = data.time
            if seq_len.sum() == 0:
                continue
            B = seq_len.size(0)
            T = data.time[0].size(1)
            free_params(self.language_model)
            frozen_params(self.tpp_model)
            if ix == 0:
                self.initialize_hidden_state(B)

            loss_stats = []
            metrics_stats = []

            for i in range(T):
                if torch.max(seq_len) <= i:
                    break
                text = data.text[0][:, i]  # [B, SEQ_LEN]
                text_length = data.text[1][:, i]
                target_text = text[:, 1:].contiguous().view(-1)

                self.language_model.initialize_hidden_state(B)
                h = self.tpp_model.get_hidden_states[0].squeeze(0)
                logits_text, s = self.language_model((text, text_length), h)
                mask = torch.arange(text.size(1), device=self.device)[None, :] < text_length[:, None]
                mask = mask.float().unsqueeze(-1)
                s = (s * mask).mean(1)
                S.append(s)
                lang_loss = self.language_model.loss(logits_text, target_text, text_length)
                lang_metrics = self.language_model.metric(logits_text, target_text, text_length)
                lang_loss['cross_entropy'] *= text_length.size(0)

                loss_stats.append(lang_loss)
                metrics_stats.append(lang_metrics)

                nozero_ix = text_length.nonzero().view(-1)
                prediction_text = logits_text.argmax(dim=1).view(B, -1)[nozero_ix]
                reconstruction_target.append(target_text.view(B, -1)[nozero_ix])
                reconstruction_prediction.append(prediction_text)

                optimizer['language_optimizer'].zero_grad()  # optimizer for the language model
                lang_loss['cross_entropy'].backward()
                optimizer['language_optimizer'].step()
                self.detach_history()
            optimizer['language_optimizer'].zero_grad()
            loss_stats = sum_dictionares(loss_stats)
            metrics_stats = sum_dictionares(metrics_stats)
            all_loss_stats.append(loss_stats)
            all_metrics_stats.append(metrics_stats)

            free_params(self.tpp_model)
            frozen_params(self.language_model)
            if ix == 0:
                self.initialize_hidden_state(B)

            S = torch.stack(S, dim=1).detach()

            t_predicted = self.tpp_model.forward(data, S) + 1e-6

            time_loss = self.tpp_model.loss_t(data, t_predicted)
            time_metrics = self.tpp_model.metric_t(1. / t_predicted, data)

            all_loss_stats.append(time_loss)
            all_metrics_stats.append(time_metrics)

            optimizer['loss_optimizer'].zero_grad()  # optimizer for the temporal model
            time_loss['time_likelihood'].backward()
            optimizer['loss_optimizer'].step()
            optimizer['loss_optimizer'].zero_grad()
            self.detach_history()

            N += seq_len.sum().item()

        all_loss_stats = sum_dictionares(all_loss_stats)
        all_metrics_stats = sum_dictionares(all_metrics_stats)
        all_loss_stats['loss'] /= N
        all_loss_stats['time_likelihood'] /= N
        all_loss_stats['cross_entropy'] /= N
        all_metrics_stats['cross_entropy'] = all_loss_stats['cross_entropy']
        all_metrics_stats['MSELoss'] /= N
        all_metrics_stats['MSELoss'] *= self.tpp_model.t_max ** 2
        all_metrics_stats['perplexity'] = torch.exp(all_loss_stats['cross_entropy'])
        reconstruction_prediction = torch.cat(reconstruction_prediction)
        reconstruction_target = torch.cat(reconstruction_target)
        return {**all_loss_stats, **all_metrics_stats,
                **{'reconstruction': (reconstruction_prediction, reconstruction_target)}}

    def validate_step(self, data: Any) -> Dict:
        _, seq_len = data.time
        N = float(seq_len.sum())
        B = seq_len.size(0)
        T = data.time[0].size(1)
        self.initialize_hidden_state(B)
        loss_stats = []
        metrics_stats = []
        reconstruction_prediction = []
        reconstruction_target = []
        max_seq_len = torch.max(seq_len)
        for i in range(T):
            if max_seq_len <= i:
                break
            text = data.text[0][:, i]
            text_length = data.text[1][:, i]
            target_text = text[:, 1:].contiguous().view(-1)

            self.language_model.initialize_hidden_state(B)
            h = self.tpp_model.get_hidden_states[0].squeeze(0)
            logits_text, s = self.language_model((text, text_length), h)
            mask = torch.arange(text.size(1), device=self.device)[None, :] < text_length[:, None]
            mask = mask.float().unsqueeze(-1)
            s = (s * mask).mean(1)

            # s = s.mean(dim=1)
            t_predicted = self.tpp_model.forward(data, s, i) + 1e-6

            lang_loss = self.language_model.loss(logits_text, target_text, text_length)
            lang_metrics = self.language_model.metric(logits_text, target_text, text_length)
            lang_loss['cross_entropy'] *= text_length.size(0)
            time_loss = self.tpp_model.loss_t_val(data, t_predicted, i)
            time_metrics = self.tpp_model.metric_t_val(1. / t_predicted, data, i)

            loss_stats.append(time_loss)
            loss_stats.append(lang_loss)
            metrics_stats.append(time_metrics)
            metrics_stats.append(lang_metrics)

            nozero_ix = text_length.nonzero().view(-1)
            prediction_text = logits_text.argmax(dim=1).view(B, -1)[nozero_ix]
            reconstruction_target.append(target_text.view(B, -1)[nozero_ix])
            reconstruction_prediction.append(prediction_text)

        loss_stats = sum_dictionares(loss_stats)
        metrics_stats = sum_dictionares(metrics_stats)

        loss_stats['loss'] /= N
        loss_stats['time_likelihood'] /= N
        loss_stats['cross_entropy'] /= N
        metrics_stats['cross_entropy'] = loss_stats['cross_entropy']
        metrics_stats['MSELoss'] /= N
        metrics_stats['MSELoss'] *= self.tpp_model.t_max ** 2
        metrics_stats['perplexity'] = torch.exp(loss_stats['cross_entropy'])
        reconstruction_prediction = torch.cat(reconstruction_prediction)
        reconstruction_target = torch.cat(reconstruction_target)
        return {**loss_stats, **metrics_stats,
                **{'reconstruction': (reconstruction_prediction, reconstruction_target)}}
