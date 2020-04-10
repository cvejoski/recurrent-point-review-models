from typing import Dict, Tuple, List, Any

import numpy as np
import tqdm
from torch.nn.modules.loss import _Loss
from tyche.trainer import BaseTrainingProcedure
from tyche.utils import param_scheduler as p_scheduler
from tyche.utils.helper import create_instance, free_params, frozen_params


class PointTextTrainer(BaseTrainingProcedure):
    def __init__(self, model, optimizer, resume, params, data_loader, train_logger=None, **kwargs):
        super(PointTextTrainer, self).__init__(model, optimizer, resume, params, data_loader, train_logger,
                                               **kwargs)
        self.reconstruction_every = kwargs.pop("reconstruction_every")
        self.num_of_rec_sentences = kwargs.pop("num_rec_sentences")

    def _train_step(self, minibatch, batch_idx: int, epoch: int, p_bar) -> Dict:
        stats = super()._train_step(minibatch, batch_idx, epoch, p_bar)
        self._log_reconstruction('train/', stats['reconstruction'][0], stats['reconstruction'][1])
        del stats['reconstruction']
        return stats

    def _validate_step(self, minibatch: Any, batch_idx: int, epoch: int, p_bar) -> Dict:
        stats = super()._validate_step(minibatch, batch_idx, epoch, p_bar)
        if batch_idx == self.n_val_batches - 1:
            self._log_reconstruction('validate/', stats['reconstruction'][0], stats['reconstruction'][1], True)
        del stats['reconstruction']
        return stats

    def _log_reconstruction(self, tag, prediction, target, force_log=False):
        if self.global_step % self.reconstruction_every != 0 and not force_log:
            return
        TOTAL_SENTENCES = prediction.size(0)
        sample_size = min(TOTAL_SENTENCES, self.num_of_rec_sentences)
        ix_ = np.random.choice(np.arange(TOTAL_SENTENCES), sample_size, replace=False)
        field = self.data_loader.train.dataset.fields['text'].nesting_field
        t = field.reverse(target[ix_])
        r = field.reverse(prediction[ix_])
        log = []
        for i, j in zip(t, r):
            log_string = "Org: " + "\n\nRec: ".join([i, j])
            log.append(log_string.replace('<unk>', '|unk|'))

        log = "\n\n ------------------------------------- \n\n".join(log)
        self.summary.add_text(tag + 'reconstruction', log, self.global_step)


BaseTrainingProcedure.register(PointTextTrainer)


class TrainingWGAN(BaseTrainingProcedure):
    def __init__(self,
                 model,
                 loss,
                 metrics,
                 optimizer,
                 critic_optimizer,
                 resume,
                 params,
                 data_loader,
                 train_logger=None,
                 **kwargs):

        super(TrainingWGAN, self).__init__(model, loss, metrics, optimizer, resume, params, train_logger, data_loader)

        self.n_updates_critic = kwargs.get('n_updates_critic', 10)

        # ======== Critic ======== #

        self._critic_optimizer = critic_optimizer

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        p_bar = tqdm.tqdm(
                desc="Training batch: ", total=self.n_train_batches, unit="batch")

        epoch_stat = self.__new_stat()

        for batch_idx, input in enumerate(self.data_loader.train):
            batch_stat = self.__train_step(input, batch_idx, epoch, p_bar)
            for k, v in batch_stat.items():
                epoch_stat[k] += v
        p_bar.close()

        self.__normalize_stats(self.n_train_batches, epoch_stat)
        self.__log_epoch("train/epoch/", epoch_stat)

        return epoch_stat

    def __train_step(self, input, batch_idx: int, epoch: int, p_bar) -> Dict:

        batch_stat = self.__new_stat()

        x_real = input.double()
        x_real = x_real.view(-1, 1)

        # Critic optimization

        self._critic_optimizer.zero_grad()

        # (i) parameters

        frozen_params(self.model.generator)
        free_params(self.model.wasserstein_distance)

        # (ii) critic loss
        for _ in range(self.n_updates_critic):
            x_fake = self.model.generator(x_real)
            critic_loss = self.model.wasserstein_distance.get_critic_loss(x_real, x_fake)
            critic_loss.backward()
            self._critic_optimizer.step()

        # Encoder-decoder optimizer:

        self.optimizer.zero_grad()

        # (i) parameters

        free_params(self.model.generator)
        frozen_params(self.model.wasserstein_distance)

        # (ii) generator loss

        x_fake = self.model.generator(x_real)
        distance = self.model.wasserstein_distance(x_real, x_fake)
        distance.backward()
        self.optimizer.step()

        # Metric:

        x_r = x_real.view(-1).detach().numpy()
        x_f = x_fake.view(-1).detach().numpy()
        metrics = [m(x_r, x_f)[0] for m in self.metrics]

        self.__update_stats(distance, metrics, batch_stat)
        self._log_train_step(epoch, batch_idx, batch_stat)

        p_bar.set_postfix_str(
                "KS: {:5.4f}, w-distance: {:5.4f}".format(
                        batch_stat['KS'],
                        batch_stat['w-distance']))
        p_bar.update()
        self.global_step += 1

        return batch_stat

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        statistics = self.__new_stat()
        with torch.no_grad():
            p_bar = tqdm.tqdm(
                    desc="TEST batch: ",
                    total=self.n_val_batches,
                    unit="batch")
            for batch_idx, batch in enumerate(self.data_loader.test):
                x_real = batch.double()
                x_real = x_real.view(-1, 1)
                x_fake = self.model.generator(x_real)
                distance = self.model.wasserstein_distance(x_real, x_fake)

                x_r = x_real.view(-1).detach().numpy()
                x_f = x_fake.view(-1).detach().numpy()
                metrics = [m(x_r, x_f)[0] for m in self.metrics]

                self.__update_stats(distance, metrics, statistics)
                self._log_validation_step(epoch, batch_idx, statistics)

                p_bar.set_postfix_str(
                        "KS: {:5.4f}, w-distance: {:5.4f}".format(
                                statistics['KS'],
                                statistics['w-distance']))

                p_bar.update()
            p_bar.close()
        self.__normalize_stats(self.n_val_batches, statistics)
        self.__log_epoch("validate/epoch/", statistics)

        return statistics

    def __normalize_stats(self, n_batches, statistics):
        for k in statistics.keys():
            statistics[k] /= n_batches

    def __log_epoch(self, log_label, statistics):
        for k, v in statistics.items():
            self.summary.add_scalar(log_label + k, v, self.global_step)

    def __update_stats(self, distance, metrics, statistics):

        statistics['w-distance'] += distance / float(self.batch_size)
        for m, value in zip(self.metrics, metrics):
            n = type(m).__name__
            statistics[n] += value / float(self.batch_size)

    def __new_stat(self):
        statistics = dict()
        statistics['w-distance'] = 0.0
        for m in self.metrics:
            statistics[type(m).__name__] = 0.0

        return statistics


class TrainingInteractingPointProcess(BaseTrainingProcedure):
    def __init__(self,
                 model,
                 loss,
                 metric,
                 optimizer,
                 resume,
                 params,
                 data_loader,
                 train_logger=None,
                 **kwargs):
        super(TrainingInteractingPointProcess, self).__init__(model, loss, metric, optimizer, resume,
                                                              params, train_logger, data_loader)

        self.bptt_size = data_loader.bptt
        self.loss.b_scheduler = create_instance('beta_scheduler', params['trainer']['args'])

    def _train_step(self, input, batch_idx: int, epoch: int, p_bar):
        """
        this was at the beggining of the training loop
        encoder.train()
        decoder.train()
        scheduler.step()

        :param input:
        :param batch_idx:
        :param epoch:
        :param p_bar:
        :return:
        """
        batch_stat = self._new_stat()
        self.optimizer.zero_grad()
        data, relations = input[0], input[1]
        data, relations = data.to(self.device), relations.to(self.device)
        target = data[:, :, 1:, :]

        logits, edges, prob, output = self.model(data)
        ipp_vae_loss = self.loss(output, target, prob, self.global_step)
        ipp_vae_loss[0].backward()
        self.optimizer.step()
        self.model.detach_history()
        metrics = []
        ipp_vae_loss = [l.item() for l in ipp_vae_loss]
        self.__update_stats(ipp_vae_loss, metrics, batch_stat)
        self._log_train_step(epoch, batch_idx, batch_stat)

        # if self.global_step % 20 == 0:
        #    self.__log_reconstruction('train/batch/', prediction, target)
        p_bar.set_postfix_str(
                "loss: {:5.4f}, nll: {:5.4f}, kl: {:5.4f}".format(batch_stat['loss'],
                                                                  batch_stat['nll'],
                                                                  batch_stat['kl']))
        p_bar.update()
        self.global_step += 1
        return batch_stat

    def _validate_epoch(self, epoch):
        # ===========================================================
        # from interacting systems
        self.model.eval()
        statistics = self._new_stat()
        with torch.no_grad():
            p_bar = tqdm.tqdm(
                    desc="Validation batch: ",
                    total=self.n_val_batches,
                    unit="batch")
            for batch_idx, (data, relations) in enumerate(self.data_loader.validate):
                data, relations = data.to(self.device), relations.to(self.device)
                target = data[:, :, 1:, :]

                logits, edges, prob, output = self.model(data)
                ipp_vae_loss = self.loss(output, target, prob, self.global_step)
                metrics = []
                ipp_vae_loss = [l.item() for l in ipp_vae_loss]
                self.__update_stats(ipp_vae_loss, metrics, statistics)
                self._log_validation_step(epoch, batch_idx, statistics)
                p_bar.set_postfix_str(
                        "loss: {:5.4f}, nll: {:5.4f}, kl: {:5.4f}".format(statistics['loss'],
                                                                          statistics['nll'],
                                                                          statistics['kl']))
                p_bar.update()
            p_bar.close()
        self._normalize_stats(self.n_val_batches, statistics)
        self._log_epoch("validate/epoch/", statistics)

        return statistics

    def __calc_beta(self):
        if self.decay_type == 'constant':
            beta = self.decay_kl
        elif self.decay_type == 'exponential':
            beta = p_scheduler.exponential(self.max_decay_iter, self.global_step, self.decay_kl)
        return beta

    def __update_stats(self, vae_loss: Tuple, metrics: List[_Loss], statistics):
        """
        acc = edge_accuracy(logits, relations)

        acc_train.append(acc)
        mse_train.append(F.mse_loss(output, target).data[0])
        nll_train.append(loss_nll.data[0])
        kl_train.append(loss_kl.data[0])

        :param vae_loss:
        :param metrics:
        :param statistics:
        :return:
        """
        batch_loss = vae_loss[1] + vae_loss[2]  # loss without beta
        statistics['nll'] += vae_loss[1] / float(self.batch_size)
        statistics['kl'] += vae_loss[2] / float(self.batch_size)
        statistics['loss'] += batch_loss / float(self.batch_size)
        statistics['beta'] = vae_loss[3]
        for m, value in zip(self.metrics, metrics):
            n = type(m).__name__
            if n == 'Perplexity':
                statistics[n] += value
            else:
                statistics[n] += value / float(self.batch_size)

    def _new_stat(self):
        """
        :return:
        """
        statistics = dict()
        statistics['loss'] = 0.0
        statistics['nll'] = 0.0
        statistics['kl'] = 0.0
        statistics['beta'] = 0.0
        for m in self.metrics:
            statistics[type(m).__name__] = 0.0
        return statistics


# ==========================================================================
#  TRAINING seq2seq
# ==========================================================================
import random

from torch import optim
from dpp.utils.plots import *
from dpp.data.extra import *

teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        input_here = input_tensor[ei]
        encoder_output, encoder_hidden = encoder(
                input_here, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(input_lang, output_lang, pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
