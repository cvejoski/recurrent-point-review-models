from typing import Dict, Any

import numpy as np
import pylab
from tyche.trainer import BaseTrainingProcedure


# from gentext.models.helper_functions import graph_from_matrix


class TextTrainer(BaseTrainingProcedure):
    def __init__(self, model, optimizer, resume, params, data_loader, train_logger=None, **kwargs):
        super(TextTrainer, self).__init__(model, optimizer, resume, params, data_loader, train_logger,
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
        B = prediction.size(0)
        sample_size = min(B, self.num_of_rec_sentences)
        ix_ = np.random.randint(1, B, sample_size)
        field = self.data_loader.train.dataset.fields['text']
        t = field.reverse(target[ix_])
        r = field.reverse(prediction[ix_])
        log = []
        for i, j in zip(t, r):
            log_string = "Org: " + "\n\n Rec: ".join([i, j])
            log.append(log_string.replace('<unk>', '|unk|'))
        log = "\n\n ------------------------------------- \n\n".join(log)

        self.summary.add_text(tag + 'reconstruction', log, self.global_step)


BaseTrainingProcedure.register(TextTrainer)



def free_params(module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module):
    for p in module.parameters():
        p.requires_grad = False
