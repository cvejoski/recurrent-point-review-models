import torch as torch
from torch import nn as nn
from gentext.models.helper_functions import vq, vq_st, svq_st


class SampleMultinomial(torch.distributions.Multinomial):
    """
    The same as torch.distributions.Multinomial,
        but sample() returns samples directly, not counts
    """

    def __init__(self, **kwargs):
        super(SampleMultinomial, self).__init__(**kwargs)

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        samples = self._categorical.sample(torch.Size((self.total_count,)) + sample_shape)
        # samples.shape is (total_count, sample_shape, batch_shape), need to change it to
        # (sample_shape, batch_shape, total_count)
        return samples

 