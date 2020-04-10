import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.modules.loss import CrossEntropyLoss, _Loss

from tyche.utils import param_scheduler as p_scheduler


def kullback_leibler(mean, sigma, reduction='mean'):
    """
    Kullback-Leibler divergence between Gaussian posterior distr.
    with parameters (mean, sigma) and a fixed Gaussian prior
    with mean = 0 and sigma = 1
    """

    kl = -0.5 * (1 + 2.0 * torch.log(sigma) - mean * mean - sigma * sigma)  # [B, D]
    skl = torch.sum(kl, dim=1)
    if reduction == 'mean':
        return torch.mean(skl)
    elif reduction == 'sum':
        return torch.sum(skl)
    else:
        return skl


class ELBO(CrossEntropyLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the :attr:`weight` argument being specified:

    .. math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch.

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', beta_scheduler=None):
        super(ELBO, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        if beta_scheduler is None:
            self.b_scheduler = p_scheduler.ConstantScheduler()
        else:
            self.b_scheduler = beta_scheduler

    def forward(self, input, target, mean, sigma, step):
        CE = super(ELBO, self).forward(input, target)
        KL = kullback_leibler(mean, sigma, reduction=self.reduction)
        beta = torch.tensor(self.b_scheduler(step))
        if beta == 0:
            loss = CE
        else:
            loss = CE + beta * KL
        return loss, CE, KL, beta


class Perplexity(CrossEntropyLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None):
        super(Perplexity, self).__init__(weight, size_average, ignore_index, reduce, 'mean')

    def forward(self, input, target):
        loss = super(Perplexity, self).forward(input, target)

        return torch.exp(loss)


class VQ(CrossEntropyLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', beta_scheduler=None):
        super(VQ, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        if beta_scheduler is None:
            self.b_scheduler = p_scheduler.ConstantScheduler()
        else:
            self.b_scheduler = beta_scheduler

    def forward(self, input, target, z_e_x, z_q_x, step):

        # Reconstruction loss
        loss_rec = super(VQ, self).forward(input, target)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_rec + loss_vq + loss_commit

        return loss, loss_rec, loss_vq, loss_commit


class GumbelLoss(CrossEntropyLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', beta_scheduler=None):
        super(GumbelLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        if beta_scheduler is None:
            self.b_scheduler = p_scheduler.ConstantScheduler()
        else:
            self.b_scheduler = beta_scheduler

    def forward(self, input, target, softmax, mean, sigma, step, epsilon=1e-20):

        # Reconstruction loss
        loss_rec = super(GumbelAELoss, self).forward(input, target)
        # KL divergence for gumbel softmax
        softmax = torch.mean(softmax, dim=0).view(latent_dim)
        # prior: uniform over all symbols
        latent_dim = softmax.shape[-1]
        priors = torch.Tensor([1 / latent_dim] * latent_dim).detach()
        kl = torch.sum(torch.log(softmax / (priors) + epsilon))

        beta = torch.tensor(self.b_scheduler(step))
        if beta == 0:
            loss = loss_rec
        else:
            loss = loss_rec + beta * kl

        return loss_rec, loss_rec, kl, beta


class WAELoss(CrossEntropyLoss):
    """
    Wasserstein Autoencoder Loss
    """
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(WAELoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target, distance, step):
        CE = super(WAELoss, self).forward(input, target)
        beta = torch.tensor(self.b_scheduler(step))
        if beta == 0:
            loss = CE
        else:
            loss = CE + beta * distance

        return loss, CE, distance, beta


class MSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = tyche.loss.MSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean', ignore_index=None):
        super(MSELoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            ix = target != self.ignore_index
            mse = F.mse_loss(input[ix], target[ix], reduction=self.reduction)
        else:
            mse = F.mse_loss(input, target, reduction=self.reduction)
        return mse


class RMSELoss(MSELoss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = tyche.loss.RMSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean', ignore_index=None):
        super(RMSELoss, self).__init__(size_average, reduce, reduction, ignore_index)

    def forward(self, input, target):
        mse = super(RMSELoss, self).forward(input, target)
        return torch.sqrt(mse)
