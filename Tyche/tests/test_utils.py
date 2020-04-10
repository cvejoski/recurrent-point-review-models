import unittest
from collections import OrderedDict
from tyche.utils.helper import unpack_cv_parameters
import yaml

PARAMS = """\
name: ptb_vae_cnn2cnn
num_runs: 1
num_workers: 1
gpus: ()
model:
  module: gentext.models.languagemodels
  name: VAE
  args:
    latent_dim:
    - 16
    - 32
    encoder:
      module: gentext.models.blocks
      name: GaussianEncoderCNN
      args:
        n_highway_layers: 10
        kernel_sizes: (3, 3)
        output_channels: (7, 7)
        normalization: false
        attention: true
        n_layers_per_block: 2
    decoder:
      module: gentext.models.blocks
      name: DecoderCNN
      args:
        kernel_sizes: (3, 3)
        output_channels: (7, 7)
        normalization: false
        n_layers_per_block: 2
data_loader:
  module: tyche.data.loader
  name: DataLoaderPTB
  args:
    path_to_data: ./data
    path_to_vectors: ./data
    batch_size: 32
    emb_dim: glove.6B.100d
    voc_size: 4000
    min_freq: 1
    fix_len: -1
optimizer:
  module: torch.optim
  name: Adadelta
  args:
    lr: 1
loss:
  module: tyche.loss
  name: ELBO
  args:
    reduction: sum
metric: !!python/tuple
- module: tyche.loss
  name: Perplexity
  args: {}
trainer:
  module: tyche.trainer
  name: TrainingVAE
  args:
    decay_type: constant
    beta_scheduler:
      module: tyche.utils.param_scheduler
      name: ConstantScheduler
      args:
        beta: 0
    bm_metric: Perplexity
  epochs: 250
  save_dir: ./results/saved/
  logging:
    tensorboard_dir: ./results/logging/tensorboard/
    logging_dir: ./results/logging/raw/
    formatters:
      verbose: '%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s'
      simple: '%(levelname)s %(asctime)s %(message)s'
"""


class MyTestCase(unittest.TestCase):
    def test_expand_params(self):
        params = yaml.full_load(PARAMS)

        self.assertEqual(PARAMS, yaml.dump(params, default_flow_style=False, sort_keys=False))

    def test_unpack_cv_parameters(self):
        params = yaml.full_load(PARAMS)
        cv_params = unpack_cv_parameters(params)
        print(cv_params)
        self.assertEqual(len(cv_params[0]), 2)


if __name__ == '__main__':
    unittest.main()
