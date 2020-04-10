from gentext.data import datasets
from torchtext import data
from torchtext.vocab import Vectors

TEXT = data.Field(lower=True, init_token='<sos>', eos_token='<eos>', include_lengths=True)

train, valid, test = datasets.BarabasiRandom.splits(TEXT, root="./data")

print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0])['text'])

# load the custom vectors
custom_vectors = Vectors("barabasi_random.word2vec.1000.30d.txt", "./embeddings")

# build the vocabulary
TEXT.build_vocab(train, vectors=custom_vectors)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))

# make iterator for splits
train_iter, val_iter = data.BucketIterator.splits(
        (train, valid),
        batch_size=64,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False
)
test_iter = data.Iterator(test, batch_size=64, sort=False, sort_within_batch=False, repeat=False)

batch = next(iter(train_iter))
print(batch.text)

# # Approach 2:
TEXT = data.Field(lower=True, init_token='<sos>', eos_token='<eos>')
train_iter, valid_iter, test_iter = datasets.BarabasiRandom.iters(TEXT, root="./data", vectors=custom_vectors)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
