import torch
from dpp.data.extra import *
from dpp.blocks import EncoderRNN,AttnDecoderRNN
from dpp.trainer import trainIters

if __name__=="__main__":
    input_lang, output_lang, pairs = prepareData('spa', 'eng', True)
    print(random.choice(pairs))

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(input_lang, output_lang, pairs, encoder1, attn_decoder1, 500, print_every=50)