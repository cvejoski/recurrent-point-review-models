# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import itertools
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import random
from random import randint

from gentext.models.languagemodels import WAE, DiscreteWAE
from tyche.utils.helper import create_instance, load_params, expand_params, get_device
from InferSent import models


def train_params(params, resume, debug=False, no_dirs=True):
    """initialise model trainer with saved model and parameters"""

    if debug:
        torch.manual_seed(params["seed"])

    print("Name of the Experiment: " + params['name'])
    device = get_device(params)

    # Data loader
    data_loader = create_instance('data_loader', params, device)

    #test for changed classes
    if params["model"]["args"]["decoder"]["name"] == "LanguageModelCNN":
        params["model"]["args"]["decoder"]["name"] = "LMDecoderCNN"

    # Model
    model = create_instance('model', params, data_loader.vocab, data_loader.fix_len)
    # Optimizers
    optimizers = dict()
    if isinstance(model, WAE) or isinstance(model, DiscreteWAE):
        model_params = itertools.chain(*[model.encoder.parameters(), model.decoder.parameters()])
        optimizer = create_instance('optimizer', params, model_params)
        critic_optimizer = create_instance('critic_optimizer', params, model.wasserstein_distance.parameters())
        optimizers['loss_optimizer'] = optimizer
        optimizers['critic_optimizer'] = critic_optimizer
    else:
        optimizer = create_instance('optimizer', params, model.parameters())
        optimizers['loss_optimizer'] = optimizer

    #rename dics to garbage folder
    if no_dirs:
        params['trainer']['save_dir'] = params['trainer']['save_dir'].replace("results", "garbage")
        params['trainer']['logging']['tensorboard_dir'] = params['trainer']['logging']['tensorboard_dir'].replace("results", "garbage")
        params['trainer']['logging']['logging_dir'] = params['trainer']['logging']['logging_dir'].replace("results", "garbage")

    # Trainer
    trainer = create_instance('trainer', params, model, optimizers, resume, params, data_loader)

    return trainer


def initialise_model(path_to_best_model):
    """initialise parameters and model"""
    resume = path_to_best_model + "/best_model.pth"
    yaml_path = path_to_best_model.replace("saved", "logging/raw") + "/config.yaml"
    params = load_params(yaml_path)

    return params, resume


def number_trainable_parameters(model):
    """count trainable parameters in model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return int(num_params)


def inference(trainer, z_sample, input_sentence, max_length, input_length):
    sentence = ""

    for i in range(max_length-1):
        input_tensor = (torch.tensor(input_sentence, device=trainer.model.device ), input_length)
        logits, _ = trainer.model.decoder(input_tensor, z_sample)  # [B, T, V]
        logits = logits.squeeze(dim=0) # [T * B, V]
        prediction = logits.argmax(dim=1)
        word = trainer.data_loader.vocab.itos[prediction[i]]
        number = trainer.data_loader.vocab.stoi[word]
        input_sentence[0][i+1] = number
        if word == "<eos>":
            break
        if sentence != "":
            sentence += " " + word
        else:
            sentence += word
    print(sentence)
    return sentence


def inference_language_model(trainer, input_sentence, max_length, input_length):
    sentence = ""
    softmax = torch.nn.Softmax(dim=1)
    trainer.model.initialize_hidden_state(1, trainer.model.device)
    for i in range(max_length-1):
        input_tensor = (torch.tensor(input_sentence, device=trainer.model.device ), torch.tensor([input_length]))
        logits = trainer.model(input_tensor)  # [B, T, V]
        prob_logits = softmax(logits)
        random_num = random.random()
        sum = 0
        voc_number = -1
        while sum < random_num:
            voc_number += 1
            if voc_number == prob_logits.size(1):
                sum = random_num
                voc_number -= 1
            else:
                sum += prob_logits[i][voc_number]
        prediction = voc_number
        word = trainer.data_loader.vocab.itos[prediction]
        number = trainer.data_loader.vocab.stoi[word]
        input_sentence[0][i+1] = number
        if word == "<eos>":
            break
        if sentence != "":
            sentence += " " + word
        else:
            sentence += word
    print(sentence)
    return sentence


def init_infersent(path_to_infersent):
    # Load model
    model_version = 1
    MODEL_PATH = path_to_infersent + "encoder/infersent%s.pkl" % model_version
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = models.InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Keep it on CPU or put it on GPU
    use_cuda = True
    model = model.cuda() if use_cuda else model

    # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
    W2V_PATH = path_to_infersent + 'GloVe/glove.840B.300d.txt' if model_version == 1 else path_to_infersent + 'fastText/crawl-300d-2M.vec'
    model.set_w2v_path(W2V_PATH)

    # Load embeddings of K most frequent words
    model.build_vocab_k_words(K=100000)

    return model


def generate_sample_from_test(trainer):
    sentences = []
    for batch_idx, data in enumerate(trainer.data_loader.test):
        text = data.text
        for list_numbers in text[0]:
            sentence = numbers_to_sentence(trainer, list_numbers)
            sentences.append(sentence)
    return sentences


def numbers_to_sentence(trainer, list_numbers):
    sentence = ""
    for number in list_numbers:
        if number not in [1, 2, 3]:
            word = trainer.model.vocab.itos[number]
            if sentence != "":
                sentence += " " + word
            else:
                sentence += word
    return sentence


def embed_sentences(infersent_model, sentences):

    # Load some sentences

    print("encoding sentences")
    embeddings = infersent_model.encode(sentences, bsize=128, tokenize=False, verbose=True)
    print('nb sentences encoded : {0}'.format(len(embeddings)))
    return embeddings


def calculate_frechet_distance(embedding_data, embedding_model):
    list= [[4.0, 2.0, 0.6],
           [4.2, 2.1, 0.59],
           [3.9, 2.0, 0.58],
           [4.3, 2.1, 0.62],
           [4.1, 2.2, 0.63]]
    np_embedding_data = np.array(embedding_data)
    np_embedding_model = np.array(embedding_model)

    print("check right dimensions for mean")
    mean_data = np.mean(np_embedding_data, 0)
    mean_model = np.mean(np_embedding_model, 0)

    print("check right dimensions for cov")
    cov_data = np.cov(np_embedding_data, rowvar=False)
    cov_model = np.cov(np_embedding_model, rowvar=False)

    cov_mul = 2*((np.dot(cov_data, cov_model))**0.5)

    mean1 = np.mean(list, 0)
    print(mean1)
    cov1 = np.cov(list, rowvar=False)
    print(cov1)
    trace = np.trace(cov1)
    print(trace)

    formula = np.linalg.norm(mean1, 2)
    print(formula)
    FID = np.linalg.norm(mean_data - mean_model, 2)**2 + np.trace(cov_data + cov_model - cov_mul)
    return FID


def sample_sentences_lm(path, name, number_of_samples):
    os.makedirs(path, exist_ok=True)
    path_to_file = path + "/" + name
    with open(path_to_file, 'a') as fp:
        for i in tqdm(range(number_of_samples)):
            #z_sample = torch.randn([1, 32], device=trainer.model.device)

            input_sentence = [[5] * 39]
            input_sentence[0][0] = 2
            sentence = inference_language_model(trainer, input_sentence, 39, 39)
            if sentence != " ":
                fp.write(sentence + "\n")


def print_latex_lm(test_epoch_stats, trainable_params):
    print("{:10.2f} & {:10.2f} & {:10.0f} & {} \\\\".format(test_epoch_stats["cross_entropy"],
                                                            test_epoch_stats["loss"],
                                                            test_epoch_stats["perplexity"],
                                                            int(trainable_params/1000000)))


def print_latex_lvm(test_epoch_stats, trainable_params):
    print("{:10.2f} & {:10.2f}({:0.3f}) & {:10.0f} & {} \\\\".format(test_epoch_stats["cross_entropy"],
                                                            test_epoch_stats["loss"],
                                                            test_epoch_stats["distance"],
                                                            test_epoch_stats["perplexity"],
                                                            int(trainable_params/1000000)))

if __name__ == '__main__':

    #calculate_frechet_distance(None, None)
    params, resume = initialise_model("/opt/mlfta/ansmodels/results_vae/saved/ptb_small_vae_rnn2cnn_annealing_80k_optimizer_lr_0.00075_optimizer_betas_(0.9, 0.999)/0901_195327")

    trainer = train_params(params, resume)
    valid_epoch_stats = trainer._validate_epoch(epoch=0)
    test_epoch_stats = trainer._test_epoch(epoch=0)
    print(valid_epoch_stats)
    print(test_epoch_stats)

    #sample_sentences_lm("/opt/mlfta/ansmodels/sample", "ptb_test.txt", 30000)



    # path_to_infersent = "/home/bit/stenzel/project001/InferSent/"
    # infersent_model = init_infersent(path_to_infersent)
    # embedding = embed_sentences(infersent_model, inference_sentences)
    #
    # with open('embedding.p', 'wb') as fp:
    #     pickle.dump(inference_sentences, fp)
    #
    trainable_params = number_trainable_parameters(trainer.model)
    table_results = {"validation": valid_epoch_stats, "test": test_epoch_stats, "trainable params": trainable_params}


    print_latex_lvm(test_epoch_stats, trainable_params)


    path_to_table_results = "/opt/mlfta/ansmodels/table_vae_jsons"
    os.makedirs(path_to_table_results, exist_ok=True)
    with open(path_to_table_results + "/{}.json".format(trainer.params["name"]), 'w') as fp:
       json.dump(table_results, fp)
    trainer.__del__()





    # z_sample = torch.randn([1, 32], device=trainer.model.device)
    # z_sample = torch.zeros([1, 32], device=trainer.model.device)
    #
    # z_sample
    #
    # input2 = sentence_to_list("when their changes are completed and after they have worked")
    # input2
    #
    # input_sentence = [[2, 76, 59, 527, 33, 811, 13, 86, 46, 41, 1400, 1, 1, 1, 1]]
    # trainer.model.vocab.itos[1]
    #
    # import numpy
    # m = numpy.array(input_sentence)
    # m.shape
    #
    # inference(trainer, z_sample, input_sentence)
    #
    # params = load_params(
    #     "/home/bit/stenzel/results/logging/raw/bit_ptb_wae_cnn2cnn_sentence_length_15_data_loader_batch_size_32/0911_154304/config.yaml")
    # resume = "/home/bit/stenzel/results/saved/bit_ptb_wae_cnn2cnn_sentence_length_15_data_loader_batch_size_32/0911_154304/best_model.pth"
    #
    # params2 = load_params(
    #     "/home/bit/stenzel/results/logging/raw/bit_ptb_wae_cnn2cnn_sentence_length_15_batch_size_128_model_latent_dim_16/0911_180812/config.yaml")
    # resume2 = "/home/bit/stenzel/results/saved/bit_ptb_wae_cnn2cnn_sentence_length_15_batch_size_128_model_latent_dim_16/0911_180812/best_model.pth"
    #
    # params3 = load_params("/home/bit/stenzel/project001/GENTEXT/experiments/ptb/wae_cnn2cnn.yaml")
    # gs_params = expand_params(params3)



    #for search in gs_params:

    #    trainer = train_params(search, resume=None)


    #
    #
    # def reconstruct(trainer, z_sample, input_sentence):
    #     sentence = []
    #     input_tensor = (torch.tensor(input_sentence, device=trainer.model.device), 15)
    #     logits, _ = trainer.model.decoder(input_tensor, z_sample)  # [B, T, V]
    #     logits = logits.squeeze(dim=0)  # [T * B, V]
    #     prediction = logits.argmax(dim=1)
    #     for num in prediction:
    #         word = trainer.model.vocab.itos[num]
    #         sentence.append(word)
    #
    #     print(sentence)
    #
    #
    # def sentence_to_list(sentence):
    #     list_of_numbers = [2]
    #     for word in sentence.split(" "):
    #         number = trainer.model.vocab.stoi[word]
    #         list_of_numbers.append(number)
    #
    #     return [list_of_numbers]

    # def inference(trainer, z_sample, input_sentence):
    #
    #     sentence = []
    #
    #     for i in range(14):
    #         print("input_sentence {}".format(input_sentence))
    #         input_tensor = (torch.tensor(input_sentence, device=trainer.model.device), 30)
    #         logits, _ = trainer.model.decoder(input_tensor, z_sample)  # [B, T, V]
    #         print("logits befroe squeeze {}".format(logits.shape))
    #         logits = logits[:, :-1].contiguous().view(-1, trainer.model.voc_dim)  # [T * B, V]
    #         print("logits after squeeze {}".format(logits.shape))
    #         prediction = logits.argmax(dim=1)
    #         print("prediction {}".format(prediction))
    #         word = trainer.model.vocab.itos[prediction[i]]
    #         print(word)
    #         sentence.append(word)
    #         number = trainer.model.vocab.stoi[word]
    #         input_sentence[0][i + 1] = number
    #
    #     print(sentence)

