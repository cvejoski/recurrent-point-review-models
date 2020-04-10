# coding: utf-8

import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from tyche.utils.helper import create_instance, get_device, sum_dictionares

logger = logging.getLogger(__file__)


def infer_rnn(args):
    model, data_loader = load_model(args)
    field = data_loader.train.dataset.fields['text'].nesting_field
    with torch.no_grad():
        p_bar = tqdm(
                desc="Infer stress batch: ",
                total=len(data_loader.validate),
                unit="batch")
        for batch_idx, data in enumerate(data_loader.validate):

            _, seq_len = data.time
            N = float(seq_len.sum())
            B = seq_len.size(0)
            T = data.time[0].size(1)
            model.initialize_hidden_state(B)
            loss_stats = []
            metrics_stats = []
            text_prediction = []
            text_target = []
            text_sample = []
            perplexity = []
            max_seq_len = torch.max(seq_len)
            for i in range(T):
                filehandle = open('text-rnn-rnn.txt', 'a+')
                if max_seq_len <= i:
                    break
                text = data.text[0][:, i]
                text_length = data.text[1][:, i]
                target_text = text[:, 1:].contiguous().view(-1)
                model.language_model.initialize_hidden_state(B)
                h = model.tpp_model.get_hidden_states[0].squeeze(0)
                logits_text, s = model.language_model((text, text_length), h)

                model.language_model.initialize_hidden_state(B)
                input = torch.ones((B, 80), dtype=torch.long).to(model.device)
                input[:, 0] = data_loader.vocab.stoi['<sos>']

                for j in range(78):
                    length = torch.ones(B, dtype=torch.long).to(model.device) + j + 1
                    logits, _ = model.language_model((input, length), h)  # [B*T, V]
                    input[:, j + 1] = logits.argmax(dim=1).view(B, -1)[:, j]

                mask = torch.arange(text.size(1), device=model.device)[None, :] < text_length[:, None]
                mask = mask.float().unsqueeze(-1)
                s = (s * mask).mean(1)
                # s = s.mean(dim=1)

                t_predicted = model.tpp_model.forward(data, s, i) + 1e-6
                lang_loss = model.language_model.loss(logits_text, target_text, text_length)
                lang_metrics = model.language_model.metric(logits_text, target_text, text_length)
                perplexity.append(lang_metrics['perplexity'].detach().item())
                lang_loss['cross_entropy'] *= text_length.size(0)
                time_loss = model.tpp_model.loss_t_val(data, t_predicted, i)
                time_metrics = model.tpp_model.metric_t_val(1. / t_predicted, data, i)

                loss_stats.append(time_loss)
                loss_stats.append(lang_loss)
                metrics_stats.append(time_metrics)
                metrics_stats.append(lang_metrics)

                nozero_ix = text_length.nonzero().view(-1)
                prediction_text = logits_text.argmax(dim=1).view(B, -1)[nozero_ix]

                text_target.append(target_text.view(B, -1)[nozero_ix])
                text_prediction.append(prediction_text)
                text_sample.append(input[nozero_ix])
                filehandle.write(f'Batch {batch_idx} Step {i} \n\n')
                sample = field.reverse(text_sample[-1])
                filehandle.write('\n\nOut of sample \n\n')
                for ix_t, listitem in enumerate(sample):
                    filehandle.write(f'{data.time[0][ix_t, i][0].item()}\t')
                    filehandle.write('%s\n' % listitem)

                sample = field.reverse(text_prediction[-1])
                filehandle.write('\n\nPrediction \n\n')
                for ix_t, listitem in enumerate(sample):
                    filehandle.write(f'{data.time[0][ix_t, i][0].item()}\t')
                    filehandle.write('%s\n' % listitem)
                sample = field.reverse(text_target[-1])

                filehandle.write('\n\nTarget Text \n\n')
                for ix_t, listitem in enumerate(sample):
                    filehandle.write(f'{data.time[0][ix_t, i][0].item()}\t')
                    filehandle.write('%s\n' % listitem)
                filehandle.flush()
                filehandle.close()
            loss_stats = sum_dictionares(loss_stats)
            metrics_stats = sum_dictionares(metrics_stats)

            loss_stats['loss'] /= N
            loss_stats['time_likelihood'] /= N
            loss_stats['cross_entropy'] /= N
            metrics_stats['cross_entropy'] = loss_stats['cross_entropy']
            metrics_stats['MSELoss'] /= N
            metrics_stats['MSELoss'] *= model.tpp_model.t_max ** 2
            metrics_stats['perplexity'] = torch.exp(loss_stats['cross_entropy'])

            p_bar.update()

        p_bar.close()


def infer_tpp(args):
    model, data_loader = load_model(args)
    field = data_loader.train.dataset.fields['text'].nesting_field
    with torch.no_grad():
        p_bar = tqdm(
                desc="Infer stress batch: ",
                total=len(data_loader.validate),
                unit="batch")
        perplexity = []
        for batch_idx, data in enumerate(data_loader.validate):

            _, seq_len = data.time
            N = float(seq_len.sum())
            B = seq_len.size(0)
            T = data.time[0].size(1)
            model.initialize_hidden_state(B)
            loss_stats = []
            metrics_stats = []
            text_prediction = []
            text_target = []
            text_sample = []

            max_seq_len = torch.max(seq_len)
            for i in range(T):
                filehandle = open('text-rnn-rnn.txt', 'a+')
                if max_seq_len <= i:
                    break
                text = data.text[0][:, i]
                text_length = data.text[1][:, i]
                target_text = text[:, 1:].contiguous().view(-1)
                model.language_model.initialize_hidden_state(B)
                h = model.tpp_model.get_hidden_states[0].squeeze(0)
                logits_text, s = model.language_model((text, text_length), h)

                model.language_model.initialize_hidden_state(B)
                input = torch.ones((B, 80), dtype=torch.long).to(model.device)
                input[:, 0] = data_loader.vocab.stoi['<sos>']
                length = torch.ones(B, dtype=torch.long).to(model.device) * 80

                for j in range(78):
                    logits, _ = model.language_model((input, length), h)  # [B*T, V]
                    input[:, j + 1] = logits.argmax(dim=1).view(B, -1)[:, j]

                mask = torch.arange(text.size(1), device=model.device)[None, :] < text_length[:, None]
                mask = mask.float().unsqueeze(-1)
                s = (s * mask).mean(1)
                # s = s.mean(dim=1)
                lang_loss = model.language_model.loss(logits_text, target_text, text_length)
                lang_metrics = model.language_model.metric(logits_text, target_text, text_length)
                lang_loss['cross_entropy'] *= text_length.size(0)
                perplexity.append(lang_metrics['perplexity'].detach().item())
                time_loss, sample = model.tpp_model.loss_t_val(data, s, i)
                time_metrics = model.tpp_model.metric_t_val(sample, data, i)

                loss_stats.append(time_loss)
                loss_stats.append(lang_loss)
                metrics_stats.append(time_metrics)
                metrics_stats.append(lang_metrics)

                nozero_ix = text_length.nonzero().view(-1)
                prediction_text = logits_text.argmax(dim=1).view(B, -1)[nozero_ix]

                text_target.append(target_text.view(B, -1)[nozero_ix])
                text_prediction.append(prediction_text)
                text_sample.append(input[nozero_ix])
                filehandle.write(f'Batch {batch_idx} Step {i} \n\n')
                sample = field.reverse(text_sample[-1])
                filehandle.write('\n\nOut of sample \n\n')
                for ix_t, listitem in enumerate(sample):
                    filehandle.write(f'{data.time[0][ix_t, i][0].item()}\t')
                    filehandle.write('%s\n' % listitem)

                sample = field.reverse(text_prediction[-1])
                filehandle.write('\n\nPrediction \n\n')
                for ix_t, listitem in enumerate(sample):
                    filehandle.write(f'{data.time[0][ix_t, i][0].item()}\t')
                    filehandle.write('%s\n' % listitem)
                sample = field.reverse(text_target[-1])

                filehandle.write('\n\nTarget Text \n\n')
                for ix_t, listitem in enumerate(sample):
                    filehandle.write(f'{data.time[0][ix_t, i][0].item()}\t')
                    filehandle.write('%s\n' % listitem)
                filehandle.flush()
                filehandle.close()
            loss_stats = sum_dictionares(loss_stats)
            metrics_stats = sum_dictionares(metrics_stats)

            loss_stats['loss'] /= N
            loss_stats['time_likelihood'] /= N
            loss_stats['cross_entropy'] /= N
            metrics_stats['cross_entropy'] = loss_stats['cross_entropy']
            metrics_stats['MSELoss'] /= N
            metrics_stats['MSELoss'] *= model.tpp_model.t_max ** 2
            metrics_stats['perplexity'] = torch.exp(loss_stats['cross_entropy'])

            p_bar.update()

        p_bar.close()


def load_model(args):
    try:
        state = torch.load(args.model, map_location='cuda:0')
        params = state["params"]
        params['gpus'][0] = '0'
        params['data_loader']['args']['batch_size'] = 1
        dtype_ = params.get("dtype", "float32")
        dtype_ = getattr(torch, dtype_)
        torch.set_default_dtype(dtype_)
        logger.info("Name of the Experiment: " + params['name'])
        device = get_device(params)
        data_loader = create_instance('data_loader', params, device, dtype_)
        model = create_instance('model', params, data_loader)
        model.load_state_dict(state["model_state"])
        model.to(device)
        model.eval()
        return model, data_loader

    except Exception as e:
        logger.error(e)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "model",
            type=Path,
            help="path to the stored model")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    infer_rnn(args)


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
