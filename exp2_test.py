

from util_get_internlm_logits import get_init_logits, get_sentence_logits
from util_dataload import *
from util_networks import *
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import json
import numpy as np
import argparse
import traceback
import datetime
import time
# Code for: Testing

# Adapted from: Sanjot's Code benchmark_exp2
# Created Date: 2 April 2024
# Last Modified Date: 9 April 2024
# Last Modified Author: Shika

# Common general Imports

# File imports


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
    return


def load_model(model, aggregator, regressor, path):
    print("at checkpoint")
    checkpoint = torch.load(path)
    print("crossed checkpoint")
    model.load_state_dict(checkpoint['model']['state_dict'], strict=False)
    aggregator.load_state_dict(checkpoint['aggregator']['state_dict'])
    regressor.load_state_dict(checkpoint['regressor']['state_dict'])
    return model, aggregator, regressor

# def load_model(model, aggregator, regressor, path):
#     print("at checkpoint")
#     checkpoint = torch.load(path)
#     print("crossed checkpoint")
#     model.load_state_dict(checkpoint['model']['state_dict'], strict=False)
#     aggregator.load_state_dict(checkpoint['aggregator']['state_dict'])
#     return model, aggregator


def weight_mode(model, trainable=False):
    for param in model.parameters():
        if trainable:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    return model


def init_test_dataloaders(config):
    test_data = get_test_data(config)
    processed_test_data = CustomDataset(
        config.img_dir, test_data, model='internlm_vl')
    pooled_test_loader = DataLoader(
        processed_test_data, batch_size=16, shuffle=False)
    return pooled_test_loader


def exp2_test_function(config, test_loader, model, aggregator, device):
    predicted_scores = []
    corresponding_name = []
    corresponding_mos = []
    for sampled_batch in tqdm(test_loader):
        img_input = sampled_batch['img'].to(device)
        mos = sampled_batch['mos'].to(device)
        name = sampled_batch['name']

        # how to get hidden states from internlm model
        if config.logit_processing_type == 'init':
            hidden_states = get_init_logits(model, img_input)
        elif config.logit_processing_type == 'sentence':
            hidden_states = get_sentence_logits(
                model, img_input, config.gen_config)

        # choose the aggregator type
        if config.network_type == "regressor":
            predicted_video_scores = aggregator(hidden_states)
        elif config.network_type == "b_attn":
            predicted_video_scores = aggregator(hidden_states)
            # predicted_video_scores = predicted_video_scores['output']
        elif config.network_type == "c_attn":
            predicted_video_scores = aggregator(hidden_states)
        predicted_scores.append(predicted_video_scores.detach().cpu())
        corresponding_name.append(name)
        corresponding_mos.append(mos.detach().cpu())

        del sampled_batch, img_input, mos, hidden_states, predicted_video_scores
        torch.cuda.empty_cache()

    predicted_scores = torch.cat(
        predicted_scores, dim=0).squeeze().numpy().tolist()
    corresponding_mos = torch.cat(corresponding_mos, dim=0).squeeze().tolist()

    try:
        corresponding_name = list(np.concatenate(corresponding_name))
    except:
        corresponding_name = corresponding_name

    return predicted_scores, corresponding_mos, corresponding_name


def exp2_2_test_function(config, test_loader, model, aggregator, regressor, device):
    predicted_scores = []
    corresponding_name = []
    corresponding_mos = []
    for sampled_batch in tqdm(test_loader):
        img_input = sampled_batch['img'].to(device)
        mos = sampled_batch['mos'].to(device)
        name = sampled_batch['name']

        # how to get hidden states from internlm model
        if config.logit_processing_type == 'init':
            hidden_states = get_init_logits(model, img_input)
        elif config.logit_processing_type == 'sentence':
            hidden_states = get_sentence_logits(
                model, img_input, config.gen_config)

        predicted_video_feats = aggregator(hidden_states)
        predicted_video_scores = regressor(predicted_video_feats)

        predicted_scores.append(predicted_video_scores.detach().cpu())
        corresponding_name.append(name)
        corresponding_mos.append(mos.detach().cpu())

        del sampled_batch, img_input, mos, hidden_states, predicted_video_scores
        torch.cuda.empty_cache()

    predicted_scores = torch.cat(
        predicted_scores, dim=0).squeeze().numpy().tolist()
    corresponding_mos = torch.cat(corresponding_mos, dim=0).squeeze().tolist()

    try:
        corresponding_name = list(np.concatenate(corresponding_name))
    except:
        corresponding_name = corresponding_name

    return predicted_scores, corresponding_mos, corresponding_name


def test_model(config):
    # Set device
    torch.cuda.set_device(config.default_device)
    device = f"cuda:{config.default_device}" if torch.cuda.is_available(
    ) else "cpu"

    # Get model and aggregator
    model = get_internLM_model(config)
    if config.network_type == "regressor":
        aggregator = NormalRegressor1(config.embed_dim)
    elif config.network_type == "b_attn" or config.network_type == "attn":
        aggregator = BasicMultiheadAttentionAggregator(
            config.embed_dim, config.num_heads, regressor_bool=False)
        regressor = NormalRegressor(config.embed_dim)  # Define regressor here

    elif config.network_type == "c_attn":
        aggregator = ComplexMultiheadAttentionAggregator(
            config.embed_dim, config.num_heads)

    # Load model and aggregator
    model, aggregator, regressor = load_model(
        model, aggregator,  regressor, config.test_result_dir + 'Train/iter_5672.tar')
    convert_models_to_fp32(model)
    model = weight_mode(model, trainable=False)
    model.eval()
    aggregator = weight_mode(aggregator, trainable=False)
    convert_models_to_fp32(aggregator)
    aggregator.eval()
    aggregator.to(device)
    regressor = weight_mode(regressor, trainable=False)
    convert_models_to_fp32(regressor)
    regressor.eval()
    regressor.to(device)
    pooled_test_loader = init_test_dataloaders(config)

    with torch.no_grad():
        test_prediction, corresponding_mos, corresponding_name = exp2_2_test_function(
            config, pooled_test_loader, model, aggregator, regressor, device)

    srcc_test_correlation = spearmanr(
        np.array(test_prediction), np.array(corresponding_mos))[0]
    plcc_test_correlation = pearsonr(
        np.array(test_prediction), np.array(corresponding_mos))[0]

    print(f"Performance on {config.test_dataset} is {srcc_test_correlation}")
    print(f"Performance on {config.test_dataset} is {plcc_test_correlation}")

    return


def configuration_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--test_result_dir', type=str,
                        default='/scratch/sanjotst/ssl1_original_parameters/Run0078/')
    parser.add_argument('--test_dataset', type=list, default='pipal')
    parser.add_argument('--img_dir', type=str,
                        default='/home/sanjotst/llm_iqa/llm-iqa/datasets/CLIVE/ChallengeDB_release/Images')
    parser.add_argument('--input_json_file', type=str,
                        default='/home/sanjotst/llm_iqa/llm-iqa/labels/livec.json')
    parser.add_argument('--embed_dim', type=int,
                        default=4096)
    parser.add_argument('--network_type', type=str,
                        default='b_attn')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--default_device', type=int, default=2)
    parser.add_argument('--num_gpus', type=int, default=1)
    config = parser.parse_args()
    return config


def main():
    config = configuration_params()

    # Load the configuration details from the json
    config_path = config.test_result_dir + 'config_details.json'
    with open(config_path) as f:
        data = json.load(f)
    for key in data:
        # Only set the attribute if it doesn't already exist
        if not hasattr(config, key):
            setattr(config, key, data[key])
    test_model(config)
    return


if __name__ == '__main__':
    print('Program started at ' +
          datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' +
          datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
