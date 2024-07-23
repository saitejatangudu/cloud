#todo : rename this file to testing_loop.py

import time
import datetime
import traceback
import argparse
import numpy as np
import json
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
import itertools

# File imports
from util_networks import *
from util_dataload import *
from util_get_internlm_logits import *


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
    return


def load_model(model, aggregator, regressor, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model']['state_dict'], strict=False)
    aggregator.load_state_dict(checkpoint['aggregator']['state_dict'])
    regressor.load_state_dict(checkpoint['regressor']['state_dict'])
    return model, aggregator, regressor

# this is a legacy function, not sure if needed 
def weight_mode(model, trainable=False):
    for param in model.parameters():
        if trainable:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    return model


def init_test_dataloaders(self):

    # get dataset indices
    train_data, test_data = get_livefb_train_data(self.config)

    # get custom datasets based on dataset
    if self.config.dataset == 'LIVE_FB':
        processed_train_data = CustomDataset(
            df_data=train_data, img_dir=self.config.img_dir, model = self.config.model_description)
    elif self.config.dataset == 'LIVE_FB_DIST2_LEVELS5':
        processed_train_data = CustomTrainDatasetAnnotatedLIVEFB(
            df_data=train_data, annotation_matrix_dir=self.annotation_directory,  synthetic_img_dir=self.synthetic_img_dir, model = self.config.model_description)
    elif self.config.dataset == 'LIVE_FB_DIST4_LEVELS4':
        processed_train_data = CustomTrainDatasetAnnotatedLIVEFB(
            df_data=train_data, annotation_matrix_dir=self.annotation_directory,  synthetic_img_dir=self.synthetic_img_dir, model = self.config.model_description)

    # get custom dataset for proccesed train data 
    processed_test_data = CustomDataset(
        df_data=test_data, img_dir=self.config.img_dir, model = self.config.model_description)

  # get dataloaders 
    self.pooled_train_loader = DataLoader(
        dataset=processed_train_data, batch_size=self.config.batch_size, shuffle=True)
    self.pooled_test_loader = DataLoader(
        dataset=processed_test_data, batch_size=self.config.test_batch_size, shuffle=False)

    return 


def get_predictions(loader, model, aggregator, regressor, device):
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

    # Get model 
    if config.model_description == 'internlm_vl'
        model = get_internLM_model(config)
    if config.model_description == 'internlm_vl2_quantised'
        model = get_internLM_quantised_model(config)
    if config.model_description == 'internlm_vl2'
        model = get_internLM_v2_model(config)
    if config.model_description == 'llava_1.5'
        model = get_in_model(config)
    if config.model_description == 'mplug_owl_2'
        model = get_internLM_model(config)
        
    if config.network_type == "regressor":
        regressor = NormalRegressor(config.embed_dim)
    elif config.network_type == "b_attn":
        aggregator = BasicMultiheadAttentionAggregator(
            config.embed_dim, config.num_heads, regressor_bool=False)
    elif config.network_type == "attn":
        aggregator = AttentionAggregator(
            config.embed_dim, config.num_heads, regressor_bool=False)            
    elif config.network_type == "c_attn":
        aggregator = ComplexMultiheadAttentionAggregator(
            config.embed_dim, config.num_heads)

    # Load model, aggregator, regressor
    model, aggregator, regressor = load_model(
        model, aggregator, regressor, config.test_result_dir + 'Train/iter_14243.tar')
    convert_models_to_fp32(model)
    model = weight_mode(model, trainable=False)
    model.eval()
    aggregator = weight_mode(aggregator, trainable=False)
    aggregator.to(device)
    convert_models_to_fp32(aggregator)
    aggregator.eval()
    regressor = weight_mode(regressor, trainable=False)
    regressor.to(device)
    convert_models_to_fp32(regressor)
    regressor.eval()

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
    """
    Set up and parse command-line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--dataset', type=str, default='LIVE_FB')
    parser.add_argument('--img_dir', type=str,
                        default='/scratch/sanjotst/LIVE_FB/')
    parser.add_argument('--input_csv_file', type=str,
                        default='/home/sanjotst/llm_iqa/llm-iqa/labels/live_fb_split.csv')
    parser.add_argument('--synthetic_img_dir_live_fb_2_dist_5_level_full', type=str,
                        default='/scratch/sanjotst/live_fb_2_dist_5_level_full/LIVE_FB_syn_5lvl_full_v2')
    parser.add_argument('--annotation_directory_live_fb_2_dist_5_level_full', type=str,
                        default='/home/sanjotst/llm_iqa/llm-iqa/annotation_matrices/annotated_fsim_matrices_complete_livefb')

    # network args or model arguments
    parser.add_argument('--network_type', type=str, default='b_attn')
    parser.add_argument('--embed_dim', type=int, default=4096)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--internlm_trainable_params',
                        type=list, default=['Qformer'])
    parser.add_argument('--internlm_v2_trainable_params',
                        type=list, default=['Qformer'])
    parser.add_argument('--optimizer_params', type=list, default=['Qformer'])
    parser.add_argument('--logit_processing_type',
                        type=str, default='init')  # or init

    # todo later : add actual model names (?)
    # i'm yet to decide on what basis to name these models( hf card or the terms used in papers)
    parser.add_argument('--model_description', type=str,
                        default='internlm_vl2')

    # training args
    parser.add_argument('--mode', type=str, default="train")  # or eval
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--default_device', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--test_epoch', type=float, default=0.5)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--alpha_scaling', type=float,
                        default=10)  # for scaling the 2 losses
    parser.add_argument('--test_at_first_iteration', type=str, default="false")

    # optimizer args
    parser.add_argument('--optim', type=str, default="adamw")
    parser.add_argument('--weight_decay', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--scheduler', type=bool, default=False)
    parser.add_argument('--lr_scheduler', type=str, default='cosine_annealing')
    parser.add_argument('--cawr_restart_iter', default=200, type=int,
                        help='Restart at cosine annealig at the following itertion')
    parser.add_argument('--lwca_warmup_iter', default=1000, type=int,
                        help='Warmup iterations for linear warmup cosine annealing')

    # saving and resuming args
    parser.add_argument('--results_dir', type=str,
                        default='/scratch/sanjotst/experiment_results')
    parser.add_argument('--resume_training', default=False,
                        type=bool, help='Resume training from a checkpoint')
    parser.add_argument('--resume_model_path', type=str,
                        default='/scratch/sanjotst/triplet_loss_runs/Run0033/Train/iter_11869.tar')
    parser.add_argument('--loss_function', type=str,
                        default='mse')
    config = parser.parse_args()

    return config

def main():
    config = configuration_params()

    # Load the configuration details from the json
    config_path = f"{config.test_result_dir}config_details.json"
    try:
        with open(config_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found. Exiting.")
        return  # Or handle the error as appropriate

    for key in data:
        # Only set the attribute if it doesn't already exist
        if not hasattr(config, key):
            setattr(config, key, data[key])

    test_model(config)

    # Return statement with a newline before
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

def run_program():
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()

    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)

    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
    
    return run_result

if __name__ == '__main__':
    run_result = run_program()
    print(run_result)