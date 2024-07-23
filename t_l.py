
# Common Imports
import json
import time
import datetime
import traceback
import argparse
import math
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import gc

# From our files
from util_dataload import *
from util_networks import get_internLM_model, NormalRegressor, ComplexMultiheadAttentionAggregator, BasicMultiheadAttentionAggregator
from exp2_test import exp2_2_test_function
from util_get_internlm_logits import get_init_logits, get_sentence_logits
from utils.custom_triplet_ordering_loss_4 import *

import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


# Training the model class
class DNNIter(nn.Module):
    def __init__(self, config):
        super(DNNIter, self).__init__()

        self.config = config

        self.lr_scheduler = self.config.lr_scheduler
        self.cawr_restart_iter = self.config.cawr_restart_iter
        self.lwca_warmup_iter = self.config.lwca_warmup_iter

        if self.config.dataset = 'LIVE_FB':
            self.img_dir = '/scratch/sanjotst/LIVE_FB/'
        if self.config.dataset = 'LIVE_FB_DIST1_LEVELS10':
            self.img_dir = '/scratch/sanjotst/LIVE_FB/'
            self.num_distortions = 1
            self.num_levels = 10
            self.annotation_directory = self.config.annotation_directory_synthetic_img_dir
            self.synthetic_img_dir = self.config.synthetic_img_dir_live_fb_2_dist_5_level_full
        if self.config.dataset = 'LIVE_FB_DIST4_LEVELS4'
            self.img_dir =     '/scratch/sanjotst/LIVE_FB/'   
            self.num_distortions = 4
            self.num_levels = 4 

        # Set device
        torch.cuda.set_device(self.config.default_device)
        self.device = f"cuda:{self.config.default_device}" if torch.cuda.is_available(
        ) else "cpu"

        # Self-Attention Model init
        if self.config.network_type == 'b_attn':
            self.aggregator = BasicMultiheadAttentionAggregator(
                embed_dim=self.config.embed_dim, num_heads=self.config.num_heads, regressor_bool=False).to(self.device)
        elif self.config.network_type == 'c_attn':
            self.aggregator = ComplexMultiheadAttentionAggregator(
                embed_dim=self.config.embed_dim, num_heads=self.config.num_heads, regressor_bool=False).to(self.device)
        self.aggregator = self.weight_mode(self.aggregator, trainable=True)
        self.aggregator.train()
        self.aggregator.to(self.device)
        self.convert_models_to_fp32(self.aggregator)

        # Regressor Head init
        self.regressor = NormalRegressor(
            embed_dim=self.config.embed_dim, pool=False).to(self.device)
        self.regressor = self.weight_mode(self.regressor, trainable=True)
        self.regressor.train()
        self.regressor.to(self.device)
        self.convert_models_to_fp32(self.regressor)

        # get model's tunable parameters
        # todo later : add tunable parameters for models other than internlm
        if self.config.model_description == 'internlm_vl':
            self.mllm_trainable_params = self.config.internlm_trainable_params
        if self.config.model_description == 'internlm_vl2':
            self.mllm_trainable_params = self.config.internlm_v2_trainable_params
         if self.config.model_description == 'internlm_quantised':
            self.mllm_trainable_params = self.config.internlm_quantised_trainable_params

        # load model
        # get the relevant model 
        # add code for llava and plug owl here 
        # if-else conditions need to be added 
        if self.config.model_description == 'internlm_vl':
            self.model = get_internLM_model(self.config)
        elif self.config.model_description == 'internlm_vl2':
            self.model = get_internLM_v2_model(self.config)
        elif self.config.model_description == 'internlm_quantised':
            self.model = get_internLM_quantised_model(self.config)        

        self.model = self.mllm_weight_mode(
            self.model, self.mllm_trainable_params, trainable=True)
            
        self.model.train() 

        #  don't convert quantised model to fp32
        if self.config.model_description != 'internlm-quantised':
            self.convert_models_to_fp32(self.model)

        # Setting randomness
        torch.manual_seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # Identify the trainable parameters from self.model and self.aggregrator and self.regressor for optimization
        # add if-else check for optimisable parameters based on the model 
        # add a regex
        optimizable_parameters = []
        self.mllm_optimizable_params = []
        models = [self.aggregator, self.model, self.regressor]
        for model in models:
            for name, param in model.named_parameters():
                if model == self.aggregator or model == self.regressor:
                    optimizable_parameters.append((name, param))
                elif any([param_name in name for param_name in self.config.optimizer_params]):
                    if param.requires_grad:  # Use 'param.requires_grad' for checking
                        optimizable_parameters.append((name, param))
                        self.mllm_optimizable_params.append((name, param))
                else:
                    param.requires_grad_(False)

        if self.config.optim == "adamw":
            self.optim = AdamW([param for _, param in optimizable_parameters], lr=self.config.lr,
                               weight_decay=self.config.weight_decay)  # note: only give trainable parameters to the optimizer
        elif self.config.optim == "sgd":
            self.optim = torch.optim.SGD([param for _, param in optimizable_parameters], lr=self.config.lr,
                                         weight_decay=self.config.weight_decay)

        self.test_dict = {}
        self.test_srocc = {'iteration': [], 'srocc': []}

        # Setting up results folder
        run_number = len(os.listdir(config.results_dir))
        # change the name of the curr_result_dir so that it creates folder name in a particular fashion
        # do the same for the name of the saved model too 
        # need to add loss terms here in the folder name, the dataset on which it's trained
        # 
        self.curr_result_dir = os.path.join(
            config.results_dir, config.model_description, self.config.loss_function, self.config.dataset, f'Run{run_number:04}')
        if not os.path.exists(self.curr_result_dir):
            os.mkdir(self.curr_result_dir)
        self.config.results_dir = self.curr_result_dir

        # Dumping config details to folder
        config_details_path = os.path.join(
            self.config.results_dir, 'config_details.json')
        json_object = json.dumps(self.config.__dict__, indent=4)
        with open(config_details_path, "w") as outfile:
            outfile.write(json_object)

        self.logger = SummaryWriter(
            (Path(self.curr_result_dir) / 'Logs').as_posix())
        self.save_flag = True

        # Setting up the huggingface parameters for sentence generation
        self.gen_config = dict(
            num_beams=1,
            do_sample=False,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1.0,
            max_new_tokens=15,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        self.config.gen_config = self.gen_config

        return

    # Below 2 functions for setting model to trainable or not
    @staticmethod
    def weight_mode(model, trainable=True):
        for name, param in model.named_parameters():
            if trainable:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        return model

    @staticmethod
    # todo later : check for what values of mllm_trainable parameters would this work for the quantised model
    def mllm_weight_mode(model, trainable_params, trainable=True):
        for name, param in model.named_parameters():
            if any([param_name in name for param_name in trainable_params]):
                # is there any use of "trainable" here 
                if trainable:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                param.requires_grad_(False)

        return model

    # converting model weights to float32
    # todo later : dont to this for the quantised model
    def convert_models_to_fp32(self, model):
        for p in model.parameters():
            p.data = p.data.float()
        return

    # Below 2 functions for getting the data for training and testing
    def init_dataloaders(self):

        # get dataset indices  
        train_data, test_data = get_livefb_train_data(self.config)

        # get custom datasets based on dataset
        if self.config.dataset = 'LIVE_FB':
            processed_train_data = CustomDataset(
                df_data=train_data, img_dir=self.config.img_dir)   
        else self.config.dataset = 'LIVE_FB_DIST1_LEVELS10':               
            processed_train_data = CustomTrainDatasetAnnotatedLIVEFB(
                df_data=train_data, annotation_matrix_dir=self.annotation_directory,  synthetic_img_dir=self.synthetic_img_dir)
        else self.config.dataset = 'LIVE_FB_DIST4_LEVELS4':               
            processed_train_data = CustomTrainDatasetAnnotatedLIVEFB(
                df_data=train_data, annotation_matrix_dir=self.annotation_directory,  synthetic_img_dir=self.synthetic_img_dir)

        processed_test_data = CustomDataset(
            df_data=test_data, img_dir=self.config.img_dir)

        self.pooled_train_loader = DataLoader(
            dataset=processed_train_data, batch_size=self.config.batch_size, shuffle=True)
        self.pooled_test_loader = DataLoader(
            dataset=processed_test_data, batch_size=self.config.test_batch_size, shuffle=False)

        return

    @staticmethod
    def get_next_batch(dataloader, iterator):
        try:
            next_batch = next(iterator)
        except StopIteration:
            print("Stop iteration encountered.")
            iterator = iter(dataloader)
            next_batch = next(iterator)
        return next_batch, iterator

    # Below 2 functions for getting network training hyperparameters
    def get_scheduler(self, total_iterations):
        total_iterations = total_iterations
        if self.lr_scheduler == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optim, T_max=total_iterations, eta_min=1e-7)
        if self.lr_scheduler == 'cosine_annealing_warm_restart':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=self.optim, T_0=self.cawr_restart_iter)

        if self.lr_scheduler == 'linear_warmup_cosine_annealing':
            lr_lambda = (
                lambda cur_iter: cur_iter / self.lwca_warmup_iter
                if cur_iter <= self.lwca_warmup_iter
                else 0.5 * (1 + math.cos(math.pi * (cur_iter - self.lwca_warmup_iter) / (self.num_iterations - self.lwca_warmup_iter)))
            )
            return torch.optim.lr_scheduler.LambdaLR(
                self.optim, lr_lambda=lr_lambda)
        return None

    # this function is never called anywhere 
    @staticmethod
    def update_learning_rate(optimizer, factor):
        for group in optimizer.param_groups:
            group['lr'] *= factor
        return

    # Below 2 functions for saving trained model and loading it for later
    def save_model(self, model, aggregator, regressor, optimizer, best=False):
        model_ckpt_path = Path(self.config.results_dir) / 'Train'
        if not os.path.exists(model_ckpt_path):
            os.mkdir(model_ckpt_path)

        if best:
            model_ckpt_path = os.path.join(model_ckpt_path, 'best.tar')
        else:
            model_ckpt_path = os.path.join(
                model_ckpt_path, f'iter_{self.current_iteration}.tar')

        # For internlm model, save only the trainable parameters as it's a huge model
        parameters = {name: param.data for name,
                      param in self.mllm_optimizable_params}

        save_model = {'state_dict': parameters}
        save_aggregator = {'state_dict': aggregator.state_dict()}
        save_regressor = {'state_dict': regressor.state_dict()}
        save_opt = {'state_dict': optimizer.state_dict()}
        full_dict = {'model': save_model, 'optimizer': save_opt, 'regressor': save_regressor,
                     'current_iteration': self.current_iteration, 'aggregator': save_aggregator}

        torch.save(full_dict, model_ckpt_path)
        return

    def load_model(self, path):
        print("Loading model to continue training")
        checkpoint = torch.load(path)
        self.model.load_state_dict(
            checkpoint['model']['state_dict'], strict=False)
        self.aggregator.load_state_dict(checkpoint['aggregator']['state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer']['state_dict'])
        self.regressor.load_state_dict(checkpoint['regressor']['state_dict'])
        self.current_iteration = checkpoint['current_iteration']
        return

    # Below 2 functions for  training the model
    def get_loss_fn(self, predictions, ground_truth):
        predictions = predictions.squeeze(-1)
        try:
            ground_truth = ground_truth.squeeze(-1)
        except:
            pass
        if self.config.loss == 'mse':
            return torch.nn.functional.mse_loss(predictions, ground_truth)
        elif self.config.loss == 'l1':
            return torch.nn.functional.l1_loss(predictions, ground_truth)

    def train_model(self):
        train_loss = []
        self.current_iteration = 1
        self.init_dataloaders()

        iterator_model = iter(self.pooled_train_loader)
        self.test_dict['test_srocc'] = {'srocc_value': [], 'iter_no': []}

        start_iteration = 1
        total_iterations = int(
            (self.config.epochs * len(self.pooled_train_loader)))
        test_iteration = int(
            (self.config.test_epoch * len(self.pooled_train_loader)))

        if self.config.resume_training == True:
            self.load_model(self.config.resume_model_path)
            start_iteration = self.current_iteration + 1
            self.test_dict['test_srocc']['iter_no'].append(
                self.current_iteration)
            self.test_during_train()

        self.model = self.mllm_weight_mode(
            self.model, self.mllm_trainable_params, trainable=True)
        self.model.train()
        self.aggregator = self.weight_mode(self.aggregator, trainable=True)
        self.aggregator.train()
        self.regressor = self.weight_mode(self.regressor, trainable=True)
        self.regressor.train()

        if self.config.scheduler == True:
            scheduler = self.get_scheduler(total_iterations)

        for iteration in tqdm(range(start_iteration, total_iterations + 1)):

            if iteration == 1:
                self.test_dict['test_srocc']['iter_no'].append(
                    self.current_iteration)
                self.test_during_train()
                self.model = self.mllm_weight_mode(
                    self.model, self.mllm_trainable_params, trainable=True)
                self.model.train()
                self.aggregator = self.weight_mode(
                    self.aggregator, trainable=True)
                self.aggregator.train()
                self.regressor = self.weight_mode(
                    self.regressor, trainable=True)
                self.regressor.train()

            sampled_batch, iterator_model = self.get_next_batch(
                self.pooled_train_loader, iterator_model)
            dist_img_input = sampled_batch['img'].to(
                self.device)  # bs, 17, c, h, w
            mos_target = sampled_batch['mos'].to(self.device)
            annotator_matrix = sampled_batch['annotation_matrix'].to(
                self.device)
            (b, d, c, h, w) = dist_img_input.shape
            dist_grouped = (dist_img_input.reshape(b * d, c, h, w)).to("cuda")

            # get hidden states from internlm model 
            # need to add if-else conditions to get the hidden states of other models 
            if self.config.logit_processing_type == 'init':
                hidden_states_dists = get_init_logits(self.model, dist_grouped)
            elif self.config.logit_processing_type == 'sentence':
                hidden_states_dists = get_sentence_logits(
                    self.model, dist_grouped, self.gen_config)

            # self-attention model
            attention_output_dists = self.aggregator(
                hidden_states_dists)

            # debugging condition
            [exit(print(f"NaN found in parameter {name}")) for name, param in self.model.named_parameters(
            ) if torch.isnan(param).any()]

            # Do feature level contrastive learning on the attention outputs of distorted versions of images
            attention_output_dists = attention_output_dists.reshape(b, d, -1)

            # Get MSE loss on regressed attention outputs of reference images
            # ref image is first image from every batch of dists images
            attention_output_refs = attention_output_dists[:, 0, :]
            scores = self.regressor(attention_output_refs)
            if scores.shape != mos_target.shape:
                scores.squeeze_()
                mos_target.squeeze_()
            loss_mse = self.get_loss_fn(scores, mos_target)
            total_loss = loss_mse 

            # compute each loss separately depending on the loss function 
            if self.config.loss_function == 'self_supervised_loss_1':
                loss_fn1 = LevelBlendedOrderingLoss()
                loss1 = loss_fn1(attention_output_dists, self.num_distortions, self.num_levels)
                loss_contrastive = loss1
                total_loss += self.config.alpha_scaling * loss_contrastive 
                self.logger.add_scalar('LevelBlendedOrderingLoss', loss1, self.current_iteration)
                
            if self.config.loss_function == 'self_supervised_loss_2':
                loss_fn1 = LevelBlendedOrderingLoss()
                loss_fn2 = Opt1_DistBlendedOrderingLoss()
                loss1 = loss_fn1(attention_output_dists, self.num_distortions, self.num_levels)
                loss2 = loss_fn2(attention_output_dists, self.num_distortions, self.num_levels)   
                loss_contrastive = loss1 + loss2 
                total_loss += self.config.alpha_scaling * loss_contrastive 
                self.logger.add_scalar('LevelBlendedOrderingLoss', loss1, self.current_iteration)
                self.logger.add_scalar('Opt1_DistBlendedOrderingLoss', loss2, self.current_iteration)


            # perform backprop
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if self.config.scheduler == True:
                scheduler.step()

            # Logging to tensorboard, log optimizer learning rate also
            # is gradient flow being stopped by loss.item()
            train_loss.append(loss.item())
            loss_dict = {'loss': train_loss[-1],
                         'iteration': self.current_iteration}

            self.logger.add_scalar(
                f'loss_mse', loss_mse, loss_dict['iteration'])
            self.logger.add_scalar(
                f'TrainLoss', loss_dict['loss'], loss_dict['iteration'])
            if self.config.loss_function != 'mse'
                self.logger.add_scalar(
                    f'loss_contrastive', loss_contrastive, loss_dict['iteration'])

            per_sample_loss = train_loss[-1] / self.config.batch_size
            per_sample_loss_mse = loss_mse / self.config.batch_size
            per_sample_loss_contrastive = loss_contrastive / self.config.batch_size

            # todo later : remove these print statements, per sample loss is already being logged 
            print(
                f'Iteration {iteration} done with per loss {per_sample_loss:0.4f}.')
            print(
                f'Iteration {iteration} done with per_sample_loss_mse {per_sample_loss_mse:0.4f}.')
            print(
                f'Iteration {iteration} done with per_sample_loss_contrastive {per_sample_loss_contrastive:0.4f}.')

            if iteration % test_iteration == 0 or iteration == total_iterations:
                self.test_dict['test_srocc']['iter_no'].append(
                    self.current_iteration)

                # saves the model according to test frequency
                print("Saving model before testing")
                self.save_model(self.model, self.aggregator, self.regressor,
                                self.optim, best=False)
                self.test_during_train()  # saves the model again if it's best here

                # I am setting to train in test function but just in case :>
                # why do this again ?
                self.model = self.mllm_weight_mode(
                    self.model, self.mllm_trainable_params, trainable=True)
                self.model.train()
                self.aggregator = self.weight_mode(
                    self.aggregator, trainable=True)
                self.aggregator.train()
                self.regressor = self.weight_mode(
                    self.regressor, trainable=True)
                self.regressor.train()

            self.current_iteration += 1

            del sampled_batch, dist_img_input, mos_target, hidden_states_dists, attention_output_dists, attention_output_refs, scores
            torch.cuda.empty_cache()

        return

    # The following function is to test in between
    def test_during_train(self):
        with torch.no_grad():
            self.model = self.mllm_weight_mode(
                self.model, self.mllm_trainable_params, trainable=False)
            self.model.eval()
            self.aggregator = self.weight_mode(
                self.aggregator, trainable=False)
            self.aggregator.eval()
            self.regressor = self.weight_mode(self.regressor, trainable=False)
            self.regressor.eval()

            self.test_dict['csv'] = {'Video_Name': [], 'MOS': [
            ], f'pred{self.current_iteration:04d}': []}

            test_prediction, corresponding_mos, corresponding_name = exp2_2_test_function(
                self.config, self.pooled_test_loader, self.model, self.aggregator, self.regressor, self.device)

            self.test_dict['csv'][f'pred{self.current_iteration:04}'] = test_prediction
            self.test_dict['csv']['Video_Name'] = corresponding_name
            self.test_dict['csv']['MOS'] = corresponding_mos

            srcc_test_correlation = spearmanr(
                np.array(test_prediction), np.array(corresponding_mos))[0]
            plcc_test_correlation = pearsonr(
                np.array(test_prediction), np.array(corresponding_mos))[0]

            self.test_dict['csv'][f'pred{self.current_iteration:04}'].append(
                srcc_test_correlation)
            self.test_dict['csv']['Video_Name'].append('SROCC')
            self.test_dict['csv']['MOS'].append(-1.0)

            del test_prediction, corresponding_mos, corresponding_name
            gc.collect()

            details_path = os.path.join(self.config.results_dir, 'details.txt')
            logging.basicConfig(filename=details_path,
                                filemode='a', level=logging.DEBUG, format='')

            print(
                f"SRCC for {self.current_iteration:04}, {srcc_test_correlation}")
            print(
                f"PLCC for {self.current_iteration:04}, {plcc_test_correlation}")
            logging.info(
                f"SRCC for {self.current_iteration:04}, {srcc_test_correlation}")
            logging.info(
                f"PLCC for {self.current_iteration:04}, {plcc_test_correlation}")
            self.logger.add_scalar(
                f'test_srocc', srcc_test_correlation, self.current_iteration)
            self.logger.add_scalar(
                f'test_plcc', plcc_test_correlation, self.current_iteration)
            # Saving test performance to disk
            if not os.path.exists((Path(self.config.results_dir) / 'Test').as_posix()):
                os.mkdir((Path(self.config.results_dir) / 'Test').as_posix())

            save_dir = (Path(self.config.results_dir) /
                        f'Test/predictions.csv').as_posix()

            if self.save_flag:
                df = pd.DataFrame.from_dict(self.test_dict['csv'])
                df.to_csv(save_dir, index=False)
            else:
                df1 = pd.read_csv(save_dir)
                df1[f'pred{self.current_iteration:04}'] = self.test_dict['csv'][f'pred{self.current_iteration:04}']
                df1.to_csv(save_dir, index=False)

            # So test_dict[test_srocc] looks like {[srocc1 srocc2] [iter 1 2]}
            self.test_dict['test_srocc']['srocc_value'].append(
                srcc_test_correlation)

            # Saving the test performance vs cycles
            pyplot.figure(1)
            pyplot.plot(self.test_dict['test_srocc']['iter_no'],
                        self.test_dict['test_srocc']['srocc_value'])
            pyplot.grid()
            pyplot.xlabel('Training Iteration')
            pyplot.ylabel('SROCC')
            pyplot.savefig(Path(self.config.results_dir) / f'Test/test.png')

            self.save_flag = False

            # Setting all network parameters to train() mode so that we can identify which trainable parameters to save
            self.model = self.mllm_weight_mode(
                self.model, self.mllm_trainable_params, trainable=True)
            self.model.train()
            self.aggregator = self.weight_mode(self.aggregator, trainable=True)
            self.aggregator.train()
            self.regressor = self.weight_mode(self.regressor, trainable=True)
            self.regressor.train()

            # Saving the model if it's the best model
            if len(self.test_srocc['srocc']) != 0:
                if srcc_test_correlation > max(self.test_srocc['srocc']):
                    self.save_model(self.model, self.aggregator,
                                    self.regressor, self.optim, best=True)

            self.test_srocc['srocc'].append(srcc_test_correlation)
            self.test_srocc['iteration'].append(self.current_iteration)

        return

# todo later : change this function's name to experiment configuration 
def configuration_params():
    parser = argparse.ArgumentParser()

    # data args
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
    parser.add_argument('--optimizer_params', type=list, default=['Qformer'])
    parser.add_argument('--logit_processing_type',
                        type=str, default='init')  # or init

    # todo later : add actual model names (?)
    # i'm yet to decide on what basis to name these models( hf card or the terms used in papers)                     
    parser.add_argument('--model_description', type=str,
                        default='internlm_vl')
    # training args
    parser.add_argument('--mode', type=str, default="train")  # or eval
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--default_device', type=int, default=3)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--test_epoch', type=float, default=0.5)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--alpha_scaling', type=float,
                        default=10)  # for scaling the 2 losses

    # optimizer args
    parser.add_argument('--optim', type=str, default="adamw")
    parser.add_argument('--weight_decay', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
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
    model = DNNIter(config)
    model.train_model()

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
