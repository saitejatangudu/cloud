import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm
from util_networks import NormalRegressor, AttentionAggregator
from util_dataload import get_qinstruct_train_test_loader, CustomDatasetQinst
from scipy.stats import spearmanr
import datetime
import json


def train(config, results_dir, experiment_name):
    print(f"Results will be saved in: {results_dir}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(results_dir, experiment_name))

    # Load train and test data
    train_data, test_data = get_qinstruct_train_test_loader(config)

    # Initialize model, loss, and optimizer
    model = AttentionAggregator(embed_dim=config.embed_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Move model to device
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    # Train loop
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} (Training)"):
            inputs, targets, _ = batch['info_tensor'].to(
                device), batch['mos'].to(device), batch['name']
            optimizer.zero_grad()
            # print(inputs.shape)  # torch.Size([32, 77, 4096])
            # print(targets.shape)  # torch.Size([32, 1])
            outputs = model(inputs)
            final_output = outputs['output']
            # print(final_output.shape)  # torch.Size([32, 1, 1])
            # Get the shape of the final output tensor
            loss = criterion(final_output.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(
            f"Epoch {epoch+1}/{config.num_epochs} (Training), Loss: {epoch_loss}")

        # Write training loss to TensorBoard
        writer.add_scalar('Train/Loss', epoch_loss, epoch)

        # Evaluate on test set
        model.eval()
        ground_truth = []
        predicted_scores = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} (Testing)"):
                inputs, targets, _ = batch['info_tensor'].to(
                    device), batch['mos'].to(device), batch['name']
                outputs = model(inputs)

                # Collect ground truth and predicted scores
                ground_truth.extend(targets.cpu().numpy())
                predicted_scores.extend(
                    outputs['output'].squeeze().cpu().numpy())

        # Compute SROCC
        srocc = spearmanr(ground_truth, predicted_scores).correlation
        print(f"Spearman Rank-Order Correlation Coefficient (SROCC): {srocc}")
        writer.add_scalar('Test/SROCC', srocc, epoch)

    # Save model parameters
    torch.save(model.state_dict(), os.path.join(
        results_dir, experiment_name, 'model.pth'))

    # Close TensorBoard writer
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train MOS prediction model')
    parser.add_argument('--h5_dir', type=str,
                        help='Directory containing H5 feature files', default='/home/sanjotst/llm_iqa/llm-iqa/qinstruct_features/livefb')
    parser.add_argument('--input_json_file', type=str,
                        help='JSON file containing image paths and MOS scores', default='/home/sanjotst/llm_iqa/llm-iqa/labels/flive.json')
    parser.add_argument('--embed_dim', type=int, default=4096,
                        help='Dimension of the feature embeddings')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--train_test_split', type=float,
                        default=0.8, help='Train test split ratio')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode: train or eval')

    config = parser.parse_args()
    # Convert configuration to a dictionary for easier manipulation
    config_dict = vars(config)
    results_dir = "exp3_results_folder"

    # Generate a unique identifier for this run, e.g., based on the current timestamp
    unique_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"exp_{unique_id}_supervised_qinst_with_aggregator"

    # Alternatively, if you're incrementing an experiment number, you'd manage that here

    # Create results directory with unique experiment name
    config_filename = f"config_{unique_id}.txt"

    # Save configuration to a file within the existing directory
    config_path = os.path.join(results_dir, config_filename)

    with open(config_path, 'w') as f:
        # Writing parameters as a JSON string for readability
        json.dump(config_dict, f, indent=4)
    # Train the model
    train(config, results_dir, experiment_name)


if __name__ == "__main__":
    main()
