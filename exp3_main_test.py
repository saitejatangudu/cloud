import os
import argparse
import torch
from torch.utils.data import DataLoader
from util_networks import NormalRegressor
from util_dataload import CustomDatasetQinst, get_qinstruct_train_test_loader
import pandas as pd
from scipy.stats import spearmanr


def predict(config):
    # Load model
    model = NormalRegressor(embed_dim=config.embed_dim)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    # Load dataset
    test_data = get_qinstruct_train_test_loader(config)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Predict MOS scores
    predicted_scores = []
    ground_truth = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['info_tensor'].to(device)
            targets = batch['mos'].cpu().numpy()
            outputs = model(inputs).squeeze().cpu().numpy()
            predicted_scores.extend(outputs)
            ground_truth.extend(targets)

    # Compute SROCC
    srocc = spearmanr(ground_truth, predicted_scores).correlation
    print(f"Spearman Rank-Order Correlation Coefficient (SROCC): {srocc}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict MOS using trained model')
    parser.add_argument('--h5_dir', type=str,
                        help='Directory containing H5 feature files', default='/home/sanjotst/llm_iqa/llm-iqa/qinstruct_features/SPAQ_512')
    parser.add_argument('--input_json_file', type=str, help='JSON file containing image paths and MOS scores',
                        default='/home/sanjotst/llm_iqa/llm-iqa/labels/spaq.json')
    parser.add_argument('--model_path', type=str,
                        help='Path to saved model parameters', default='/home/sanjotst/llm_iqa/llm-iqa/code/baselines/exp3_results_folder/exp_20240408_235051_supervised_qinst_no_aggregator/model.pth')
    parser.add_argument('--embed_dim', type=int, default=4096,
                        help='Dimension of the feature embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--mode', type=str, default='eval',
                        help='Mode: train or eval')

    config = parser.parse_args()

    # Perform prediction and compute SROCC
    predict(config)


if __name__ == "__main__":
    main()
