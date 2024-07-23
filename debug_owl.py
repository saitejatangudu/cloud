import torch

checkpoint = torch.load(
    '/scratch/sanjotst/experiment_results/mplug_owl_mse_LIVE_FB_Run0497/Train/iter_15128.tar')
print(checkpoint['optimizer']['state_dict'])

state_dict = checkpoint['optimizer']['state_dict']
model_params = {name: param for name, param in checkpoint.named_parameters()}
filtered_state_dict = {k: v for k,
                       v in state_dict.items() if k in model_params}
optimizer.load_state_dict(filtered_state_dict)
