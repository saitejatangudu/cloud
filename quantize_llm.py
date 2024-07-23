import torch
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy
import argparse
import os
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

# File imports
from util_networks import *
from util_dataload import *
from llm_feature_extraction import *


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
    return


def load_model(model, aggregator, regressor, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model']['state_dict'], strict=False)
    aggregator.load_state_dict(checkpoint['aggregator']['state_dict'])
    regressor.load_state_dict(checkpoint['regressor']['state_dict'])
    return model, aggregator, regressor


def weight_mode(model, trainable=False):
    for param in model.parameters():
        param.requires_grad_(trainable)
    return model


def init_test_dataloaders(config):
    test_data = get_test_data(config)
    processed_test_data = CustomDataset(
        config.img_dir, test_data, config.model_description)
    pooled_test_loader = DataLoader(
        processed_test_data, batch_size=16, shuffle=False)
    return pooled_test_loader


def configuration_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--embed_dim', type=int, default=4096)
    parser.add_argument('--network_type', type=str, default='b_attn')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--default_device', type=int, default=2)
    parser.add_argument('--model_description', type=str,
                        default='internlm_vl')
    parser.add_argument('--quantization_type', type=str,
                        default='dynamic')
    parser.add_argument('--num_gpus', type=int,
                        default=1)
    config = parser.parse_args()
    return config


def main():
    config = configuration_params()
    print('The Configuration for current run:\n', config)

    # Set device to CPU
    device = torch.device("cpu")

    tokenizer = None
    image_processor = None

    # Get model
    if config.model_description == 'internlm_vl':
        model = get_internLM_model(config).to(device)
    if config.model_description == 'internlm_vl2_quantised':
        model = get_internLM_quantised_model(config).to(device)
    if config.model_description == 'internlm_vl2':
        model = get_internLM_v2_model(config).to(device)
    if config.model_description == 'llava':
        model, tokenizer, image_processor = get_llava_model(config)
        model = model.to(device)
    if config.model_description == 'mplug_owl':
        model, tokenizer, image_processor = get_mplug_owl_model(config)
        model = model.to(device)

    regressor = NormalRegressor1(config.embed_dim).to(device)

    if config.network_type == "b_attn":
        aggregator = BasicMultiheadAttentionAggregator(
            config.embed_dim, config.num_heads, regressor_bool=False).to(device)
    elif config.network_type == "attn":
        aggregator = AttentionAggregator(
            config.embed_dim, config.num_heads, regressor_bool=False).to(device)
    elif config.network_type == "c_attn":
        aggregator = ComplexMultiheadAttentionAggregator(
            config.embed_dim, config.num_heads).to(device)

    # Load model, aggregator, regressor
    model_path = config.model_path
    print(model_path)
    model, aggregator, regressor = load_model(
        model, aggregator, regressor, model_path)
    convert_models_to_fp32(model)
    model = weight_mode(model, trainable=False)
    model.eval()
    aggregator = weight_mode(aggregator, trainable=False)
    convert_models_to_fp32(aggregator)
    aggregator.eval()
    regressor = weight_mode(regressor, trainable=False)
    convert_models_to_fp32(regressor)
    regressor.eval()

    # Create example inputs for quantization

    example_inputs_aggregator = torch.randn(8, 78, 4096)
    example_inputs_regressor = torch.randn(8, 4096)
    text = ['Please introduce the person in this picture in detail.']
    samples = { 'text_input' : text } 
    # image = 'examples/images/aiyinsitan.jpg'
    if config.quantization_type == 'dynamic':
        # Perform dynamic quantization
        qconfig_mapping = QConfigMapping().set_global(
            torch.ao.quantization.default_dynamic_qconfig)

        # Prepare and convert the model
        model_to_quantize = copy.deepcopy(model).to(device)
        model_prepared = quantize_fx.prepare_fx(
            model_to_quantize, qconfig_mapping, samples)
        model_quantized_dynamic = quantize_fx.convert_fx(model_prepared)

        # Prepare and convert the aggregator
        aggregator_to_quantize = copy.deepcopy(aggregator).to(device)
        aggregator_prepared = quantize_fx.prepare_fx(
            aggregator_to_quantize, qconfig_mapping, example_inputs_aggregator)
        aggregator_quantized_dynamic = quantize_fx.convert_fx(
            aggregator_prepared)

        # Prepare and convert the regressor
        regressor_to_quantize = copy.deepcopy(regressor).to(device)
        regressor_prepared = quantize_fx.prepare_fx(
            regressor_to_quantize, qconfig_mapping, example_inputs_regressor)
        regressor_quantized_dynamic = quantize_fx.convert_fx(
            regressor_prepared)

    elif config.quantization_type == 'static':
        # Perform static quantization
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")

        # Prepare and convert the model
        model_to_quantize = copy.deepcopy(model).to(device)
        model_prepared = quantize_fx.prepare_fx(
            model_to_quantize, qconfig_mapping, example_inputs_model)

        # Calibrate the model
        with torch.no_grad():
            for _ in range(20):
                model_prepared(example_inputs_model)

        model_quantized_static = quantize_fx.convert_fx(model_prepared)

        # Prepare and convert the aggregator
        aggregator_to_quantize = copy.deepcopy(aggregator).to(device)
        aggregator_prepared = quantize_fx.prepare_fx(
            aggregator_to_quantize, qconfig_mapping, example_inputs_aggregator)

        # Calibrate the aggregator
        with torch.no_grad():
            for _ in range(20):
                aggregator_prepared(example_inputs_aggregator)

        aggregator_quantized_static = quantize_fx.convert_fx(
            aggregator_prepared)

        # Prepare and convert the regressor
        regressor_to_quantize = copy.deepcopy(regressor).to(device)
        regressor_prepared = quantize_fx.prepare_fx(
            regressor_to_quantize, qconfig_mapping, example_inputs_regressor)

        # Calibrate the regressor
        with torch.no_grad():
            for _ in range(20):
                regressor_prepared(example_inputs_regressor)

        regressor_quantized_static = quantize_fx.convert_fx(regressor_prepared)

    # Create a directory for the quantized models
    quantized_dir = os.path.join(os.path.dirname(
        model_path), f"{os.path.basename(model_path)}_quantized_{config.quantization_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(quantized_dir, exist_ok=True)

    # Save quantized models
    quantized_model = {
        'model': model_quantized_dynamic.state_dict() if config.quantization_type == 'dynamic' else model_quantized_static.state_dict(),
        'aggregator': aggregator_quantized_dynamic.state_dict() if config.quantization_type == 'dynamic' else aggregator_quantized_static.state_dict(),
        'regressor': regressor_quantized_dynamic.state_dict() if config.quantization_type == 'dynamic' else regressor_quantized_static.state_dict()
    }

    torch.save(quantized_model, os.path.join(
        quantized_dir, "quantized_model.pth"))

    print(f"Quantized models saved in {quantized_dir}")


if __name__ == "__main__":
    main()

# output
# (llm8) [sanjotst@compute baselines]$ python quantize_llm.py --model_path /scratch/sanjotst/old_script_results/Run0012/Train/iter_5672.tar
# The Configuration for current run:
#  Namespace(model_path='/scratch/sanjotst/old_script_results/Run0012/Train/iter_5672.tar', embed_dim=4096, network_type='b_attn', num_heads=8, default_device=2, model_description='internlm_vl', quantization_type='dynamic', num_gpus=1)
# Init VIT ... Done
# Init Perceive Sampler ... Done
# Init InternLM ... Done
# Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:09<00:00,  2.45s/it]
# Configured device_map.
# 2
# /scratch/sanjotst/old_script_results/Run0012/Train/iter_5672.tar
#  the samples are
# Proxy(get)
# =========================
# Traceback (most recent call last):
#   File "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/quantize_llm.py", line 208, in <module>
#     main()
#   File "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/quantize_llm.py", line 130, in main
#     model_prepared = quantize_fx.prepare_fx(
#   File "/home/sanjotst/anaconda3/envs/llm8/lib/python3.9/site-packages/torch/ao/quantization/quantize_fx.py", line 380, in prepare_fx
#     return _prepare_fx(
#   File "/home/sanjotst/anaconda3/envs/llm8/lib/python3.9/site-packages/torch/ao/quantization/quantize_fx.py", line 133, in _prepare_fx
#     graph_module = GraphModule(model, tracer.trace(model))
#   File "/home/sanjotst/anaconda3/envs/llm8/lib/python3.9/site-packages/torch/fx/_symbolic_trace.py", line 778, in trace
#     (self.create_arg(fn(*args)),),
#   File "/home/sanjotst/.cache/huggingface/modules/transformers_modules/internlm/internlm-xcomposer-vl-7b/8a8a3ae062068c45a0c25875146237cc8b5e20e1/modeling_InternLM_XComposer.py", line 383, in forward
#     has_img = 'images' in samples.keys()
#   File "/home/sanjotst/anaconda3/envs/llm8/lib/python3.9/site-packages/torch/fx/proxy.py", line 385, in __iter__
#     return self.tracer.iter(self)
#   File "/home/sanjotst/anaconda3/envs/llm8/lib/python3.9/site-packages/torch/fx/proxy.py", line 285, in iter
#     raise TraceError('Proxy object cannot be iterated. This can be '
# torch.fx.proxy.TraceError: Proxy object cannot be iterated. This can be attempted when the Proxy is used in a loop or as a *args or **kwargs function argument. See the torch.fx docs on pytorch.org for a more detailed explanation of what types of control flow can be traced, and check out the Proxy docstring for help troubleshooting Proxy iteration errors
