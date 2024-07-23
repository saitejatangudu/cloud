
from transformers import AutoModel, AutoTokenizer
import torch
from utils.utils import auto_configure_device_map
from accelerate import dispatch_model

# # Load the model and tokenizer
checkpoint = 'internlm/internlm-xcomposer-vl-7b'
model = AutoModel.from_pretrained(
    checkpoint, trust_remote_code=True, torch_dtype=torch.int8).cuda().eval()
device_map = auto_configure_device_map(
    1, 7)
print("Configured device_map.")
model = dispatch_model(model, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, trust_remote_code=True)
model.tokenizer = tokenizer
text = [" this is a test run"]
samples = { 'text_input' : text } 
# model(samples=samples)
model(samples)
