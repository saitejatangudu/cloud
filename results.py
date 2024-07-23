def get_init_logits_llava(model, images, tokenizer, image_processor):
    # report the int() usage because period is before image
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    text = "User: Rate the quality of the image <image> ." 
    batch_size = images.size(0)
    print(f"printing batch size : {batch_size}")
    texts = [text] * batch_size
    print("printing image shapes")
    print(images.shape) # 1 * 3 * 336 * 336
    text = text + DEFAULT_IMAGE_TOKEN + " . "

    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    print(images.shape)
    print(IMAGE_TOKEN_INDEX)
    input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    print(f" length of input_ids is : {len(input_ids)}")
    print(f"  input_ids[0] shape : {input_ids[0].shape}")
    print(f" values of input_ids : {input_ids}")
    outputs = model(input_ids=input_ids, images=images, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    print("printing last hidden state shape")
    print(last_hidden_state.shape)
    exit()
    return last_hidden_state

# Program started at 29/06/2024 04:53:00 PM
# total iterations : 37820
# printing batch size : 8
# printing image shapes
# torch.Size([8, 3, 336, 336])
# torch.Size([8, 3, 336, 336])
# -200
#  prompt chunks : [[1, 4911, 29901, 390, 403, 278, 11029, 310, 278, 1967, 29871], [1, 29871, 869], [1, 29871, 869, 29871]]
#  length of input_ids is : 1
#   input_ids[0] shape : torch.Size([18])
#  values of input_ids : tensor([[    1,  4911, 29901,   390,   403,   278, 11029,   310,   278,  1967,
#          29871,  -200, 29871,   869,  -200, 29871,   869, 29871]],
#        device='cuda:0')
#  printing shape of image features : 8 
# torch.Size([576, 4096])
# the number of images is : 2
#  out of bound index : 0
#  out of bound index : 1
# printing last hidden state shape
# torch.Size([1, 1168, 4096])

#############################################################################################################

# adding 5th dimension to images

def get_init_logits_llava(model, images, tokenizer, image_processor):
    # report the int() usage because period is before image
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    text = "User: Rate the quality of the image <image> ." 
    batch_size = images.size(0)
    print(f"printing batch size : {batch_size}")
    texts = [text] * batch_size
    print("printing image shapes")
    print(images.shape) # 1 * 3 * 336 * 336
    text = text + DEFAULT_IMAGE_TOKEN + " . "

    if len(images.shape) == 4:
        images = images.unsqueeze(0)
    print(images.shape)
    print(IMAGE_TOKEN_INDEX)
    input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    print(f" length of input_ids is : {len(input_ids)}")
    print(f"  input_ids[0] shape : {input_ids[0].shape}")
    print(f" values of input_ids : {input_ids}")
    outputs = model(input_ids=input_ids, images=images, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    print("printing last hidden state shape")
    print(last_hidden_state.shape)
    exit()
    return last_hidden_state


# Traceback (most recent call last):
#   File "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/training_loop.py", line 797, in <module>
#     main()
#   File "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/training_loop.py", line 787, in main
#     model.train_model()
#   File "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/training_loop.py", line 447, in train_model
#     hidden_states_dists = get_init_logits_llava(
#   File "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/llm_feature_extraction.py", line 176, in get_init_logits_llava
#     outputs = model(input_ids=input_ids, images=images, output_hidden_states=True, return_dict=True)
#   File "/home/sanjotst/anaconda3/envs/llava/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/sanjotst/anaconda3/envs/llava/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/sanjotst/anaconda3/envs/llava/lib/python3.10/site-packages/accelerate/hooks.py", line 166, in new_forward
#     output = module._old_forward(*args, **kwargs)
#   File "/home/sanjotst/LLaVA/llava/model/language_model/llava_llama.py", line 81, in forward
#     ) = self.prepare_inputs_labels_for_multimodal(
#   File "/home/sanjotst/LLaVA/llava/model/llava_arch.py", line 263, in prepare_inputs_labels_for_multimodal
#     cur_image_features = image_features[cur_image_idx]

###########################################################################################################

# removing     text = text + DEFAULT_IMAGE_TOKEN + " . " line and checking

def get_init_logits_llava(model, images, tokenizer, image_processor):
    # report the int() usage because period is before image
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    text = "User: Rate the quality of the image <image> ." 
    batch_size = images.size(0)
    print(f"printing batch size : {batch_size}")
    texts = [text] * batch_size
    print("printing image shapes")
    print(images.shape) # 1 * 3 * 336 * 336

    if len(images.shape) == 4:
        images = images.unsqueeze(0)
    print(images.shape)
    print(IMAGE_TOKEN_INDEX)
    input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    print(f" length of input_ids is : {len(input_ids)}")
    print(f"  input_ids[0] shape : {input_ids[0].shape}")
    print(f" values of input_ids : {input_ids}")
    outputs = model(input_ids=input_ids, images=images, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    print("printing last hidden state shape")
    print(last_hidden_state.shape)
    exit()
    return last_hidden_state

# Program started at 29/06/2024 05:54:35 PM
# total iterations : 37820
# printing batch size : 8
# printing image shapes
# torch.Size([8, 3, 336, 336])
# torch.Size([1, 8, 3, 336, 336])
# -200
#  prompt chunks : [[1, 4911, 29901, 390, 403, 278, 11029, 310, 278, 1967, 29871], [1, 29871, 869]]
#  length of input_ids is : 1
#   input_ids[0] shape : torch.Size([14])
#  values of input_ids : tensor([[    1,  4911, 29901,   390,   403,   278, 11029,   310,   278,  1967,
#          29871,  -200, 29871,   869]], device='cuda:0')
#  printing shape of image features : 1 
# torch.Size([4608, 4096])
# the number of images is : 1
#  out of bound index : 0
# printing last hidden state shape
# torch.Size([1, 4621, 4096])

# changing batch size to 16

# Program started at 29/06/2024 05:56:44 PM
# total iterations : 18910
# printing batch size : 16
# printing image shapes
# torch.Size([16, 3, 336, 336])
# torch.Size([1, 16, 3, 336, 336])
# -200
#  prompt chunks : [[1, 4911, 29901, 390, 403, 278, 11029, 310, 278, 1967, 29871], [1, 29871, 869]]
#  length of input_ids is : 1
#   input_ids[0] shape : torch.Size([14])
#  values of input_ids : tensor([[    1,  4911, 29901,   390,   403,   278, 11029,   310,   278,  1967,
#          29871,  -200, 29871,   869]], device='cuda:0')
#  printing shape of image features : 1 
# torch.Size([9216, 4096])
# the number of images is : 1
#  out of bound index : 0
# printing last hidden state shape
# torch.Size([1, 9229, 4096])

#################################################
# IMPORTANT
#################################################
# important info 2nd dimension can be resized 
#################################################
# calculations 
# 9216 / 16 = 576
# 4608 / 8 = 576
################################################################################################

# not adding an extra dimension in the above setup

def get_init_logits_llava(model, images, tokenizer, image_processor):
    # report the int() usage because period is before image
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    text = "User: Rate the quality of the image <image> ." 
    batch_size = images.size(0)
    print(f"printing batch size : {batch_size}")
    texts = [text] * batch_size
    print("printing image shapes")
    print(images.shape) # 1 * 3 * 336 * 336

    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    print(images.shape)
    print(IMAGE_TOKEN_INDEX)
    input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    print(f" length of input_ids is : {len(input_ids)}")
    print(f"  input_ids[0] shape : {input_ids[0].shape}")
    print(f" values of input_ids : {input_ids}")
    outputs = model(input_ids=input_ids, images=images, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    print("printing last hidden state shape")
    print(last_hidden_state.shape)
    exit()
    return last_hidden_state

# Program started at 29/06/2024 06:00:45 PM
# total iterations : 18910
# printing batch size : 16
# printing image shapes
# torch.Size([16, 3, 336, 336])
# torch.Size([16, 3, 336, 336])
# -200
#  prompt chunks : [[1, 4911, 29901, 390, 403, 278, 11029, 310, 278, 1967, 29871], [1, 29871, 869]]
#  length of input_ids is : 1
#   input_ids[0] shape : torch.Size([14])
#  values of input_ids : tensor([[    1,  4911, 29901,   390,   403,   278, 11029,   310,   278,  1967,
#          29871,  -200, 29871,   869]], device='cuda:0')
#  printing shape of image features : 16 
# torch.Size([576, 4096])
# the number of images is : 1
#  out of bound index : 0
# printing last hidden state shape
# torch.Size([1, 589, 4096])

#################################################################################################

# removing <image> placeholder 

def get_init_logits_llava(model, images, tokenizer, image_processor):
    # report the int() usage because period is before image
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    text = "User: Rate the quality of the image  ." 
    batch_size = images.size(0)
    print(f"printing batch size : {batch_size}")
    texts = [text] * batch_size
    print("printing image shapes")
    print(images.shape) # 1 * 3 * 336 * 336

    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    print(images.shape)
    print(IMAGE_TOKEN_INDEX)
    input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    print(f" length of input_ids is : {len(input_ids)}")
    print(f"  input_ids[0] shape : {input_ids[0].shape}")
    print(f" values of input_ids : {input_ids}")
    outputs = model(input_ids=input_ids, images=images, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    print("printing last hidden state shape")
    print(last_hidden_state.shape)
    exit()
    return last_hidden_state

# Program started at 29/06/2024 06:07:12 PM
# total iterations : 18910
# printing batch size : 16
# printing image shapes
# torch.Size([16, 3, 336, 336])
# torch.Size([16, 3, 336, 336])
# -200
#  prompt chunks : [[1, 4911, 29901, 390, 403, 278, 11029, 310, 278, 1967, 29871, 869]]
#  length of input_ids is : 1
#   input_ids[0] shape : torch.Size([12])
#  values of input_ids : tensor([[    1,  4911, 29901,   390,   403,   278, 11029,   310,   278,  1967,
#          29871,   869]], device='cuda:0')
#  printing shape of image features : 16 
# torch.Size([576, 4096])
# the number of images is : 0
# Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:6! (when checking argument for argument tensors in method wrapper_CUDA_cat)
# Program ended at 29/06/2024 06:07:22 PM
# Execution time: 0:00:10.109958


#   File "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/training_loop.py", line 797, in <module>
#     main()
#   File "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/training_loop.py", line 787, in main
#     model.train_model()
#   File "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/training_loop.py", line 447, in train_model
#     hidden_states_dists = get_init_logits_llava(
#   File "/home/sanjotst/llm_iqa/llm-iqa/code/baselines/llm_feature_extraction.py", line 175, in get_init_logits_llava
#     outputs = model(input_ids=input_ids, images=images, output_hidden_states=True, return_dict=True)
#   File "/home/sanjotst/anaconda3/envs/llava/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/sanjotst/anaconda3/envs/llava/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/sanjotst/anaconda3/envs/llava/lib/python3.10/site-packages/accelerate/hooks.py", line 166, in new_forward
#     output = module._old_forward(*args, **kwargs)
#   File "/home/sanjotst/LLaVA/llava/model/language_model/llava_llama.py", line 81, in forward
#     ) = self.prepare_inputs_labels_for_multimodal(
#   File "/home/sanjotst/LLaVA/llava/model/llava_arch.py", line 239, in prepare_inputs_labels_for_multimodal
#     cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:6! (when checking argument for argument tensors in method wrapper_CUDA_cat)

#############################################################################################################################

# adding     text = text + DEFAULT_IMAGE_TOKEN + " . "
def get_init_logits_llava(model, images, tokenizer, image_processor):
    # report the int() usage because period is before image
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    text = "User: Rate the quality of the image  ." 
    batch_size = images.size(0)
    print(f"printing batch size : {batch_size}")
    texts = [text] * batch_size
    print("printing image shapes")
    print(images.shape) # 1 * 3 * 336 * 336
    
    text = text + DEFAULT_IMAGE_TOKEN + " . "

    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    print(images.shape)
    print(IMAGE_TOKEN_INDEX)
    input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    print(f" length of input_ids is : {len(input_ids)}")
    print(f"  input_ids[0] shape : {input_ids[0].shape}")
    print(f" values of input_ids : {input_ids}")
    outputs = model(input_ids=input_ids, images=images, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    print("printing last hidden state shape")
    print(last_hidden_state.shape)
    exit()
    return last_hidden_state


# Program started at 29/06/2024 06:12:51 PM
# total iterations : 18910
# printing batch size : 16
# printing image shapes
# torch.Size([16, 3, 336, 336])
# torch.Size([16, 3, 336, 336])
# -200
#  prompt chunks : [[1, 4911, 29901, 390, 403, 278, 11029, 310, 278, 1967, 29871, 869], [1, 29871, 869, 29871]]
#  length of input_ids is : 1
#   input_ids[0] shape : torch.Size([16])
#  values of input_ids : tensor([[    1,  4911, 29901,   390,   403,   278, 11029,   310,   278,  1967,
#          29871,   869,  -200, 29871,   869, 29871]], device='cuda:0')
#  printing shape of image features : 16 
# torch.Size([576, 4096])
# the number of images is : 1
#  out of bound index : 0
# printing last hidden state shape
# torch.Size([1, 591, 4096])
