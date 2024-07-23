import os
import h5py
import torch
import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, img_dir, df_data, transform):
        self.img_dir = img_dir
        self.df_data = df_data
        self.transform = transform

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        img_path = self.df_data.iloc[idx]['img_path']  # Updated line
        x = Image.open(join(self.img_dir, img_path))
        if x.mode != 'RGB':
            x = x.convert('RGB')
        x = self.transform(x)

        return {
            "img": x,
            "mos": torch.tensor(self.df_data.iloc[idx]['gt_score'], dtype=torch.float32),
            "name": img_path  # Updated line
        }


class DNNIter:
    def __init__(self):
        self.exp_name = 'llm_feature_extraction'
        self.random_seed = 0
        self.default_device = 1
        self.batch_size = 1  # Changed batch size to 1
        torch.cuda.set_device(self.default_device)
        self.device = f"cuda:{self.default_device}" if torch.cuda.is_available(
        ) else "cpu"
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def data_loader(self):
        img_dir = '/home/sanjotst/llm_iqa/internlm-sst/mnt_sanjot/LIVE_FB'
        input_json = '/home/sanjotst/llm_iqa/sanjot_json/flive.json'
        df_data = pd.read_json(input_json).astype({'gt_score': np.float32})
        train_transform = transforms.Compose([
            transforms.Resize(
                (224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        train_data = CustomDataset(img_dir, df_data, train_transform)
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=False)
        return train_loader

    def get_model(self):
        checkpoint = 'DLight1551/internlm-xcomposer-vl-7b-qinstruct-full'
        model = AutoModel.from_pretrained(
            checkpoint, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True)
        model.tokenizer = tokenizer
        return model

    def llm_outputs(self, model, image):
        text = "User: Describe the quality of the image <ImageHere> ."

        with torch.cuda.amp.autocast():
            img_embeds = model.encode_img(image)
        prompt_segs = text.split('<ImageHere>')
        prompt_seg_tokens = [
            model.tokenizer(seg, return_tensors='pt', add_special_tokens=i == 0).
            to(model.internlm_model.model.embed_tokens.weight.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]

        prompt_seg_embs = [
            model.internlm_model.model.embed_tokens(
                seg).expand(img_embeds.shape[0], -1, -1)
            for seg in prompt_seg_tokens
        ]
        prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)
        outputs = model.internlm_model(
            inputs_embeds=prompt_embs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        return hidden_states.squeeze(0)

    def feature_extraction(self, train_loader, model):
        os.makedirs(
            '/home/sanjotst/llm_iqa/internlm-sst/features/livefb', exist_ok=True)
        os.makedirs(
            '/home/sanjotst/llm_iqa/internlm-sst/features/livefb/patches/voc_emotic_ava', exist_ok=True)
        os.makedirs(
            '/home/sanjotst/llm_iqa/internlm-sst/features/livefb/EE371R', exist_ok=True)
        os.makedirs(
            '/home/sanjotst/llm_iqa/internlm-sst/features/livefb/blur_dataset', exist_ok=True)
        output_path = "/home/sanjotst/llm_iqa/internlm-sst/features/livefb"
        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():
            for data in tqdm(train_loader, desc="extracting_features"):
                image_tensor = data['img'].to(self.device)
                gt_score = data['mos'].to(self.device)
                # Extracting the first item as it seems to be a list
                image_name = data['name'][0]

                info_tensor = self.llm_outputs(model, image_tensor)

                file_path = os.path.join(output_path, f"{image_name}.h5")
                with h5py.File(file_path, 'w') as hf:
                    hf.create_dataset('image_name', data=image_name)
                    hf.create_dataset(
                        'info_tensor', data=info_tensor.cpu().numpy())
                    hf.create_dataset('gt_score', data=gt_score.cpu().numpy())


if __name__ == '__main__':
    dnn_iter = DNNIter()
    data_loader = dnn_iter.data_loader()
    model = dnn_iter.get_model()
    dnn_iter.feature_extraction(data_loader, model)
