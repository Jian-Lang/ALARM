import argparse
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os
from pathlib import Path


DATASETS = ['FHM', 'MAMI', 'ToxiCN']

parser = argparse.ArgumentParser(description="Encode memes with Jina-CLIP for the retrieval stage")
parser.add_argument('--dataset', nargs='+', default=DATASETS, choices=DATASETS,
                    help="Datasets to encode (default: all of them)")
datasets = parser.parse_args().dataset

model = SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True, truncate_dim=512, device='cuda')
batch_size = 128

# class MemeDatset:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.data_df = pd.read_json(f'data/{dataset}/data.jsonl', lines=True)

#     def __len__(self):
#         return len(self.data_df)
    
#     def __getitem__(self, idx):
#         text = self.data_df.iloc[idx]['text']
#         img = self.data_df.iloc[idx]['img']
#         img = Image.open(f'data/{self.dataset}/img/{img}').convert("RGB")
#         return text, img
        

# iterate datasets
for dataset in datasets:
    joint_fea_dict = {}
    image_fea_dict = {}
    text_fea_dict = {}

    data_df = pd.read_json(f'data/{dataset}/data.jsonl', lines=True)
    # iterate label_df
    num_samples = len(data_df)
    for start_idx in tqdm(range(0, num_samples, batch_size), desc=f'Processing {dataset}'):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_df = data_df.iloc[start_idx:end_idx]

        texts = batch_df["text"].tolist()
        ids = batch_df["id"].tolist()
        img_paths = batch_df["img"].tolist()

        images = []
        for img_name in img_paths:
            img_path = f'data/{dataset}/img/{img_name}'
            img = Image.open(img_path).convert("RGB")
            images.append(img)

        image_embeddings = model.encode(images, normalize_embeddings=True, convert_to_tensor=True)
        text_embeddings = model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)

        for mid, img_emb, txt_emb in zip(ids, image_embeddings, text_embeddings):
            join_embeddings = torch.cat([img_emb, txt_emb], dim=-1)
            joint_fea_dict[mid] = join_embeddings.cpu()
            image_fea_dict[mid] = img_emb.cpu()
            text_fea_dict[mid] = txt_emb.cpu()
    
    fea_path = Path(f'data/{dataset}/fea/')
    # Create the directory if it doesn't exist
    os.makedirs(fea_path, exist_ok=True)
    torch.save(joint_fea_dict, fea_path / 'joint_embed.pt')
    torch.save(image_fea_dict, fea_path / 'image_embed.pt')
    torch.save(text_fea_dict, fea_path / 'text_embed.pt')
