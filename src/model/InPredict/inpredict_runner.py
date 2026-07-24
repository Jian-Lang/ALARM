import numpy as np
from typing import Optional, Union
from PIL import Image
import torch
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline, AutoProcessor, AutoModel
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import json
import multiprocessing
from functools import partial

from ..utils.mllm_factory import MLLMFactory


prompt_template = """
Given a meme with its image and textual element '{text}' embedded in the image, your task is to determine whether this meme is harmful or benign by considering its multimodal content.
Moreover, a set of valuable detection references are also provided to serve as additional knowledge, which will help you in determine the class of the given meme.
References set:
{reference_set}.
Please leverage your pre-trained knowledge and the references to analyze and understand this meme, and give your final judgment.
Your output should strictly follow the format: Thought: [Your analysis] Answer: [harmful/harmless].
"""

class ReferenceSetManager:
    def __init__(self, size: int, strategy: str = 'topk'):
        self.size = size
        self.set = []
        self.set.append({'reference': 'placeholder'})
        # Using Jina CLIP for multimodal embeddings with 512 dimensions
        if strategy != 'topk':
            self.encoder = SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True, truncate_dim=512, device='cuda')
            
        self.reference_text_embeddings = None
        self.reference_image_embeddings = None
        self.item_to_references_map = {}
        
    def init_from_df(self, df: pd.DataFrame):
        # Convert DataFrame to list of dictionaries, with each row becoming a dict
        self.set = df.to_dict(orient='records')
        
    def get_cur_set_str(self, tok_k):
        # Limit the references to the top k
        top_k_references = self.set[:tok_k]
        
        # Format: index: reference
        return '\n'.join([f"{i}: {reference['reference']}" for i, reference in enumerate(top_k_references)])
    
    def extract_image_features(self, image_path):
        """Extract features from an image using CLIP model"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            # Preprocess image
            inputs = self.image_processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {key: val.to('cuda') for key, val in inputs.items()}
            # Get image features
            with torch.no_grad():
                outputs = self.image_model(**inputs)
                image_features = outputs.image_embeds
            return image_features[0].cpu()
        except Exception as e:
            logger.error(f"Error extracting image features from {image_path}: {e}")
            return None
    
    def compute_reference_embeddings(self, reference_img_dir):
        """Compute and store embeddings for references text and images"""
        logger.info("Computing embeddings for referencs (text and images)")
        
        # Get reference texts
        reference_texts = [item['reference'] for item in self.set]
        
        # Get reference image paths
        reference_images = []
        for item in self.set:
            if 'img' in item and item['img']:
                img_path = os.path.join(reference_img_dir, item['img'])
                reference_images.append(img_path)
            else:
                # If no image is specified, use a placeholder
                reference_images.append(None)
        
        # Encode texts
        self.reference_text_embeddings = self.encoder.encode(reference_texts, normalize_embeddings=True)
        
        # Encode images - Jina CLIP can handle file paths directly
        valid_images = [img for img in reference_images if img is not None and os.path.exists(img)]
        valid_indices = [i for i, img in enumerate(reference_images) if img is not None and os.path.exists(img)]
        
        if valid_images:
            image_embeddings = self.encoder.encode(valid_images, normalize_embeddings=True)
            
            # Initialize with zeros for all references
            self.reference_image_embeddings = np.zeros((len(reference_texts), self.reference_text_embeddings.shape[1]))
            
            # Fill in embeddings for valid images
            for idx, embedding_idx in enumerate(valid_indices):
                self.reference_image_embeddings[embedding_idx] = image_embeddings[idx]
        else:
            # If no valid images, use zeros
            self.reference_image_embeddings = np.zeros((len(reference_texts), self.reference_text_embeddings.shape[1]))
            
        # Convert to torch tensors if needed
        if not isinstance(self.reference_text_embeddings, torch.Tensor):
            self.reference_text_embeddings = torch.tensor(self.reference_text_embeddings)
        
        if not isinstance(self.reference_image_embeddings, torch.Tensor):
            self.reference_image_embeddings = torch.tensor(self.reference_image_embeddings)
        
    def compute_item_embeddings_and_topk_map(self, input_df, img_dir, reference_img_dir=None, top_k=5, alpha=0.2):
        """
        Compute embeddings for all items and map them to their top-k references
        considering both text and image similarities with 1:1 ratio
        
        Args:
            input_df: DataFrame containing item data
            img_dir: Directory containing item images
            reference_img_dir: Directory containing v images
            top_k: Number of top references to retrieve per item
            
        Returns:
            Dictionary mapping item IDs to their top-k reference indices
        """
        logger.info(f"Computing embeddings and top-{top_k} references for all items (text + image)")
        
        if reference_img_dir is None:
            reference_img_dir = img_dir
        
        # Ensure reference embeddings are computed
        if self.reference_text_embeddings is None or self.reference_image_embeddings is None:
            self.compute_reference_embeddings(reference_img_dir)
            
        # Prepare item data for embedding
        item_texts = []
        item_image_paths = []
        item_ids = []
        
        for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Preparing items"):
            img_path = os.path.join(img_dir, row["img"])
            # Assuming all images exist - no need to check
            item_texts.append(row["text"])
            item_image_paths.append(img_path)
            item_ids.append(row["id"])
                
        # Encode texts directly using Jina CLIP
        item_text_embeddings = self.encoder.encode(item_texts, normalize_embeddings=True)
        
        # Encode images directly using Jina CLIP
        item_image_embeddings = self.encoder.encode(item_image_paths, normalize_embeddings=True)
        
        # Convert to torch tensors if needed
        if not isinstance(item_text_embeddings, torch.Tensor):
            item_text_embeddings = torch.tensor(item_text_embeddings)
        
        if not isinstance(item_image_embeddings, torch.Tensor):
            item_image_embeddings = torch.tensor(item_image_embeddings)
        # print dtype to debug
        logger.info(f"Item text embeddings dtype: {item_text_embeddings.dtype}")
        logger.info(f"Item image embeddings dtype: {item_image_embeddings.dtype}")
        
            
        # Compute text similarity 
        text_similarity_scores = torch.matmul(item_text_embeddings, self.reference_text_embeddings.T)
        
        # Compute image similarity
        image_similarity_scores = torch.matmul(item_image_embeddings, self.reference_text_embeddings.T)
        
        # Combine text and image similarities with 1:1 ratio
        # combined_similarity = 0.2 * text_similarity_scores + 0.8 * image_similarity_scores
        combined_similarity = alpha * text_similarity_scores + (1-alpha) * image_similarity_scores
        
        # For each item, get the top-k references
        item_to_references = {}
        for i, item_id in enumerate(item_ids):
            # Get indices of top-k similar references using combined similarity
            top_k_indices = torch.topk(combined_similarity[i], k=min(top_k, len(self.set))).indices.tolist()
            # Map item ID to its top-k references
            item_to_references[item_id] = top_k_indices
            
        logger.info(f"Created top-{top_k} reference map for {len(item_to_references)} items")
        self.item_to_references_map = item_to_references
        
        self.encoder = None
        self.reference_text_embeddings = None
        self.reference_image_embeddings = None
        return item_to_references
    
    def get_item_specific_references(self, item_id, default_k=5, strategy='topk'):
        """Get the specific top-k references for an item"""
        if strategy == 'topk':
            return self.get_cur_set_str(default_k)
        else:
            if item_id in self.item_to_references_map:
                top_indices = self.item_to_references_map[item_id]
                references = [self.set[idx]['reference'] for idx in top_indices]
                return '\n'.join([f"{i}: {reference}" for i, reference in enumerate(references)])
            else:
                # Fallback to default top-k if item not found
                return self.get_cur_set_str(default_k)

def process_single_item(row, model_id, model_params, img_dir, reference_set_manager, prompt_template, custom_prompt=None, strategy=None):
    """
    Process a single item in a separate process (API models only: every call builds
    its own model client, which is prohibitively expensive for a local checkpoint)

    Args:
        row: DataFrame row containing data information
        model_id: Model ID to use
        model_params: Model parameters
        img_dir: Directory containing images
        reference_set_manager: Manager for references with item-specific mapping
        prompt_template: Template for the prompt
        custom_prompt: Custom prompt if provided

    Returns:
        Dictionary with processing result or None if failed
    """
    model = MLLMFactory(model_id, model_params)
    return process_single_item_with_model(
        row, model, img_dir, reference_set_manager, custom_prompt, strategy=strategy
    )


def process_single_item_with_model(row, model, img_dir, reference_set_manager, custom_prompt, strategy=None):
    """
    Process a single item with an already initialised model

    Args:
        row: DataFrame row containing data information
        model: Model instance shared across items
        img_dir: Directory containing images
        reference_set_manager: Manager for references with item-specific mapping
        custom_prompt: Prompt template to fill in
        strategy: Strategy used to select the references shown to the model

    Returns:
        Dictionary with processing result or None if failed
    """
    try:
        # Get image path and text
        img_path = os.path.join(img_dir, row["img"])
        if not os.path.exists(img_path):
            logger.warning(f"Image {img_path} not found")
            return None
            
        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        text = row["text"]
        label = row["label"]
        item_id = row["id"]
        if 'race' in row and 'entity' in row:
            extra_data = f"race: {row['race']}, entity: {row['entity']}"
            text = f"{text} {extra_data}"
        
        # Get item-specific references
        reference_set_str = reference_set_manager.get_item_specific_references(item_id, strategy=strategy)
        
        # Create prompt
            # logger.info(f"Using custom prompt: {custom_prompt}")
        prompt = custom_prompt.format(
            text=text,
            reference_set=reference_set_str
        )
        
        # Get model prediction
        thought = model.chat_img(prompt, image)
        prediction = thought.split("Answer:")[-1].strip()
        note = 'success'
        
        if 'harmful' in prediction.lower():
            prediction = 1
        elif 'harmless' in prediction.lower():
            prediction = 0
        elif 'benign' in prediction.lower():
            prediction = 0
        elif '1' in prediction:
            prediction = 1
        elif '0' in prediction:
            prediction = 0
        else:
            prediction = 1
            note = prediction
            logger.warning(f"Prediction is not harmful or benign: {prediction}")
        
        return {
            "id": row["id"],
            "pred": prediction,
            "label": label,
            "thought": thought,  # Keep the full response for debugging
            "note": note
        }
    except Exception as e:
        logger.error(f"Error processing item {row['id']}: {e}")
        return None

class InPredict_Runner():
    def __init__(self, cfg: DictConfig, cid: str, log_dir: Path):
        self.cfg = cfg
        self.model_id = cfg.model_id
        self.model_short_name = cfg.model_short_name
        self.dataset = cfg.dataset
        self.batch_size = cfg.batch_size
        self.cid = cid
        self.model_params = cfg.para if cfg.para else {}
        
        self.log_dir = log_dir
        self.output_dir = self.log_dir
        self.img_dir = f"data/{self.dataset}/img"
        self.reference_img_dir = f"data/{self.dataset}/reference_img" if os.path.exists(f"data/{self.dataset}/reference_img") else self.img_dir
        self.reference_name = cfg.reference_name
        
        self.reference_df = pd.read_json(f"data/{self.dataset}/reference/{self.reference_name}.jsonl", lines=True)
        self.data_file = f"data/{self.dataset}/data.jsonl"
        self.output_file = f"{self.output_dir}/inpred.jsonl"
        self.k = cfg.top_k
        
        self.prompt = OmegaConf.select(self.cfg, "prompt", default=prompt_template)
        logger.info(f"Using prompt: {self.prompt}")
        
        self.strategy = OmegaConf.select(self.cfg, "strategy", default="topk")
        self.reference_set_manager = ReferenceSetManager(size=30, strategy=self.strategy)
        self.reference_set_manager.init_from_df(self.reference_df)
        self.reference_str = self.reference_set_manager.get_cur_set_str(self.k)
        
        self.alpha = OmegaConf.select(self.cfg, "alpha", default=0.2)
        if OmegaConf.select(self.cfg, "extra_data"):
            self.extra_data = pd.read_json(f'data/{self.dataset}/model/PromptHate/clean_caption/mem.json')
        else:
            self.extra_data = None
        
    
    def log_result(self):
        result = self.get_result()
        logger.info(f"Accuracy: {result['acc']}, F1: {result['f1']}")
        logger.info(f"Positive F1: {result['positive_f1']}, Negative F1: {result['negative_f1']}")
        
    def get_result(self):
        df = pd.read_json(self.output_file, lines=True)
        accuracy = accuracy_score(df['label'].astype(int), df['pred'].astype(int))
        f1 = f1_score(df['label'].astype(int), df['pred'].astype(int), average='macro')
        
        # F1 scores for positive and negative samples
        pos_mask = df['label'] == 1
        neg_mask = df['label'] == 0
        
        # Calculate class-specific F1 scores
        # For positive class (label=1), use pos_label=1
        positive_f1 = f1_score(
            df[pos_mask]['label'].astype(int),
            df[pos_mask]['pred'].astype(int),
            average='binary',
            pos_label=1
        ) if pos_mask.any() else 0

        # For negative class (label=0), we consider 0 as the positive class
        # This ensures we're calculating F1 for the negative class correctly
        negative_f1 = f1_score(
            df[neg_mask]['label'].astype(int),
            df[neg_mask]['pred'].astype(int),
            average='binary',
            pos_label=0
        ) if neg_mask.any() else 0
        
        return {
            "acc": accuracy,
            "f1": f1,
            "positive_f1": positive_f1,
            "negative_f1": negative_f1
        }

    def run(self, num_processes=4):
        """
        Process a dataset using the specified MLLM model in parallel and save predictions
        
        Args:
            num_processes: Number of parallel processes to use
        """
        # Create output directory
        logger.info(f"Current template: {self.prompt}")
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = self.output_file

        # Read input data
        input_df = pd.read_json(self.data_file, lines=True)
        input_df = input_df[input_df['split'] == 'test']
        # Shuffle the dataframe with a fixed seed for reproducibility
        input_df = input_df.sample(frac=1, random_state=2025).reset_index(drop=True)
        
        # Check for already processed entries
        processed_df = pd.DataFrame(columns=['id', 'pred', 'label', 'thought'])
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            processed_df = pd.read_json(output_file, lines=True)
        processed_ids = set(processed_df['id'].values)

        # Filter unprocessed data
        to_process_df = input_df[~input_df['id'].isin(processed_ids)].reset_index(drop=True)
        
        # If no items to process, exit
        if len(to_process_df) == 0:
            logger.info("No items to process.")
            return
        
        # Pre-compute embeddings and create item-to-references mapping before processing
        logger.info("Pre-computing embeddings and creating item-to-references mapping with Jina CLIP")
        if self.strategy != 'topk':
            self.reference_set_manager.compute_item_embeddings_and_topk_map(
                input_df, 
                self.img_dir, 
                self.reference_img_dir, 
                top_k=self.k,
                alpha=self.alpha
            )
        
        items_to_process = []
        for _, row in to_process_df.iterrows():
            if self.extra_data is not None:
                entity = self.extra_data[self.extra_data['img'] == row['img']]['entity'].values[0]
                race = self.extra_data[self.extra_data['img'] == row['img']]['race'].values[0]
                row['race'] = race
                row['entity'] = entity
            items_to_process.append(row)
            
        
        logger.info(f"Processing {len(items_to_process)} items using {num_processes} processes")
        
        # Create partial function with fixed arguments
        process_func = partial(
            process_single_item,
            model_id=self.model_id,
            model_params=self.model_params,
            img_dir=self.img_dir,
            reference_set_manager=self.reference_set_manager,
            prompt_template=self.prompt,
            custom_prompt=self.prompt,
            strategy=self.strategy
        )
        
        results = []

        def collect(result):
            """Append one prediction, checkpointing the output file every 5 items"""
            nonlocal processed_df
            if not result:
                return
            processed_df = pd.concat([processed_df, pd.DataFrame([result])], ignore_index=True)
            results.append(result)
            if len(results) % 5 == 0:
                processed_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
                # Calculate current metrics
                accuracy = accuracy_score(processed_df['label'].astype(int), processed_df['pred'].astype(int))
                f1 = f1_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
                logger.info(f"Processed {len(results)} items. Current Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        if num_processes <= 1:
            # Single-process mode: the local checkpoint is loaded once and reused. Forking a
            # worker per item would reload the whole LMM for every meme, and a CUDA context
            # cannot be re-initialised in a forked child at all.
            logger.info("Using single-process mode (required for local CUDA models)")
            model = MLLMFactory(self.model_id, self.model_params)
            for row in tqdm(items_to_process, desc="Processing items"):
                collect(process_single_item_with_model(
                    row, model, self.img_dir, self.reference_set_manager, self.prompt, strategy=self.strategy
                ))
            processed_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        else:
            # Multi-process mode: only suitable for API models (see process_single_item)
            logger.info("Using multi-process mode (suitable for API models)")
            with multiprocessing.Pool(processes=min(num_processes, len(items_to_process))) as pool:
                # Process items in parallel and collect results
                for result in tqdm(
                    pool.imap_unordered(process_func, items_to_process),
                    total=len(items_to_process),
                    desc="Processing items"
                ):
                    collect(result)

                # Save final results
                processed_df.to_json(output_file, orient='records', lines=True, force_ascii=False)

        # Calculate final metrics
        accuracy = accuracy_score(processed_df['label'].astype(int), processed_df['pred'].astype(int))
        f1 = f1_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
        
        # Calculate F1 scores for positive and negative samples separately
        pos_samples = processed_df[processed_df['label'] == 1]
        neg_samples = processed_df[processed_df['label'] == 0]

        # Check if there are any positive samples before calculating F1
        if len(pos_samples) > 0:
            positive_f1 = f1_score(
                pos_samples['label'].astype(int),
                pos_samples['pred'].astype(int),
                average='binary'
            )
        else:
            positive_f1 = 0.0
            logger.warning("No positive samples found for F1 calculation")

        # Check if there are any negative samples before calculating F1
        if len(neg_samples) > 0:
            negative_f1 = f1_score(
                neg_samples['label'].astype(int),
                neg_samples['pred'].astype(int),
                average='binary'
            )
        else:
            negative_f1 = 0.0
            logger.warning("No negative samples found for F1 calculation")

        # Print F1 scores for both classes
        logger.info(f"Positive F1: {positive_f1:.4f},  Negative F1: {negative_f1:.4f}")
        logger.info(f"Processing complete. Final Accuracy: {accuracy:.4f}, F1: {f1:.4f}")