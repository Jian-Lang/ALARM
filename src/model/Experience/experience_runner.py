from ..utils.mllm_factory import MLLMFactory
from PIL import Image
from pathlib import Path
from loguru import logger
from typing import Dict
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
import os
import multiprocessing
from functools import partial

class Experience_Runner:
    def __init__(self, cfg: DictConfig, **kwargs):
        """
        Initialize Demo Runner
        
        Args:
            model_id: ID of the model to use
        """
        self.model_id = cfg.model_id
        self.model_params = cfg.para if cfg.para else {}
        self.model = MLLMFactory(self.model_id, self.model_params)
        self.image_path = f'data/{cfg.dataset}/img'
        self.result = None
        self.pairs_name = cfg.pairs_name
        self.experience_name = cfg.experience_name
        self.pair_df = pd.read_json(f'data/{cfg.dataset}/retrieve/{self.pairs_name}.jsonl', lines=True, dtype={'id1': str, 'id2': str})
        self.data_df = pd.read_json(f'data/{cfg.dataset}/data.jsonl', lines=True, dtype={'id': str})
        
        # self.pair_df = self.pair_df[self.pair_df['similarity']]
        # get top 1000 pairs
        # self.pair_df = self.pair_df.sort_values(by='similarity', ascending=False).head(1000)
        self.pair_df = self.pair_df.sort_values(by='similarity', ascending=False)
        self.save_path = f'data/{cfg.dataset}/experience/{self.experience_name}.jsonl'
        
        # if save_path exists, load it
        if os.path.exists(self.save_path):
            self.save_df = pd.read_json(self.save_path, lines=True, dtype={'id1': str, 'id2': str, 'experience': str})
        else:
            self.save_df = pd.DataFrame(columns=['id1', 'id2', 'experience']).astype({'id1': str, 'id2': str})
        
    def preprocess(self):
        pass
    
    def run(self, num_processes=4):
        """
        Run inference on multiple image pairs using multiprocessing
        
        Args:
            num_processes: Number of parallel processes to use
        
        Returns:
            None: Results are saved to the output file
        """
        # Get pairs that haven't been processed yet
        pairs_to_process = []
        for _, row in self.pair_df.iterrows():
            # Check if this exact pair (id1, id2) has already been processed
            if len(self.save_df) > 0 and ((self.save_df['id1'] == row['id1']) & (self.save_df['id2'] == row['id2'])).any():
                continue
            pairs_to_process.append((row['id1'], row['id2']))
        
        # If no pairs to process, exit
        if not pairs_to_process:
            logger.info("No pairs to process.")
            return
        
        logger.info(f"Processing {len(pairs_to_process)} pairs using {num_processes} processes")

        # Create partial function with fixed arguments
        process_func = partial(
            process_pair,
            model_id=self.model_id,
            model_params=self.model_params,
            image_path=self.image_path,
            data_df=self.data_df
        )

        results = []

        if num_processes <= 1:
            # Single-process mode: the local checkpoint is loaded once in __init__ and reused.
            # A forked worker cannot re-initialise CUDA, so a Pool would fail outright here.
            logger.info("Using single-process mode (required for local CUDA models)")
            for pair in tqdm(pairs_to_process, desc="Processing pairs"):
                results.append(process_pair_with_model(pair, self.model, self.image_path, self.data_df))
                if len(results) % 5 == 0:
                    results = self._save_results(results)
        else:
            # Multi-process mode: only suitable for API models (see process_pair)
            logger.info("Using multi-process mode (suitable for API models)")
            with multiprocessing.Pool(processes=min(num_processes, len(pairs_to_process))) as pool:
                # Process pairs in parallel and collect results
                for result in tqdm(
                    pool.imap_unordered(process_func, pairs_to_process),
                    total=len(pairs_to_process),
                    desc="Processing pairs"
                ):
                    results.append(result)

                    # Save results periodically
                    if len(results) % 5 == 0:
                        results = self._save_results(results)

        # Save remaining results
        if results:
            self._save_results(results)

        logger.info("All pairs processed successfully")

    def _save_results(self, results):
        """Append the buffered results to the save file and return an empty buffer"""
        new_results_df = pd.DataFrame(results)
        self.save_df = pd.concat([self.save_df, new_results_df], ignore_index=True).astype({'id1': str, 'id2': str})
        self.save_df.to_json(self.save_path, lines=True, index=False, orient='records')
        logger.info(f"Saved {len(self.save_df)} experiences so far")
        return []
        
    def log_result(self):
        pass
        # logger.info(f"experience: {self.result}")

def process_pair(pair_info, model_id, model_params, image_path, data_df):
    """
    Process a single pair of images in a separate process (API models only: every call
    builds its own model client, which is prohibitively expensive for a local checkpoint)

    Args:
        pair_info: Tuple containing (id1, id2)
        model_id: Model ID to use
        model_params: Model parameters
        image_path: Path to images
        data_df: DataFrame containing image and text data

    Returns:
        Dictionary with experience result
    """
    # Create a new model instance for this process
    model = MLLMFactory(model_id, model_params)
    return process_pair_with_model(pair_info, model, image_path, data_df)


def process_pair_with_model(pair_info, model, image_path, data_df):
    """
    Process a single pair of images with an already initialised model

    Args:
        pair_info: Tuple containing (id1, id2), where id1 is the meme predicted benign
                   and id2 the meme predicted harmful (see conduct_retrieval.greedy_matching)
        model: Model instance shared across pairs
        image_path: Path to images
        data_df: DataFrame containing image and text data

    Returns:
        Dictionary with experience result
    """
    id1, id2 = pair_info

    # Get image paths and texts
    img1_path = data_df[data_df['id'] == id1]['img'].values[0]
    img2_path = data_df[data_df['id'] == id2]['img'].values[0]
    
    # Load images
    img1 = Image.open(f'{image_path}/{img1_path}').convert("RGB")
    img2 = Image.open(f'{image_path}/{img2_path}').convert("RGB")
    
    # Get text content
    text1 = data_df[data_df['id'] == id1]['text'].values[0]
    text2 = data_df[data_df['id'] == id2]['text'].values[0]
    
    # The contrastive pairs are stored as (id1 = benign, id2 = harmful), so Meme1 is the
    # benign member and Meme2 the harmful one. The prompt must name them in that order,
    # otherwise the gathered experience explains the contrast with the polarity reversed.
    prompt = f"""
Meme1 Text: {text1}\nMeme2 Text: {text2}
Given two memes that are visually or structurally similar but belong to distinct categories: Meme2, which is harmful, and Meme1, which is benign. Please complete the following two steps:
Step 1:
Clearly summarize the content of each meme by carefully analyzing its image and textual element embedded in the image, and considering any implicit or explicit messages it conveys.
Step 2:
Based on the content of two memes, contrast the key differences between them to explain why Meme2 is classified as harmful content, while Meme1 remains benign.
    """
    
    # Get experience
    experience = model.chat_multi_img(prompt, [img1, img2])
    
    return {
        'id1': id1,
        'id2': id2,
        'experience': experience
    }