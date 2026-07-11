from ..utils.mllm_factory import MLLMFactory
from typing import Optional, Union
from PIL import Image
import torch
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline
from pathlib import Path
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from ..utils.data_factory import DataloaderFactory, DataDfFactory


class Label_Runner():
    def __init__(self, cfg: DictConfig, cid: str, log_dir: Path):
        self.cfg = cfg
        self.cid = cid
        self.log_dir = log_dir
        self.output_dir = self.log_dir
        self.img_dir = f"data/{self.cfg.dataset}/img"
        
        # self.output_file = f"{self.output_dir}/pred.jsonl"
        self.label_name = cfg.label_name
        self.output_file = Path('data') / self.cfg.dataset / 'label' / f'{self.label_name}.jsonl'
        
        if self.cfg.prompt:
            logger.info(f"Using custom prompt: {self.cfg.prompt}")
        self.para = self.cfg.para if self.cfg.para else {}
        
        self.model = MLLMFactory(self.cfg.model_id, self.para)
        self.datadf = DataDfFactory(self.cfg.dataset, **self.cfg.data)
        self.prompt = self.cfg.prompt if self.cfg.prompt else ""
    
    def log_result(self):
        processed_df = pd.read_json(self.output_file, lines=True)
        y_true = processed_df['label'].astype(int)
        y_prob0 = processed_df['prob0'].astype(float)
        y_prob1 = processed_df['prob1'].astype(float)

        # Calculate proper confidence scores - the probability of the predicted class
        # For each sample, confidence is the probability corresponding to the predicted class
        predicted_class = (y_prob1 >= 0.5).astype(int)
        confidence_scores = np.array([y_prob1[i] if pred == 1 else y_prob0[i] for i, pred in enumerate(predicted_class)])
        
        # Test different confidence thresholds
        thresholds = np.arange(0.999, 1.0, 0.00001)
        
        total_samples = len(y_true)
        
        for threshold in thresholds:
            # Get indices of high confidence predictions
            confident_idx = confidence_scores >= threshold
            
            # Only evaluate on high confidence predictions
            y_true_confident = y_true[confident_idx]
            y_pred_confident = (y_prob1[confident_idx] >= 0.5).astype(int)
            
            # Calculate metrics
            if len(y_true_confident) > 0:
                acc = accuracy_score(y_true_confident, y_pred_confident)
                f1 = f1_score(y_true_confident, y_pred_confident, average='macro')
                coverage = len(y_true_confident) / total_samples
                
                logger.info(f"Confidence threshold {threshold:.5f} - Total samples: {total_samples}, Samples above threshold: {len(y_true_confident)}, Coverage: {coverage:.2%}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
            else:
                logger.info(f"No samples above confidence threshold {threshold:.1f} out of {total_samples} total samples")
                
        # Sort samples by confidence
        sorted_indices = np.argsort(confidence_scores)[::-1]  # Descending order
        sorted_confidences = confidence_scores[sorted_indices]
        logger.info(f"Top 10 confidences: {sorted_confidences[:10]}")
        sorted_y_true = y_true.iloc[sorted_indices]
        sorted_y_pred = predicted_class[sorted_indices]
        
        total_samples = len(y_true)
        
        # Evaluate coverage levels from 10% to 50% with 10% increments
        coverage_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        logger.info("Performance at different coverage levels:")
        logger.info("--------------------------------")
        
        # Debug: Print all confidence scores to verify sorting
        logger.info(f"Verifying confidence scores are properly sorted (descending): {np.all(np.diff(sorted_confidences) <= 0)}")
        
        for coverage in coverage_levels:
            # Calculate the number of samples for this coverage level
            n_samples = int(total_samples * coverage)
            
            if n_samples == 0:
                continue
                
            # Get threshold for this coverage level
            threshold = sorted_confidences[n_samples-1] if n_samples < len(sorted_confidences) else 0
            
            # Log information about thresholds at different positions
            if coverage == 0.1:
                logger.info(f"Confidence scores at positions 0, n_samples/2, n_samples-1: "
                           f"{sorted_confidences[0]:.5f}, {sorted_confidences[n_samples//2]:.5f}, {sorted_confidences[n_samples-1]:.5f}")
            
            # Calculate metrics for samples at this coverage level
            current_y_true = sorted_y_true.iloc[:n_samples]
            current_y_pred = sorted_y_pred[:n_samples]
            
            acc = accuracy_score(current_y_true, current_y_pred)
            f1 = f1_score(current_y_true, current_y_pred, average='macro')
            
            logger.info(f"Coverage: {coverage:.1%} - Threshold: {threshold:.5f}, Samples: {n_samples}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
             
    def get_result(self):
        processed_df = pd.read_json(self.output_file, lines=True)
        accuracy = accuracy_score(processed_df['label'].astype(int), processed_df['pred'].astype(int))
        f1 = f1_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
        return {
            "acc": accuracy,
            "f1": f1
        }

    def run(self, num_processes):
        """
        Process a dataset using the specified MLLM model and save predictions
        
        Args:
            input_file: Path to input jsonl file
            output_dir: Directory to save output predictions
            model_name: Name of the model to use
            img_dir: Directory containing images
        """
        # Create output directory
        output_file = self.output_file

        # Read input data
        input_df = self.datadf
        input_df = input_df.sample(frac=1).reset_index(drop=True)
        
        # Check for already processed entries
        processed_df = pd.DataFrame(columns=['id', 'pred', 'label'])
        if os.path.exists(output_file):
            processed_df = pd.read_json(output_file, lines=True)
        processed_ids = set(processed_df['id'].values)

        # Filter unprocessed data
        to_process_df = input_df[~input_df['id'].isin(processed_ids)]
        
        accuracy = 0.0
        f1 = 0.0
        pbar = tqdm(total=len(to_process_df), desc="Processing", position=0, leave=True)
        for idx, row in to_process_df.iterrows():
            pbar.set_description(f"ID: {row['id']}, ACC: {accuracy:.2f}, F1: {f1:.2f}")
            
            # Get image path and text
            img_path = os.path.join(self.img_dir, row["img"])
            if not os.path.exists(img_path):
                continue
                
            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            text = row["text"]
            label = row["label"]

            prompt = f"""
Given the meme with its image and the textual element '{text}' embedded in the image, your task is to carefully and critically assess whether this meme is harmful or not, in order to maintain the benignness and integrity of information on the Internet. 
Please leverage your extensive knowledge to deeply analyze and understand this meme, and give your final judgment.
Please not that while your primary goal is to provide a judgment after thoughtful analysis, it’s important to avoid overgeneralizing or being overly conclusive in cases where ambiguity exists.
You must only return `1' for harmful, or `0' for benign.
            """
            if self.prompt:
                prompt = f"Given the meme, with the Text: '{text}' embedded in the image\n{self.prompt}"
            # Get model prediction
            prediction, probs = self.model.chat_label(prompt, image)
            
            # Save prediction
            output_dict = {
                "id": row["id"],
                "pred": prediction,
                "label": label,
                "prob0": probs[0],
                "prob1": probs[1]
            }
            
            processed_df = pd.concat([processed_df, pd.DataFrame([output_dict])], ignore_index=True)
            processed_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
            # calculate accuracy and macro f1 score
            accuracy = accuracy_score(processed_df['label'].astype(int), processed_df['pred'].astype(int))
            f1 = f1_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
            # print(f"Accuracy: {accuracy}, Macro F1 Score: {f1}")
            
            pbar.update(1)

        pbar.close()
        
        # calculate accuracy and macro f1 score
        accuracy = accuracy_score(processed_df['label'].astype(int), processed_df['pred'].astype(int))
        f1 = f1_score(processed_df['label'].astype(int), processed_df['pred'].astype(int), average='macro')
        # print(f"Accuracy: {accuracy}, Macro F1 Score: {f1}")