import argparse
import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score  # Add import for metrics

DATASETS = ['FHM', 'MAMI', 'ToxiCN']


def compute_batch_similarities(query_features_img: np.ndarray,
                             query_features_txt: np.ndarray, 
                             base_features_img: np.ndarray,
                             base_features_txt: np.ndarray,
                             query_ids: List[int],
                             base_ids: List[int],
                             batch_size: int = 100) -> List[Dict]:
    """
    Compute similarities between query and base features in batches for both image and text
    """
    results = []
  
    for i in tqdm(range(0, len(query_features_img), batch_size)):
        batch_query_img = query_features_img[i:i+batch_size]
        batch_query_txt = query_features_txt[i:i+batch_size]
        batch_ids = query_ids[i:i+batch_size]
      
        # Compute cosine similarities for both modalities
        similarities_img = 1 - cdist(batch_query_img, base_features_img, metric='cosine')
        similarities_txt = 1 - cdist(batch_query_txt, base_features_txt, metric='cosine')
        
        # Combined similarities (simple addition)
        similarities = similarities_img + similarities_txt
      
        for idx, (qid, sims) in enumerate(zip(batch_ids, similarities)):
            matches = [
                {
                    'id': base_ids[i],
                    'similarity': float(sims[i])
                }
                for i in range(len(sims))
                if qid != base_ids[i]  # Avoid self-matching
            ]
            # Sort by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)
          
            results.append({
                'id': qid,
                'matches': matches
            })
  
    return results

def greedy_matching(similarities: List[Dict],
                    data_df: pd.DataFrame,
                    max_pairs: int = None) -> List[Tuple[int, int, float]]:
    """
    Perform greedy matching based on similarities, 
    ensuring that id1 is non-hateful (label=0) and id2 is hateful (label=1).
    """
    all_pairs = []
    used_ids = set()
    data_df['pred'] = data_df['pred'].astype(int)
    # Create id-to-label mapping
    id_to_label = dict(zip(data_df['id'], data_df['pred']))
  
    for item in similarities:
        query_id = item['id']
        query_label = id_to_label[query_id]
      
        for match in item['matches']:
            base_id = match['id']
            base_label = id_to_label[base_id]
          
            if query_label == 0 and base_label == 1:
                all_pairs.append((query_id, base_id, match['similarity']))
  
    # Sort pairs by similarity
    all_pairs.sort(key=lambda x: x[2], reverse=True)
  
    # Greedy matching
    final_pairs = []
    for q_id, b_id, sim in all_pairs:
        if q_id not in used_ids and b_id not in used_ids:
            final_pairs.append((q_id, b_id, sim))
            used_ids.add(q_id)
            used_ids.add(b_id)
          
            if max_pairs and len(final_pairs) >= max_pairs:
                break
  
    return final_pairs

def main(datasets: List[str]):
    # Create a range of coverage rates from 0.1 to 1.0 with interval of 0.1
    coverage_rates = [round(0.1 * i, 1) for i in range(1, 11)]  # [0.1, 0.2, ..., 1.0]
  
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        dataset_path = Path('data') / dataset
      
        # Load features for both modalities
        features_img = torch.load(dataset_path / 'fea' / 'image_embed.pt', weights_only=True)
        features_txt = torch.load(dataset_path / 'fea' / 'text_embed.pt', weights_only=True)
      
        # Load predictions and data
        pred_df = pd.read_json(dataset_path / 'label/label_pred.jsonl', lines=True)
        data_df = pd.read_json(dataset_path / 'data.jsonl', lines=True)
        
        y_true = pred_df['label'].astype(int)
        y_prob0 = pred_df['prob0'].astype(float)
        y_prob1 = pred_df['prob1'].astype(float)
      
        # Filter to training data first
        data_df = data_df[data_df['split'] == 'train']
        
        # Keep only predictions for training data
        pred_df = pred_df[pred_df['id'].isin(data_df['id'])]
        
        predicted_class = (y_prob1 >= 0.5).astype(int)
        
        # Calculate confidence scores - use max probability as confidence
        pred_df['confidence'] = np.array([y_prob1[i] if pred == 1 else y_prob0[i] for i, pred in enumerate(predicted_class)])
        
        # Process each coverage rate
        for coverage_rate in coverage_rates:
            print(f"\nProcessing with coverage rate: {coverage_rate:.1f}")
            
            # Sort by confidence (highest first) and take top percentage
            n_samples = int(len(pred_df) * coverage_rate)
            high_conf_df = pred_df.sort_values('confidence', ascending=False).head(n_samples)
            high_conf_ids = high_conf_df['id'].tolist()
            
            # Calculate and log metrics for high confidence samples
            high_conf_y_true = high_conf_df['label'].astype(int)
            high_conf_y_pred = high_conf_df['pred'].astype(int)
            
            accuracy = accuracy_score(high_conf_y_true, high_conf_y_pred)
            f1 = f1_score(high_conf_y_true, high_conf_y_pred, average='macro')
            
            print(f"Metrics for top {coverage_rate:.1%} confidence samples:")
            print(f"Coverage: {coverage_rate:.1%} - Samples: {n_samples}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            print("----------------------------")
            
            # Filter data to only include high confidence samples
            filtered_data_df = data_df[data_df['id'].isin(high_conf_ids)]
          
            filtered_high_conf_ids = filtered_data_df['id'].tolist()
            print(f"Number of high confidence samples: {len(filtered_high_conf_ids)}")
          
            # Get features for high confidence predictions
            high_conf_features_img = []
            high_conf_features_txt = []
            valid_ids = []
          
            for id_ in filtered_high_conf_ids:
                if id_ in features_img and id_ in features_txt:
                    high_conf_features_img.append(features_img[id_].cpu().numpy())
                    high_conf_features_txt.append(features_txt[id_].cpu().numpy())
                    valid_ids.append(id_)
                    
            print(f"Number of valid high confidence samples: {len(valid_ids)}")
          
            high_conf_features_img = np.array(high_conf_features_img)
            high_conf_features_txt = np.array(high_conf_features_txt)
          
            # Compute similarities using both modalities
            similarities = compute_batch_similarities(
                high_conf_features_img,
                high_conf_features_txt,
                high_conf_features_img,
                high_conf_features_txt,
                valid_ids,
                valid_ids
            )
          
            # Perform matching
            matched_pairs = greedy_matching(similarities, pred_df)
          
            # Save results with coverage rate in filename
            output_path = dataset_path / 'retrieve' / f'pairs_coverage_{coverage_rate:.1f}.jsonl'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                for pair in matched_pairs:
                    result = {
                        'id1': pair[0],
                        'id2': pair[1],
                        'similarity': pair[2]
                    }
                    f.write(json.dumps(result) + '\n')
          
            print(f"Found {len(matched_pairs)} pairs")
            print(f"Results saved to {output_path}")
            print("----------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build contrastive (benign, harmful) pairs from the label stage output")
    parser.add_argument('--dataset', nargs='+', default=DATASETS, choices=DATASETS,
                        help="Datasets to process (default: all of them)")
    args = parser.parse_args()
    main(args.dataset)