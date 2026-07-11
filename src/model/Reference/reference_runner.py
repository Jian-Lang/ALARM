import re
from ..utils.mllm_factory import MLLMFactory
from PIL import Image
from pathlib import Path
from loguru import logger
from typing import Dict
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
import os
import json
import multiprocessing
from functools import partial
from omegaconf import OmegaConf

prompt_template = """
You have a set of experiences for identifying harmful memes, originally created by comparing similar but contradictory categories of two memes.
Now, a new experience arrives containing the description of one harmful and one similar but benign one, and a summary of the differences between the them.
Your task is to distill new references from the experience and update the existing references by choosing one operation: : ADD, EDIT, UPVOTE, and DOWNVOTE.

Strict Rules:

1. ADD only if:
   - add new references that are very different from exisitng references and relevatnt for other detection.
2. EDIT must:
   - if any existing reference is not general enough or can be enhanced, rewrite and improve it.
3. UPVOTE if:
   - if the existing reference is strongly relevant for current reference
4. DOWNVOTE if:
   - if one exisiing reference is contradictory or similar/duplicated to other existing reference. 
5. Maximum {size} references preserved
6. Output only valid JSON

Context:
Current references Set (importance order):
{cur_set_str}

New Coming Experience:
{new_experience}

Processing Steps:

1. Ensure the added and edited references are concise, clear while keeping them 2 or 3 sentences.
2. Ensure the references are concise and easy to follow.
3. Actively downvote references that are vague or hard to understand, and maintain the reference set at {size} items.
4. Try to make every reference useful, make more upvotes, and downvotes.
5. Refine references to emphasize distinct signals that uniquely identify specific harmful patterns.
6. Generalize references to extract universal principles that capture common traits of harmful content.
7. If UPVOTE/DOWNVOTE is used in an operation, include only 'none' in the reference of the operation.
8. Return only the JSON operations with the below format:
[
  {{
    "operation": "<ADD|EDIT|UPVOTE|DOWNVOTE>",
    "target": "<index/none>",
    "reference": "<new/revised text>"
  }}
]
"""

class ReferenceSetManager:
    def __init__(self, size: int):
        self.size = size
        self.set = []
        # Add a placeholder with a very high importance so it cannot be removed.
        self.set.append({
            "reference": "placeholder",
            "importance": 2
        })
      
    def init_from_df(self, df: pd.DataFrame):
        # Convert DataFrame to list of dictionaries with each row becoming a dictionary.
        self.set = df.to_dict(orient='records')

    def extract_instruction(self, json_str: str):
        """
        Expects a JSON string of operations in the form:
        [
          {
            "operation": "<ADD|EDIT|UPVOTE|DOWNVOTE>",
            "target": <index of target if applicable>,
            "reference": "<new or updated reference text if applicable>"
          }
        ]
        """
        pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
        match = pattern.search(json_str)
        if match:
            json_str = match.group(1).strip()
        try:
            operations = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {json_str}")
            return
        # if operations is dict turn to list
        if isinstance(operations, dict):
            operations = [operations]
      
        for operation in operations:
            op_type = operation.get("operation", "").upper()
            target = operation.get("target", None)
            reference_text = operation.get("reference", None)
            
            # if target is str, try to convert it to int
            if isinstance(target, str) and target != "" and target != 'none':
                try:
                    target = int(target)
                except ValueError as e:
                    # logger the detailed error
                    if op_type != 'ADD':
                        logger.error(f"Failed to convert target to int: {target}")
                        logger.error(f"Error: {e}")
                        continue
          
            if op_type == "ADD":
                self._add(reference_text)
            elif op_type == "EDIT":
                self._edit(target, reference_text)
            elif op_type == "UPVOTE":
                self._upvote(target)
            elif op_type == "DOWNVOTE":
                self._downvote(target)
            else:
                logger.error(f"Unknown operation: {op_type}")

        # remove items with importance <= 0
        self.set = [item for item in self.set if item["importance"] > 0]
        self.set.sort(key=lambda x: x["importance"], reverse=True)
        
        # if size of reference set reach max + 15, randomly delete the less important one and oldest one
        if len(self.set) > self.size + 5:
            # Sort the set by importance, then randomly select items to remove
            self.set.sort(key=lambda x: x["importance"], reverse=True)
            # Remove the oldest item that isn't the placeholder.
            self.set = self.set[:self.size]

    def _add(self, reference: str):
        logger.debug(f"Adding new reference with importance=2: {reference}")
        new_item = {"reference": reference, "importance": 2}
        self.set.append(new_item)
        # self._enforce_size()

    def _edit(self, index: int, new_reference: str):
        if not isinstance(index, int):
            logger.error(f"Index {index} is not an integer")
            return
        if index < 0 or index >= len(self.set):
            logger.error(f"Index {index} out of bounds")
            return
      
        logger.debug(f"Editing reference at index {index}, incrementing importance")
        self.set[index]["reference"] = new_reference
        self.set[index]["importance"] += 1

    def _upvote(self, index: int):
        if not isinstance(index, int):
            logger.error(f"Index {index} is not an integer")
            return
        if index < 0 or index >= len(self.set):
            logger.error(f"Index {index} out of bounds")
            return
      
        logger.debug(f"Upvoting reference at index {index}")
        self.set[index]["importance"] += 1

    def _downvote(self, index: int):
        if not isinstance(index, int):
            logger.error(f"Index {index} is not an integer")
            return
        if index < 0 or index >= len(self.set):
            logger.error(f"Index {index} out of bounds")
            return
      
        logger.debug(f"Downvoting reference at index {index}")
        # Avoid changing the placeholder's importance.
        if self.set[index]["importance"] < 9999:
            self.set[index]["importance"] -= 1

    def _enforce_size(self):
        # If we exceed the desired size, remove the oldest item that isn't the placeholder.
        while len(self.set) > self.size:
            # Check if the first item is the placeholder. If so, remove the next one.
            if self.set[0]["importance"] == 9999:
                logger.debug("Removing next oldest reference (placeholder is kept).")
                self.set.pop(1)
            else:
                logger.debug("Removing oldest reference.")
                self.set.pop(0)

    def save_to_df(self, path: str):
        df = pd.DataFrame(self.set)
        df.to_json(path, lines=True, index=False, orient='records')

    def get_cur_set_str(self):
        # Format each reference as index: reference (importance: X)
        return "\n".join(
            f"{i}: {item['reference']} (importance: {item['importance']})"
            for i, item in enumerate(self.set)
        )

    def export_data(self):
        """
        Export the internal set data as a serializable structure.
        
        Returns:
            dict: A serializable representation of the reference set
        """
        return {
            'size': self.size,
            'set': self.set.copy()  # Deep copy of the set data
        }
    
    def import_data(self, data):
        """
        Import data to recreate the reference set.
        
        Args:
            data (dict): Data structure containing 'size' and 'set' keys
        """
        if 'size' in data:
            self.size = data['size']
        if 'set' in data:
            self.set = data['set']
        # Sort by importance after import
        self.set.sort(key=lambda x: x["importance"], reverse=True)

class Reference_Runner:
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
        self.data_df = pd.read_json(f'data/{cfg.dataset}/data.jsonl', lines=True, dtype={'id': str})
        
        self.experience_name = cfg.experience_name
        self.reference_name = cfg.reference_name
        
        self.experience_df = pd.read_json(f'data/{cfg.dataset}/experience/{self.experience_name}.jsonl', lines=True, dtype={'id1': str, 'id2': str, 'experience': str})
        
        self.output_path = f'data/{cfg.dataset}/reference/{self.reference_name}.jsonl'
        self.reference_set_manager = ReferenceSetManager(size=cfg.size)
        # if save_path exists, load it
        if os.path.exists(self.output_path):
            self.save_df = pd.read_json(self.output_path, lines=True, dtype={'reference': str})
            self.reference_set_manager.init_from_df(self.save_df)
        else:
            self.save_df = pd.DataFrame()
        
        self.start = OmegaConf.select(cfg, "start", default=0)
        

    def run(self, num_processes=1):
        """
        Run inference on experiences in parallel with real-time experience set updates
        using a more elegant approach with data serialization
        
        Args:
            num_processes: Number of parallel processes to use
        
        Returns:
            None: Results are saved to the output file
        """
        # Get experiences that haven't been processed yet
        num_processes = 1
        experiences_to_process = []
        for index, row in self.experience_df.iterrows():
            # Skip processing if the index is less than the start value
            if index < self.start:
                continue
            experiences_to_process.append(row)
        # If no experiences to process, exit
        if not experiences_to_process:
            logger.info("No experiences to process.")
            return
        
        logger.info(f"Processing {len(experiences_to_process)} experiences using {num_processes} processes with synchronized experience set")
        
        # Create a multiprocessing manager to share data between processes
        with multiprocessing.Manager() as manager:
            # Create a lock for synchronization
            lock = manager.Lock()
            
            # Create a shared list to store results
            results = manager.list()
            
            # Create a shared dictionary to store the experience set data
            shared_reference = manager.dict()
            shared_reference['data'] = self.reference_set_manager.export_data()
            
            # Process experiences with a pool of workers
            with multiprocessing.Pool(processes=min(num_processes, len(experiences_to_process))) as pool:
                # Create a list of tasks, each with access to the shared experience data
                tasks = []
                for i, row in enumerate(experiences_to_process):
                    tasks.append((i, row, self.model_id, self.model_params, 
                                 self.reference_set_manager.size, 
                                 shared_reference, lock, results, self.output_path))
                
                # Process experiences in parallel with the shared experience set
                for _ in tqdm(
                    pool.imap_unordered(process_experience_with_sync, tasks),
                    total=len(tasks),
                    desc="Processing experiences"
                ):
                    pass
            
            # Convert shared list to regular list
            results_list = list(results)
            
            # Update the main experience set manager with the final shared data
            self.reference_set_manager.import_data(shared_reference['data'])
        
        # Final save of the experience set
        self.reference_set_manager.save_to_df(self.output_path)
        logger.info("All experiences processed successfully")

    def log_result(self):
        pass

def process_experience_with_sync(args):
    """
    Process a single experience with synchronized experience set updates
    using the exported/imported data approach
    
    Args:
        args: Tuple containing (index, row, model_id, model_params, size, 
                               shared_experiences, lock, results, output_path)
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    idx, row, model_id, model_params, size, shared_experiences, lock, results, output_path = args
    
    try:
        # Create a new model instance for this process
        model = MLLMFactory(model_id, model_params)
        
        id1 = row['id1']
        id2 = row['id2']
        experience = row['experience']
        
        # Create a temporary experience set manager and get the current experience set
        with lock:
            # Get the current shared experience set data
            current_data = shared_experiences['data']
        
        # Create a local experience set manager instance with the current data
        local_manager = ReferenceSetManager(size=size)
        local_manager.import_data(current_data)
        
        # Get the formatted string representation for the prompt
        current_experience_set_str = local_manager.get_cur_set_str()
        
        # Create prompt with the current experience set
        prompt = prompt_template.format(
            size=size,
            cur_set_str=current_experience_set_str,
            new_experience=experience
        )
        
        # Get instruction from model
        instruction = model.chat_text(prompt)
        
        result = {
            'id1': id1,
            'id2': id2,
            'experience': experience,
            'instruction': instruction
        }
        
        # Critical section: update the shared experience set and save results
        with lock:
            # Get the latest shared experience data (may have been updated by other processes)
            updated_data = shared_experiences['data']
            
            # Create an updated reference manager with the latest data
            updated_manager = ReferenceSetManager(size=size)
            updated_manager.import_data(updated_data)
            
            # Apply the instruction to update the reference set
            updated_manager.extract_instruction(instruction)
            
            # Update the shared experience data
            shared_experiences['data'] = updated_manager.export_data()
            
            # Save the updated experience set
            updated_manager.save_to_df(output_path)
            
            # Add result to the shared results list
            results.append(result)
            
            # Log progress every few items
            if len(results) % 5 == 0:
                logger.info(f"Processed {len(results)} experiencs")
        
        return True
    except Exception as e:
        logger.error(f"Error processing experience: {e}")
        return False