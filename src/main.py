import hydra
from omegaconf import DictConfig
import wandb
from loguru import logger
from datetime import datetime
from pathlib import Path
import sys
import os
from omegaconf import OmegaConf
from dotenv import load_dotenv

from model.Label.label_runner import Label_Runner
from model.Experience.experience_runner import Experience_Runner
from model.Reference.reference_runner import Reference_Runner
from model.InPredict.inpredict_runner import InPredict_Runner
from model.utils.models.api_model import APIModel
from utils.core_utils import (
    copy_config_file,
    calculate_md5
)

log_path = Path(f'log/{datetime.now().strftime("%m%d-%H%M%S")}')

class Inferrer():
    def __init__(self, cfg: DictConfig, cid: str, log_dir: Path):
        self.cfg = cfg
        
        match cfg.task:
            case "label":
                self.runner = Label_Runner(
                    cfg=self.cfg,
                    cid=cid,
                    log_dir=log_dir,
                )
            case "experience":
                self.runner = Experience_Runner(
                    cfg=self.cfg,
                    cid=cid,
                    log_dir=log_dir,
                )
            case "reference":
                self.runner = Reference_Runner(
                    cfg=self.cfg,
                )
            case "inpredict":
                self.runner = InPredict_Runner(
                    cfg=self.cfg,
                    cid=cid,
                    log_dir=log_dir,
                )
            case _:
                raise NotImplementedError(f"Task {cfg.task} not implemented")

    def run(self):
        # Local LMM backbones must stay single-process: one checkpoint is loaded and reused,
        # and a CUDA context cannot be re-initialised inside a forked worker. Only API
        # backbones are parallelised, using num_process (config) or NUM_PROCESSES (.env).
        if self.cfg.model_id in APIModel.model_list():
            num_processes = int(OmegaConf.select(self.cfg, "num_process", default=os.getenv("NUM_PROCESSES", 1)))
        else:
            num_processes = 1
        logger.info(f"Running task {self.cfg.task} with {num_processes} process(es)")
        self.runner.run(num_processes=num_processes)
        self.runner.log_result()
@hydra.main(version_base=None, config_path="config", config_name="label_qwen_FHM")
def main(cfg: DictConfig):
    config_str = OmegaConf.to_yaml(cfg)
    config_md5 = calculate_md5(config_str)[:8]
    
    logger.remove()
    log_dir = Path(f'log/{cfg.task}-{cfg.model_short_name}-{cfg.dataset}/{config_md5}')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{datetime.now().strftime("%m%d-%H%M%S")}.log'
    logger.add(log_path, level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Log Path: {log_path}")
    
    inferrer = Inferrer(cfg, cid=config_md5, log_dir=log_dir)
    inferrer.run()

if __name__ == "__main__":
    load_dotenv()
    copy_config_file()
    main()