from tdgan.main import TDGANSampler
import os,shutil
import json
import numpy as np
import argparse
import logging
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level= logging.INFO)

OUTPUT_PATH = "./output/data/"
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

CONFIG = {

            "NUMERICAL_COLS": ["Total Costs of OSHPD Projects", "Number of OSHPD Projects"],
            "DATE_COLS": ["Data Generation Date"], 
            "DATE_DELIMITER": "/", 
            "MODELPATH": "/mnt/new/research/TGAN_tf/output/model/date_model.pkl", 
            "GPU": "0,1,2,3", 
            "final_date_columns": None
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass the Configure file and the datapath")
    parser.add_argument("--num_samples",type=int, default=10000, description = "NUMBER OF SYNTHETIC SAMPLES")
    parser.add_argument("--model_path",type=str,description = "TRAINED MODEL PATH",default="./output/model/date_model.pkl")
    pargs = parser.parse_args()

    sampler = TDGANSampler(CONFIG=CONFIG)
    sampled_df = sampler(n_samples = pargs.num_samples, model_path= "/mnt/new/research/TGAN_tf/output/model/date_model.pkl")
    sampled_df.to_csv(os.path.join(OUTPUT_PATH, 'sample.csv'), index = False)
