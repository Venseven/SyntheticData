import os,shutil
import json
import numpy as np
from tqdm import tqdm
import argparse
import logging
import warnings
import pandas as pd
from tdgan.model import TDGANModel
from tdgan.data import Preprocessing, DatePostProcess
from tdgan.utils.config import model_loader
from tdgan.utils.utils import get_column_info, save_metadata, check_integrity, update_config, directory_updates
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level= logging.INFO)


CACHE_DIR = "./caches/"
METAFILE = os.path.join(CACHE_DIR, 'metafile.json')
PROCESSED_DATA_FILE = 'Processed_data.csv'
OUTPUT_PATH = "./output/"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

class TDGANtrainer(Preprocessing):

    def __init__(self, data, datapath, CONFIG, initialize_training = True):
        super().__init__(data, CONFIG) 
        self.datecolumns = None
        self.initialize_training = initialize_training
        self.data = data
        self.datapath = datapath
        self.data,self.metadata,self.numerical_columns = self.get_data()
        self.CONFIG = CONFIG
        self.CACHE_DIR =  "./caches/"
        self.METAFILE =  os.path.join(self.CACHE_DIR, 'metafile.json')
        if self.CONFIG['DATE_COLS']:
            self.datecolumns = [self.metadata["columns"][i] for i in self.metadata["datecolumns"]]
        save_metadata(path=METAFILE,data=self.metadata)


    def initialize_model(self, PARAMS):
        return model_loader(model=TDGANModel,
                            model_path=self.CONFIG['MODELPATH'],
                            numerical_columns=self.numerical_columns,
                            gpu_in_str=self.CONFIG['GPU'],
                            epoch=PARAMS['max_epoch'],
                            params=PARAMS,
                            datecolumns=self.datecolumns
                            )
                            
    def train(self, PARAMS):
        model = self.initialize_model(PARAMS)
        logging.info("Started Training Model")
        model = model.fit(self.data,self.CONFIG,self.datapath)
        logging.info("Training Finished")
        
class TDGANSampler(TDGANModel):

    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.date_columns, self.numerical_columns = get_column_info(meta_path=METAFILE,config_file=self.CONFIG)

    def __call__(self,n_samples, model_path):
        model = self.load(model_path)
        samples = model.sample(n_samples)
        samples = check_integrity(samples, self.numerical_columns)
        postprocess_pipeline = DatePostProcess(samples, METAFILE, self.CONFIG)
        date_exists = True if self.date_columns else False
        sampled_df = postprocess_pipeline.post_process_dates(date_exists=date_exists)
        return sampled_df
