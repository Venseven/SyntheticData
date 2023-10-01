from tdgan.main import TDGANtrainer
import os,shutil
import json
import numpy as np
import argparse
import logging
import warnings
import pandas as pd

CONFIG = {

            "NUMERICAL_COLS": ["Total Costs of OSHPD Projects", "Number of OSHPD Projects"],
            "DATE_COLS": ["Data Generation Date"], 
            "DATE_DELIMITER": "/", 
            "MODELPATH": "/mnt/new/research/TGAN_tf/output/model/date_model.pkl", 
            "GPU": "0,1,2,3", 
            "final_date_columns": None
        }

PARAMS = {
            "max_epoch":5,
            "steps_per_epoch":10000,
            "batch_size":128,
            "z_dim":200,
            "noise":0.2,
            "l2norm":0.00001,
            "learning_rate":0.001,
            "num_gen_rnn":100,
            "num_gen_feature":100,
            "num_dis_layers":1,
            "num_dis_hidden":100
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass the Configure file and the datapath")
    parser.add_argument("--datapath",type=str, default="./data/date.csv", description = "TRAIN DATAFILE PATH")
    parser.add_argument("--data_format",choices=["csv", "excel"],default="csv", description = "TRAIN DATAFILE FORMAT")
    parser.add_argument("--resume_training",default=True, type=bool)
    pargs = parser.parse_args()

    if pargs.data_format=="csv":
        data = pd.read_csv(pargs.datapath)
    else:
        data = pd.read_excel(pargs.datapath)
       

    logging.info("Loaded Dataset")
    #Model Initialization

    model = TDGANtrainer(
                data=data,
                datapath=pargs.datapath,
                initialize_training=pargs.resume_training,
                CONFIG=CONFIG
                )

    model.train(PARAMS=PARAMS)
