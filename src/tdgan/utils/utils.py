import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sn
from tqdm import tqdm
import os, glob, shutil
from functools import partial
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import warnings
from dateutil.parser import parse
import json
import logging
from pathlib import Path



def get_categorical(out_mat):
    out_val_ret = np.zeros((len(out_mat), 2))
    for i, out in enumerate(out_mat):
        out_val_ret[i][int(out)] = 1
    return out_val_ret

def argmax(lis, thrsh):
    out_list = []
    for ent in lis:
        ent = list(ent)
        ent_out = 1 if ent[1] > thrsh else 0
        out_list.append(ent_out)
    return out_list

def get_unique_indications(indication_list):
    unique_list = []
    for indication_val in indication_list:
        if isinstance(indication_val, str) and not (indication_val == '-'):
            val = indication_val.lower()
            val = val.replace('and',',')
            val = val.replace(' ','')
            local_indication_list = list(val.split(","))
            for indication in local_indication_list:
                if indication not in unique_list:
                    unique_list.append(indication)
    return unique_list

def expand_indication(indications, pivot_indication):
    if isinstance(indications, str) and not (indications == '-'):
        val = indications.lower()
        val = val.replace('and',',')
        val = val.replace(' ','')
        indication_list = list(val.split(","))
        presence_val  = 1 if pivot_indication in indication_list else 0
    else:
        presence_val = 0
    return presence_val

def fill_nan(dataf, col_names, categorical = False):
    for col in col_names:
        if dataf[col].isnull().values.any():
            if categorical == True:
                dataf[col] = dataf[col].astype(str)
                replace_val = 'NULL'
                dataf[col].fillna(replace_val, inplace = True)
            else:
                try:
                    val = dataf[col].astype(float).mean()
                    dataf[col].fillna(val, inplace = True)
                except:
                    print(f"EXCEPTION: Can't take mean for the value in the {col} column")
    return dataf

def val_cat_encoding(dataf, cols,cat_encode_dict):
    #cols= list(cat_encode_dict.keys())
    for col in cols:
        col_label_dict = cat_encode_dict[col]
        new_val = []
        for val in dataf[col].values:
            if val in col_label_dict:
                new_val.append(col_label_dict[val])
            elif 'OTHER' in col_label_dict:
                new_val.append(col_label_dict['OTHER'])
            elif 0.8 in col_label_dict:
                new_val.append(col_label_dict[0.8])
            elif 'NULL' in col_label_dict:
                new_val.append(col_label_dict['NULL'])
            elif 0.9 in col_label_dict:
                new_val.append(col_label_dict[0.9])
            else:
                new_val.append(col_label_dict[list(col_label_dict.keys())[0]])
        dataf[col] = new_val
    return dataf

def train_cat_encoding(dataf, cat_cols, out_file_name= False):
    le = LabelEncoder()
    cat_encode_dict = OrderedDict()
    for col in cat_cols:
        try:
            classes = le.fit(dataf[col]).classes_
        except:
            print(f'The column name {col} cannot be label encoded')
        col_label_dict = OrderedDict()
        for label, clas in enumerate(classes):
            col_label_dict[clas] = label
        cat_encode_dict[col] = col_label_dict
    if out_file_name:
        with open(out_file_name, 'wb') as f:
            pkl.dump(cat_encode_dict, f)
        print(f"The label encode dict is saved in {out_file_name}")
    for col in cat_cols:
        col_label_dict  =cat_encode_dict[col]
        label_val = [col_label_dict[val] for val in dataf[col].values]
        dataf[col] = label_val
    return dataf, cat_encode_dict

def get_metrics(act_out, pred_out):
    print(f"The accuracy is {accuracy_score(act_out, pred_out)}")
    print("The Classification Report \n")
    print(classification_report(act_out, pred_out))
    cm = confusion_matrix(act_out, pred_out)
    uniques = list(range(2))
    df_cm = pd.DataFrame(cm, index = [i for i in uniques], columns= [i for i in uniques])
    plt.figure(figsize = (7,6))
    cm_plot = sn.heatmap(df_cm, annot = True, fmt = 'd')
    print("Confusion Matrix")
    plt.show()

def get_iter_metrics(act_out, pred_out):
    class_report = classification_report(act_out, pred_out, output_dict= True)
    acc = accuracy_score(act_out, pred_out)
    classes = [ent for ent in list(class_report.keys()) if len(ent) < 5]
    out_metric_text = f"Accuracy: {acc}"
    metrics = ['precision', 'recall']

    for metric in metrics:
        out_metric_text += f" {metric} ["
        for clas in classes:
            out_metric_text += f" {clas}:{class_report[clas][metric]}"
        out_metric_text += "]"
    return out_metric_text


def save_metadata(path, data):
    with open(path,"w") as file:
        file.write(json.dumps(data))
    logging.info("Saved metadata and config file")

def directory_updates(CACHE_DIR, OUTPUT_PATH):
    log_dir = os.path.join(OUTPUT_PATH, "./logs")
    model_dir =  os.path.join(OUTPUT_PATH, "./model")
    backup_dir = os.path.join(model_dir, "./backup")
    if os.path.exists(os.path.join(CACHE_DIR, 'cluster_data.json')):
                os.remove(os.path.join(CACHE_DIR, 'cluster_data.json'))
    if os.path.exists(log_dir):
        files = glob.glob(f'{log_dir}/*')
        for f in files:
            os.remove(f)
    #creating backup
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
   
    # fetch all files
    for file_name in os.listdir(model_dir):
        # construct full file path
        source = os.path.join(model_dir , file_name)
        destination = os.path.join(backup_dir , file_name)
        # move only files
        if os.path.isfile(source):
            shutil.move(source, destination)

def update_config(CONFIG, config_file_path):
    with open(config_file_path,"r") as read_file:
                config=json.load(read_file)
    CONFIG["final_date_columns"] = config['final_date_columns']
    return CONFIG

def check_integrity(samples, numerical_columns):
    for column in numerical_columns:
        samples[column] = samples[column].apply(lambda x: 0 if x<0 else round(x))
    samples.fillna("", inplace = True)
    return samples

def get_column_info(meta_path, config_file):
    with open(meta_path, 'r') as read_file:
            metadata = json.load(read_file)
    if config_file['DATE_COLS']:
            date_columns = [metadata["columns"][i] for i in metadata["datecolumns"]]
    numerical_columns = metadata["continous_columns"]
    return date_columns, numerical_columns