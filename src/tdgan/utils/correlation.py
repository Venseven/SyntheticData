#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import numpy as np
from tqdm import tqdm
import argparse
import logging
import warnings
import seaborn as sns
import pandas as pd
from tensorpack import logger
from scipy import stats
from tdgan.model import TGANModel
from preprocess import Preprocessing, DatePostProcess
from utils.config import Evaluator,Db,retrieve
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
logging.basicConfig(level= logging.INFO)
parser=argparse.ArgumentParser(description="Pass the original datapath and synthetic transformation datapath")
parser.add_argument("--original",type=str,required=True)
parser.add_argument("--synthetic",type=str,required=True,help="syn_transform.csv in the generated data folder")
parser.add_argument("--config",default="Config.json",help="Configure file",type=str)
args = vars(parser.parse_args())
from matplotlib import cm
def gen_correlation_image(dataF, save_path= None):
    #rs = np.random.RandomState(0)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(dataF.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels= dataF.columns.tolist()
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    if save_path:
        plt.savefig(save_path,  bbox_inches='tight', pad_inches=0.0)
    #fig.savefig("a.png")
    return fig

# In[ ]:



METAFILE = 'metafile.json'


# In[ ]:


data=pd.read_csv(args["original"])
syn=pd.read_csv(args["synthetic"])

with open(args["config"],"r") as read_file:
    config=json.load(read_file)


preproc = Preprocessing(data, config,args)
data,meta,con_columns = preproc.get_data()

# In[ ]:
o_corr=data.corr()
sns.heatmap(o_corr).get_figure().savefig("correlation/sns_original.png")
#sns.pairplot(o_corr).get_figure().savefig("correlation/sns_pairwise_original.png")

pd.DataFrame(o_corr).to_csv("Original_data_correlation.csv")
#try:
#logger.info(stats.describe(o_corr))
gen_correlation_image(o_corr).savefig("correlation/abalone_original.png")
#except:
 #   print(1)
# In[ ]:

syn=syn.corr()
sns.heatmap(syn).get_figure().savefig("correlation/sns_synthetic.png")
#sns.pairplot(syn).get_figure().savefig("correlation/sns_pairwise_synthetic.png")
pd.DataFrame(syn).to_csv("synthetic_data_correlation.csv")
#logger.info(stats.describe(syn))

gen_correlation_image(syn).savefig("correlation/abalone_synthetic.png")
sns.heatmap(syn).get_figure().savefig("correlation/sns_synthetic.png")


