#!/usr/bin/env python
# coding: utf-8

# In[12]:



from warnings import filterwarnings
filterwarnings("ignore")
from tdgan.model import TGANModel
#from config import Evaluator,Db,retrieve
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm
from tensorpack import logger
import os
import shutil
import math
import dateutil
import math

from tdgan.data import Preprocessing, DatePostProcess
from utils.config import Db,retrieve

data_db=retrieve()
count=data_db[len(data_db)-1][1]
count +=1
database=Db(count)
database.store()


# In[7]:



numsamples=2000


# In[ ]:



parser=argparse.ArgumentParser(description="Pass the Configure file and the datapath")
parser.add_argument("--modelpath",type=str,default="models/saama123_test.pkl")
parser.add_argument("--numsamples",type=int,required=True)
parser.add_argument("--configure",type=str,default="caches/Config_copy.json")
parser.add_argument("--metafile",type=str,default="caches/metafile.json")


args = vars(parser.parse_args())

        
with open(args["configure"],"r") as read_file:
    config=json.load(read_file)

# In[8]:

print("done")
model=TGANModel.load(args["modelpath"])
print("done 2")
new_samples=model.sample(args["numsamples"])


# In[14]:


METAFILE=args["metafile"]


# In[10]:


new_samples.to_csv("generated_data/syn_transform-{}.csv".format(count))


# In[17]:


post_proc = DatePostProcess(new_samples, METAFILE, args["configure"])


# In[18]:
logger.info(config["DATE_COLS"])
if config["DATE_COLS"]:
    post_proc.post_process_dates()
    final_df = post_proc.get_data()
else:
    print(True)
    final_df=post_proc.post_process_dates(False)

# In[24]:



final_df.to_csv("generated_data/generated_data-{}.csv".format(count))


# In[23]:


#final_df.drop(final_df.columns[0],axis=1,inplace=True) #dropiing study number column add if you want


# In[ ]:




