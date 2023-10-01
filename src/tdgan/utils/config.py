#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:37:41 2019

@author: venseven
"""


import sqlite3 as sq
import os
import logging

class model_loader:
    
    def __init__(self,model,model_path,numerical_columns,gpu_in_str,epoch,params,datecolumns=None):
        self.datecolumns=datecolumns
        self.model_path=model_path
        self.model=model(numerical_columns,
                        datecolumns=self.datecolumns,
                        gpu=gpu_in_str,
                        max_epoch=params["max_epoch"],
                        steps_per_epoch=params["steps_per_epoch"],
                        batch_size=params["batch_size"],
                        z_dim=params["z_dim"],
                        noise=params["noise"],
                        l2norm=params["l2norm"],
                        learning_rate=params["learning_rate"],
                        num_gen_rnn=params["num_gen_rnn"],
                        num_gen_feature=params["num_gen_feature"],
                        num_dis_layers=params["num_dis_layers"],
                        num_dis_hidden=params["num_dis_hidden"]
        )
        
        
    def fit(self,data,config,datapath):
        self.model.fit(data,config,datapath)
        self.model.save(self.model_path)
        logging.info(f"Model Saved at {self.model_path}")
        return self.model
    
    def save(self):
        self.model.save(self.model_path)
        #self.model.save(self.model_path,force=True) #use if the model checkpoint already exist
    
    
    
    
    
    def generate(self,model,num_samples):
        new_samples=model.sample(num_samples)
        return new_samples

class Db:
    
    def __init__(self,i=None):

        self.i=i
    
    def check():

            conn=sq.connect("saama.db")
            curs=conn.cursor()
            curs.execute('''CREATE TABLE storage(name VARCHAR NOT NULL,value INT NOT NULL)''')
            curs.execute('INSERT INTO storage (name, value) values("a",{})'.format(0))
            conn.commit()

    def store(self):

        if not os.path.exists("saama.db"):
            
            self.check()
            
        else:
            conn=sq.connect("saama.db")
            curs=conn.cursor()
            curs.execute('INSERT INTO storage (name, value) values("a",{})'.format(self.i))
            conn.commit()
        
        conn.close()

def retrieve():    

        if not os.path.exists("saama.db"):

            Db.check()    

        conn = sq.connect('saama.db')
        curs = conn.cursor()
        curs.execute('SELECT * FROM storage WHERE name="a"')
        data_3 = curs.fetchall()
        conn.commit()
        conn.close()
        return data_3
