# import warnings
# warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast, os
import bambi as bmb
import pymc as pm
import arviz as az
import scipy.stats as stat
from collections import Counter
import itertools
import hssm
from utils import *
from hssm.likelihoods import DDM
import nutpie

# hssm.set_floatX("float32")

#reads in all the NAMES of the data files from the 'data' folder. 
df_clean_exposure = pd.read_csv('df_clean_exposure.csv')
df_clean_memory = pd.read_csv('df_clean_memory.csv')

#Group for each stimulus
df_clean_exposure_grouped = df_clean_exposure.groupby(['stim chosen', 'participant', 'block']).mean(numeric_only=True).reset_index()

#Group for each block (across stimulus, used to assign exposure accuracy to 'new' items for recog test) 
df_clean_exposure_acc = df_clean_exposure_grouped.groupby(['participant', 'block']).mean(numeric_only=True).reset_index()

#Join first to incorporate exposure accuracy for old stimuli
df_clean_memory_joined = pd.merge(df_clean_memory, df_clean_exposure_grouped, left_on=['participant', 'test item', 'block'], 
                                  right_on=['participant', 'stim chosen', 'block'], suffixes=('', '_exposure'), how = 'left')


#Replace the new stimuli exposure accuracy NaNs with averages for that participant/block.
for ppt in df_clean_exposure_acc.participant.unique():
    for b in range(3):
        df_clean_memory_joined.loc[((df_clean_memory_joined.node_type == 'new') & (df_clean_memory_joined.participant == ppt) & (df_clean_memory_joined.block == b)), 'accuracy_exposure'] = df_clean_exposure_acc.loc[((df_clean_exposure_acc.participant == ppt)&(df_clean_exposure_acc.block == b)), 'accuracy'].values[0]

#Remove trials that are too quick (<250 miliseconds)
df_clean_memory_joined = df_clean_memory_joined.loc[((df_clean_memory_joined.rt > 0.25))].reset_index(drop=True)


    

#Parametrize the DDM. The version below is the winning model

ddm_model = hssm.HSSM(noncentered = True, prior_settings = 'safe',  
                      data = df_clean_memory_joined.loc[df_clean_memory_joined.condition == 'unstructured'], #Specify 'structured' or 'unstructured' condition participants here. Could be cleaner. 
                      include=[
                                {
                                    "name": "v",
                                    "formula": "v ~ 0 + accuracy_exposure + C(node_type) + C(block) + (C(node_type)|participant_id)",
                                },
                          {
                              "name": "z",
                              "formula": "z ~ C(block) + (1|participant_id)"
                          },

                          {
                              "name": "a",
                              "formula": "a ~ C(block) + (1|participant_id)"

                          },
                          
                          {
                              "name": "t",
                              "formula": "t ~ (1|participant_id)"
                          },
                          
                          
                              ],
                     )

#Sample
ddm_model_samples = ddm_model.sample(idata_kwargs=dict(log_likelihood=True))

#Save
az.to_netcdf(ddm_model._inference_obj, f'hssm_results/unstructured_v~acc_exp-block-nodetype-nodetype|ppt_a~block-|ppt_z~block-|ppt_t~|ppt.nc')


