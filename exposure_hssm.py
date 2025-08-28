import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast, os, sys
import bambi as bmb
import pymc as pm
import arviz as az
import scipy.stats as stat
from collections import Counter
import itertools
import hssm
from hssm.likelihoods import DDM
import nutpie

df_clean_exposure = pd.read_csv('df_clean_exposure.csv')

df_clean_exposure['response'] = df_clean_exposure['accuracy']
df_clean_exposure.loc[df_clean_exposure['response'] == 0, 'response'] = -1

df_clean_exposure['node_type'] = df_clean_exposure['node type']
df_clean_exposure['log_trials'] = np.log(df_clean_exposure['trials']+1)
cond = sys.argv[1]

ddm_model = hssm.HSSM(noncentered = True, prior_settings = 'safe',  
                      data = df_clean_exposure.loc[df_clean_exposure.condition == cond], #Specify 'structured' or 'unstructured' condition participants here. Could be cleaner. 
                      include=[
                                {
                                    "name": "v",
                                    "formula": "v ~ 0 + C(node_type):log_trials + (1|participant)",
                                },
                          # {
                          #     "name": "z",
                          #     "formula": "z ~ C(block) + (1|participant_id)"
                          # },

                          # {
                          #     "name": "a",
                          #     "formula": "a ~ C(block) + (1|participant_id)"

                          # },
                          
                          # {
                          #     "name": "t",
                          #     "formula": "t ~ (1|participant_id)"
                          # },
                          
                          
                              ],
                     )

#Sample
ddm_model_samples = ddm_model.sample(idata_kwargs=dict(log_likelihood=True))

#Save
az.to_netcdf(ddm_model._inference_obj, f'model_results/posteriors/{cond}_exposure_v~nt:logtrials_1|ppt.nc')
