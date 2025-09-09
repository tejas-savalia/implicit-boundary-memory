import warnings
warnings.filterwarnings('ignore')
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
import nutpie
import os, sys

ppt_id = int(sys.argv[1])
df_structured_filtered = pd.read_csv('structured_filtered.csv')
all_ppts = df_structured_filtered.participant_id.unique()
ppt = all_ppts[ppt_id]

filtered = df_structured_filtered.loc[((df_structured_filtered.participant_id == ppt)), ['rt', 'response', 'node_type']].reset_index(drop=True)
ddm_model = hssm.HSSM(noncentered = True, prior_settings = 'safe', model = 'full_ddm',
                      data = filtered,#, ['participant_id', 'response', 'rt', "node_type", "block", "accuracy_exposure"]],  
                      include=[
                                {
                                    "name": "v",
                                    "formula": "v ~ 0 + C(node_type)",
                                },
                          
                              ],
                     )
samples = ddm_model.sample()
ddm_model_summary = az.summary(samples)
ddm_model_summary.to_csv(f'model_results/indiv_subs/{ppt}.csv')
print('Participant done: ', ppt)