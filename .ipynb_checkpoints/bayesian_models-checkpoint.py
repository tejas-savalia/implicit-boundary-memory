# Import the required packages
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
from utils import *
import os, sys

dv = sys.argv[1] #command line argument can be 'accuracy' or 'rt'

if dv == 'accuracy':
    family = 'bernoulli'
else:
    family = 'gaussian'
#Read exposure data. Rename some columns for bambi requirements
df_clean_exposure = pd.read_csv('df_clean_exposure.csv')
df_clean_exposure['node_type'] = df_clean_exposure['node type']
df_clean_exposure['log_trials'] = np.log(df_clean_exposure['trials']+1)

#Model fit and save
bmb_exposure_accuracy_model = bmb.Model(f'{dv} ~ log_trials*condition*node_type + (1|participant)', data = df_clean_exposure, family = family)
samples = bmb_exposure_accuracy_model.fit(nuts_sampler = 'nutpie')
az.to_netcdf(samples, f'model_results/exposure_nt_{dv}.nc')