# import warnings
# warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import bambi as bmb
import pymc as pm
import arviz as az
import hssm
import os, sys

# 3 CLI arguments
# 1st: bootstrap or fit
# 2nd: n_itr (1 for fit)
# 3rd: structured or unstructured

# #reads in all the NAMES of the data files from the 'data' folder. 
# df_clean_exposure = pd.read_csv('df_clean_exposure.csv')
# df_clean_memory = pd.read_csv('df_clean_memory.csv')

# #Group for each stimulus
# df_clean_exposure_grouped = df_clean_exposure.groupby(['stim chosen', 'participant', 'block']).mean(numeric_only=True).reset_index()

# #Group for each block (across stimulus, used to assign exposure accuracy to 'new' items for recog test) 
# df_clean_exposure_acc = df_clean_exposure_grouped.groupby(['participant', 'block']).mean(numeric_only=True).reset_index()

# #Join first to incorporate exposure accuracy for old stimuli
# df_clean_memory_joined = pd.merge(df_clean_memory, df_clean_exposure_grouped, left_on=['participant', 'test item', 'block'], 
#                                   right_on=['participant', 'stim chosen', 'block'], suffixes=('', '_exposure'), how = 'left')


# #Replace the new stimuli exposure accuracy NaNs with averages for that participant/block.
# for ppt in df_clean_exposure_acc.participant.unique():
#     for b in range(3):
#         df_clean_memory_joined.loc[((df_clean_memory_joined.node_type == 'new') & (df_clean_memory_joined.participant == ppt) & (df_clean_memory_joined.block == b)), 'accuracy_exposure'] = df_clean_exposure_acc.loc[((df_clean_exposure_acc.participant == ppt)&(df_clean_exposure_acc.block == b)), 'accuracy'].values[0]

# The operations above are computed and saved in a separate csv file. Just load it instead.
df_clean_memory_joined = pd.read_csv('df_clean_memory_joined.csv').drop('Unnamed: 0', axis = 1)
#Remove trials that are too quick (<250 miliseconds)
df_clean_memory_joined = df_clean_memory_joined.loc[((df_clean_memory_joined.rt > 0.25))].reset_index(drop=True)

itr = int(sys.argv[2])

CONDITION = sys.argv[3]


#Parametrize the DDM. The version below is the winning model
def def_model(boot=True):
    #If bootstrapping, relabel the node types
    if boot:
        for ppt in df_clean_memory_joined.participant.unique():
            # Randomly select 6 boundary nodes
            boundary_ids = np.random.choice(np.arange(15), size=6, replace=False)
            nonboundary_ids = np.delete(np.arange(15), boundary_ids)

            # reassign labels to randomly selected boundary/nonboundary nodes
            df_clean_memory_joined.loc[((df_clean_memory_joined['path id'].isin(boundary_ids)) & (df_clean_memory_joined['participant'] == ppt)), 'node_type'] = 'boundary'
            df_clean_memory_joined.loc[((df_clean_memory_joined['path id'].isin(nonboundary_ids)) & (df_clean_memory_joined['participant'] == ppt)), 'node_type'] = 'nonboundary'

    
    #Parameterize model.
    ddm_model = hssm.HSSM(noncentered = True, prior_settings = 'safe',  
                          data = df_clean_memory_joined.loc[df_clean_memory_joined.condition == CONDITION], #Specify 'structured' or 'unstructured' condition participants here. Could be cleaner. 
                          include=[
                                    {
                                        "name": "v",
                                        "formula": "v ~ 0 + accuracy_exposure:node_type + node_type + (node_type|participant_id)",
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
    return ddm_model

#Declare model and Sample

if sys.argv[1] == 'bootstrap':
    ddm_model = def_model()
    ddm_model.sample()
    ddm_model.summary().to_csv(f'model_results/bootstrap/{CONDITION}/{itr}.csv')
    print(f'Bootstrap iteration done: {n_itr+1}')
else:
    ddm_model = def_model(boot=False)
    ddm_model.sample()
    az.to_netcdf(ddm_model._inference_obj, f'model_results/posteriors/hssm_posteriors/{CONDITION}_v~nt_ntaccexp-|ppt_a~block|ppt_z~block|ppt_t~|ppt.nc')
    

