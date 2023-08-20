import pandas as pd
import numpy as np
import argparse
import pdb
import os
import ast
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--sequences_action_df', type=str, default='/home/c_yeung/workspace6/python/statsbomb_conference_2023/script/analysis/sequences_encoded_action.csv')
parser.add_argument('--sequences_zone_df', type=str, default='/home/c_yeung/workspace6/python/statsbomb_conference_2023/script/analysis/sequences_encoded_zone2.csv')
parser.add_argument('--metrics_df', type=str, default='/home/c_yeung/workspace6/python/statsbomb_conference_2023/script/analysis/possession_metrics_df.csv')
parser.add_argument('--out_path', type=str, default='/home/c_yeung/workspace6/python/statsbomb_conference_2023/script/analysis')
args = parser.parse_args()

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

#read in the data
sequences_action_df = pd.read_csv(args.sequences_action_df)
sequences_zone_df = pd.read_csv(args.sequences_zone_df)
metrics_df = pd.read_csv(args.metrics_df)

#convert string to list
sequences_action_df['act_encoded'] = sequences_action_df['act_encoded'].apply(ast.literal_eval)
sequences_zone_df['zone_encoded'] = sequences_zone_df['zone_encoded'].apply(ast.literal_eval)

#flatten the list
sequences_action_df['flattened_act'] = sequences_action_df['act_encoded'].apply(lambda x: [item for sublist in x for item in sublist])
sequences_zone_df['flattened_zone'] = sequences_zone_df['zone_encoded'].apply(lambda x: [item for sublist in x for item in sublist])

'''
# For action sequences
"['1', '1', '1', '1', '1', '1', '1']",3878    #pass based
"['2', '2', '2', '2', '2', '2']",4088    #dribble based
"['1', '2', '1', '2', '1', '2', '1', '2', '1', '2']",3947 
and 
"['2', '1', '2', '1', '2', '1', '2', '1', '2', '1']",3887 # pass follows dribble or the opposite
# For zone sequences
"['1', '1', '1', '1', '1']",3204 # opponent-wing based *less then 0.3 support, minimum 5 action 
"['3', '3', '3', '3', '3']",5510 # own-half based
# Other
'''
def has_ordered_items(input_list, target_items):
    list_idx = 0  # Index to track progress in the input_list
    
    for item in input_list:
        if item == target_items[list_idx]:
            list_idx += 1
            
            if list_idx == len(target_items):
                return True  # All target items found in order
        
    return False

pattern_list = []
#loop through each possesion in the df row
for i in tqdm(range(len(sequences_action_df))):
    # pdb.set_trace()
    if len(sequences_action_df['flattened_act'][i])<6:
        pass_based = 0
        dirbble_based = 0
        pass_and_dirbble_based = 0
    else:
        input_list = sequences_action_df['flattened_act'][i]
        if has_ordered_items(input_list, [1, 1, 1, 1, 1, 1, 1])==True:
            pass_based = 1
        else:
            pass_based = 0
        if has_ordered_items(input_list, [2, 2, 2, 2, 2, 2])==True:
            dirbble_based = 1
        else:
            dirbble_based = 0
        if has_ordered_items(input_list, [2, 1, 2, 1, 2, 1, 2, 1, 2, 1])==True:
            pass_and_dirbble_based = 1
        elif has_ordered_items(input_list, [1, 2, 1, 2, 1, 2, 1, 2, 1, 2])==True:
            pass_and_dirbble_based = 1
        else:
            pass_and_dirbble_based = 0

    if len(sequences_zone_df['flattened_zone'][i])<5:
        wing_based = 0
        own_half_based = 0
    else:
        input_list = sequences_zone_df['flattened_zone'][i]
        if has_ordered_items(input_list, [1, 1, 1, 1, 1])==True:
            wing_based = 1
        else:
            wing_based = 0

        if has_ordered_items(input_list, [3, 3, 3, 3, 3])==True:
            own_half_based = 1
        else:
            own_half_based = 0


    if pass_based==0 and dirbble_based==0 and pass_and_dirbble_based==0 and wing_based==0 and own_half_based==0:
        other = 1
    else:
        other = 0
    seq_len = len(sequences_action_df['flattened_act'][i])
    pattern_list.append([pass_based, dirbble_based, pass_and_dirbble_based, wing_based, own_half_based, other,seq_len])

#convert to df
pattern_df = pd.DataFrame(pattern_list, columns=['pass_based', 'dirbble_based', 'pass_and_dirbble_based', 'wing_based', 'own_half_based', 'other','seq_len'])

#concat with the original df
sequences_action_df = pd.concat([sequences_action_df[["match_id","possession","possession_team"]], pattern_df,sequences_action_df['flattened_act'],sequences_zone_df['flattened_zone']], axis=1)
#merge with metrics
sequences_action_df = pd.merge(sequences_action_df, metrics_df, on=['match_id','possession','possession_team'], how='left')
#save_df
sequences_action_df.to_csv(os.path.join(args.out_path, 'sequences_pattern.csv'), index=False)
# pdb.set_trace()
