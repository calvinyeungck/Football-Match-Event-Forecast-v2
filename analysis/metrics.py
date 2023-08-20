import pandas as pd
import numpy as np
import argparse
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--prediction_df_path', type=str, default='/home/c_yeung/workspace6/python/statsbomb_conference_2023/result/final/prediction/prediction_df.csv')
parser.add_argument('--out_path', type=str, default='/home/c_yeung/workspace6/python/statsbomb_conference_2023/script/analysis')
args = parser.parse_args()

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

# Load prediction_df
prediction_df = pd.read_csv(args.prediction_df_path)
print("data loaded")
print("start processing")
time_columns='pred_deltaT'
zone_columns=['pred_zone1','pred_zone2','pred_zone3','pred_zone4','pred_zone5','pred_zone6','pred_zone7','pred_zone8','pred_zone9','pred_zone10','pred_zone11','pred_zone12',
 'pred_zone13','pred_zone14','pred_zone15','pred_zone16','pred_zone17','pred_zone18','pred_zone19','pred_zone20']
action_columns=['pred_action1','pred_action2','pred_action3','pred_action4','pred_action5']

# Calculate the minimum value across all action columns
min_value = prediction_df[action_columns].min().min()

# Add the absolute min value to each element in the DataFrame
normalized_df = prediction_df[action_columns].apply(lambda x: x + abs(min_value))

# Normalize the values so that they sum up to 1 for each row
normalized_df = normalized_df.div(normalized_df.sum(axis=1), axis=0)

# Replace the original action columns with the normalized probabilities
prediction_df[action_columns] = normalized_df

# Calculate the minimum value across all zone columns
min_value = prediction_df[zone_columns].min().min()

# Add the absolute min value to each element in the DataFrame
normalized_df = prediction_df[zone_columns].apply(lambda x: x + abs(min_value))

# Normalize the values so that they sum up to 1 for each row
normalized_df = normalized_df.div(normalized_df.sum(axis=1), axis=0)

# Replace the original zone columns with the normalized probabilities
prediction_df[zone_columns] = normalized_df

#Caculate HAS
prediction_df["HAS_zone"]=(prediction_df["pred_zone4"]+prediction_df["pred_zone5"]+prediction_df["pred_zone10"]+prediction_df["pred_zone11"]+prediction_df["pred_zone14"]+prediction_df["pred_zone16"]+prediction_df["pred_zone18"])*5+(prediction_df["pred_zone6"]+prediction_df["pred_zone12"]+prediction_df["pred_zone20"])*10
prediction_df["HAS_action"]=(prediction_df["pred_action1"]+prediction_df["pred_action2"])*5+(prediction_df["pred_action4"]+prediction_df["pred_action5"])*10
prediction_df["HAS_t"]= prediction_df[time_columns].apply(lambda x: x if x>1 else 1)

def kernal(x):
    return np.exp(-0.3*(x-1))

possession_metrics_list = []
# Calculate metrics HPUS
#group by match_id
prediction_df_grouped = prediction_df.groupby('match_id')
#loop through each match
for match_id, group in prediction_df_grouped:
    #group by possession
    group_possession = group.groupby('possession')
    #loop through each possession
    for possession, group_possession in group_possession:
        #calculate HPUS
        # pdb.set_trace()
        if group_possession.HAS_zone.isna().sum()>0 and group_possession.HAS_action.isna().sum()>0:
            continue
        HPUS=0
        event_count=1
        shot_or_cross=False
        n=len(group_possession)
        for index, row in group_possession.iterrows():
            HPUS+=kernal(n+1-event_count)*((row['HAS_zone']*row['HAS_action'])**0.5)/row['HAS_t']
            event_count+=1
            if row['act']=='shot' or row['act']=='cross':
                shot_or_cross=True
        if len(group_possession.possession_team.unique())>1:
            print("Error: more than one team in possession")
            pdb.set_trace()
        possession_dict = {'match_id': match_id, 'possession': possession,'possession_team':str(group_possession.possession_team.unique()[0]), 'HPUS': HPUS,'attack':shot_or_cross}
        possession_metrics_list.append(possession_dict)

# create possession_metrics_df
possession_metrics_df = pd.DataFrame(possession_metrics_list)
#save possession_metrics_df
possession_metrics_df.to_csv(os.path.join(args.out_path,'possession_metrics_df.csv'),index=False)

print("end processing")





'''
centroid_x=[ 8.5 , 25.25, 41.75, 58.25, 74.75, 91.5,8.5 , 25.25, 41.75, 58.25, 74.75, 
   91.5,33.5, 66.5,33.5, 66.5,33.5, 66.5,8.5,91.5] #from 0 to 100
centroid_y=[89.45, 89.45, 89.45, 89.45, 89.45, 89.45,10.55, 10.55, 10.55, 10.55, 10.55, 10.55,
   71.05, 71.05,50., 50.,28.95, 28.95, 50.,50.] #from 0 to 100

0: 1,2,3,7,8,9,13,15,17,19
5: 4,5,10,11,14,16,18
10: 6,12,20

act_encode_dict = {'pass': 0, 'dribble': 1, 'end': 2, 'shot': 3, 'cross': 4} #+1
0: 3
5: 1,2
10: 4,5
'''