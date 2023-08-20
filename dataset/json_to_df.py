
#read the path in /data_pool_1/statsbomb_2023/events_and_frames/data
#read the json file and convert it to dataframe
#save the dataframe to /data_pool_1/statsbomb_2023/events_and_frames/data

import json
import pandas as pd
import os
import pdb  
import numpy as np
from tqdm import tqdm
import os
import argparse
#get the path in /data_pool_1/statsbomb_2023/events_and_frames/data
parser = argparse.ArgumentParser()
parser.add_argument('--path_360', type=str, default='/data_pool_1/statsbomb_2023/events_and_frames/data/')
parser.add_argument('--path_events', type=str, default='/data_pool_1/statsbomb_2023/events_and_frames/data/events/') #n=580, dropping matches with no 360 frames, n=579
parser.add_argument('--path_lineups', type=str, default='/data_pool_1/statsbomb_2023/events_and_frames/data/lineups/')
parser.add_argument('--path_matches', type=str, default='/data_pool_1/statsbomb_2023/events_and_frames/data/matches/2/')
parser.add_argument('--out_path', type=str, default='/home/c_yeung/workspace6/python/statsbomb_conference_2023/data')
args = parser.parse_args()
path_360 = args.path_360
path_events= args.path_events
path_lineups= args.path_lineups
path_matches= args.path_matches
out_path= args.out_path

if not os.path.exists(out_path):
    os.makedirs(out_path)

# join 360-frames to events using event_uuid in frames and id in events.

lineup_df_list= []
file_lineups = os.listdir(path_lineups)
for lineups_i_path in file_lineups:
    with open(path_lineups+lineups_i_path) as f:
        data = json.load(f)
    lineups_i = pd.DataFrame(data)
    lineups_i["match_id"]=int(lineups_i_path.replace('.json', ''))
    lineup_df_list.append(lineups_i)
lineup_df = pd.concat(lineup_df_list, ignore_index=True)

matches_df_list = []
file_matches = os.listdir(path_matches)
for matches_i_path in file_matches:
    with open(path_matches+matches_i_path) as f:
        data = json.load(f)
    matches_i = pd.DataFrame(data)
    matches_df_list.append(matches_i)
matches_df = pd.concat(matches_df_list, ignore_index=True)

event_df_list = []
file_events = os.listdir(path_events)
# print(len(file_events))
for file_event_i_path in tqdm(file_events):

    with open(path_events+file_event_i_path) as f:
        data = json.load(f)
    file_events_i = pd.DataFrame(data)
    file_events_i["match_id"]=int(file_event_i_path.replace('.json', ''))

    with open(path_360+file_event_i_path) as f:
        data = json.load(f)
    file_360_i = pd.DataFrame(data)

    if file_360_i.empty:
        print(file_event_i_path, "is empty")
        continue
    else:
        file_360_i.rename(columns={'event_uuid': 'id'}, inplace=True)
        merged_df = pd.merge(file_events_i, file_360_i, on='id', how='left')
        event_df_list.append(merged_df)
# print(len(event_df_list))
event_df = pd.concat(event_df_list, ignore_index=True)



event_df.to_csv(out_path+"/event_df.csv", index=False)
lineup_df.to_csv(out_path+"/lineup_df.csv", index=False)
matches_df.to_csv(out_path+"/matches_df.csv", index=False)