import pandas as pd
import numpy as np
import pdb
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--preprocessed_event_df_path","-d", help="event_df.csv path",default="/home/c_yeung/workspace6/python/statsbomb_conference_2023/data/preprocessed_event_df.csv")
parser.add_argument("--match_df_path","-m", help="match_df.csv path",default="/home/c_yeung/workspace6/python/statsbomb_conference_2023/data/matches_df.csv")
parser.add_argument("--out_path","-o", help="output folder path",default="/home/c_yeung/workspace6/python/statsbomb_conference_2023/data/")
parser.add_argument("--sample_df","-s", help="output a sample of event_df.csv",default=True, required=False)
args = parser.parse_args()

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

'''
Train/valid/test split
original study 0.8/0.1/0.1 and time ordered
'''

#load the match df
event_df = pd.read_csv(args.preprocessed_event_df_path)
match_df = pd.read_csv(args.match_df_path)
print("df_loaded")

#drop features
drop_features_list=['zone_dist_0',
'zone_dist_1',
'zone_dist_2',
'zone_dist_3',
'zone_dist_4',
'zone_dist_5',
'zone_dist_6',
'zone_dist_7',
'zone_dist_8',
'zone_dist_9',
'zone_dist_10',
'zone_dist_11',
'zone_dist_12',
'zone_dist_13',
'zone_dist_14',
'zone_dist_15',
'zone_dist_16',
'zone_dist_17',
'zone_dist_18',
'zone_dist_19',
'zone_degree_0',
'zone_degree_1',
'zone_degree_2',
'zone_degree_3',
'zone_degree_4',
'zone_degree_5',
'zone_degree_6',
'zone_degree_7',
'zone_degree_8',
'zone_degree_9',
'zone_degree_10',
'zone_degree_11',
'zone_degree_12',
'zone_degree_13',
'zone_degree_14',
'zone_degree_15',
'zone_degree_16',
'zone_degree_17',
'zone_degree_18',
'zone_degree_19',
'freeze_frame']
event_df = event_df.drop(columns=drop_features_list)

#get the match id in event_df
match_id_list = event_df.match_id.unique()
#drop match_df match id not in match_id_list
match_df = match_df[match_df.match_id.isin(match_id_list)]

#sort the match_df by date
match_df = match_df.sort_values(by=['match_date'])
#use only the 2021/2022 season
analysis_df=match_df[match_df.season=="{'season_id': 235, 'season_name': '2022/2023'}"]
match_df = match_df[match_df.season=="{'season_id': 108, 'season_name': '2021/2022'}"]
#reset the index
match_df = match_df.reset_index(drop=True)
analysis_df = analysis_df.reset_index(drop=True)

#split the match_df.match_id into 2 train/valid/test set and with the 60/20/20 split ratio
#split one use for model that create feature from the freeze frame data (if needed)
#split two use for the NMSTPP model training
match_id_list = match_df.match_id.unique()
match_id_list1=match_id_list[:int(len(match_id_list)/2)]
match_id_list2=match_id_list[int(len(match_id_list)/2):]
analysis_id_list = analysis_df.match_id.unique()

train_set1_split = int(len(match_id_list1)*0.6)
valid_set1_split = int(len(match_id_list1)*0.8)
train_match_id_list1 = match_id_list1[:train_set1_split]
valid_match_id_list1 = match_id_list1[train_set1_split:valid_set1_split]
test_match_id_list1 = match_id_list1[valid_set1_split:]

train_set2_split = int(len(match_id_list2)*0.6)
valid_set2_split = int(len(match_id_list2)*0.8)
train_match_id_list2 = match_id_list2[:train_set2_split]
valid_match_id_list2 = match_id_list2[train_set2_split:valid_set2_split]
test_match_id_list2 = match_id_list2[valid_set2_split:]

train_df1 = event_df[event_df.match_id.isin(train_match_id_list1)]
valid_df1 = event_df[event_df.match_id.isin(valid_match_id_list1)]
test_df1 = event_df[event_df.match_id.isin(test_match_id_list1)]

train_df2 = event_df[event_df.match_id.isin(train_match_id_list2)]
valid_df2 = event_df[event_df.match_id.isin(valid_match_id_list2)]
test_df2 = event_df[event_df.match_id.isin(test_match_id_list2)]

analysis_df = event_df[event_df.match_id.isin(analysis_id_list)]

#print the number of match in each set
print("number of match in train set 1 : ",len(train_df1.match_id.unique()))
print("number of match in valid set 1 : ",len(valid_df1.match_id.unique()))
print("number of match in test set 1 : ",len(test_df1.match_id.unique()))
print("number of match in train set 2 : ",len(train_df2.match_id.unique()))
print("number of match in valid set 2 : ",len(valid_df2.match_id.unique()))
print("number of match in test set 2 : ",len(test_df2.match_id.unique()))

'''
previous study
number of match in train :  73
number of match in valid :  7
number of match in test :  30 + other leagues
'''

required_df=train_df2 #consider the events features for train_df2 only

#print the number for each action type and the overall
print("number of pass : ",len(required_df[required_df.act=="pass"]))
print("number of cross : ",len(required_df[required_df.act=="cross"]))
print("number of dribble : ",len(required_df[required_df.act=="dribble"]))
print("number of shot : ",len(required_df[required_df.act=="shot"]))
print("number of end : ",len(required_df[required_df.act=="end"]))
print("number of total : ",len(required_df[required_df.act.isin(["pass","dribble","shot","end"])]))

#print the max and min of the features
print("max deltaT : ",required_df.deltaT.max())
print("min deltaT : ",required_df.deltaT.min())
print("max zone_s : ",required_df.zone_s.max())
print("min zone_s : ",required_df.zone_s.min())
print("max zone_deltax : ",required_df.zone_deltax.max())
print("min zone_deltax : ",required_df.zone_deltax.min())
print("max zone_deltay : ",required_df.zone_deltay.max())
print("min zone_deltay : ",required_df.zone_deltay.min())
print("max zone_sg : ",required_df.zone_sg.max())
print("min zone_sg : ",required_df.zone_sg.min())
print("max zone_thetag : ",required_df.zone_thetag.max())
print("min zone_thetag : ",required_df.zone_thetag.min())

#save the train/valid/test df
train_df1.to_csv(args.out_path+"train_df1.csv",index=False)
valid_df1.to_csv(args.out_path+"valid_df1.csv",index=False)
test_df1.to_csv(args.out_path+"test_df1.csv",index=False)
train_df2.to_csv(args.out_path+"train_df2.csv",index=False)
train_df2.head(100000).to_csv(args.out_path+"train_df2_reduced.csv",index=False)
valid_df2.to_csv(args.out_path+"valid_df2.csv",index=False)
test_df2.to_csv(args.out_path+"test_df2.csv",index=False)
analysis_df.to_csv(args.out_path+"analysis_df.csv",index=False)

print("train/valid/test df saved")
print("--------done--------")


'''
remained features in event df
'id',
'match_id',
'period',
'timestamp',
'type',
'possession',
'possession_team',
'duration',
'pass',
'location_x',
'location_y',
'pass_type',
'pass_cross',
'act',
'player0_teammate',
'player0_actor',
'player0_keeper',
'player0_location_x',
'player0_location_y',
'player1_teammate',
'player1_actor',
'player1_keeper',
'player1_location_x',
'player1_location_y',
'player2_teammate',
'player2_actor',
'player2_keeper',
'player2_location_x',
'player2_location_y',
'player3_teammate',
'player3_actor',
'player3_keeper',
'player3_location_x',
'player3_location_y',
'player4_teammate',
'player4_actor',
'player4_keeper',
'player4_location_x',
'player4_location_y',
'player5_teammate',
'player5_actor',
'player5_keeper',
'player5_location_x',
'player5_location_y',
'player6_teammate',
'player6_actor',
'player6_keeper',
'player6_location_x',
'player6_location_y',
'player7_teammate',
'player7_actor',
'player7_keeper',
'player7_location_x',
'player7_location_y',
'player8_teammate',
'player8_actor',
'player8_keeper',
'player8_location_x',
'player8_location_y',
'player9_teammate',
'player9_actor',
'player9_keeper',
'player9_location_x',
'player9_location_y',
'player10_teammate',
'player10_actor',
'player10_keeper',
'player10_location_x',
'player10_location_y',
'player11_teammate',
'player11_actor',
'player11_keeper',
'player11_location_x',
'player11_location_y',
'player12_teammate',
'player12_actor',
'player12_keeper',
'player12_location_x',
'player12_location_y',
'player13_teammate',
'player13_actor',
'player13_keeper',
'player13_location_x',
'player13_location_y',
'player14_teammate',
'player14_actor',
'player14_keeper',
'player14_location_x',
'player14_location_y',
'player15_teammate',
'player15_actor',
'player15_keeper',
'player15_location_x',
'player15_location_y',
'player16_teammate',
'player16_actor',
'player16_keeper',
'player16_location_x',
'player16_location_y',
'player17_teammate',
'player17_actor',
'player17_keeper',
'player17_location_x',
'player17_location_y',
'player18_teammate',
'player18_actor',
'player18_keeper',
'player18_location_x',
'player18_location_y',
'player19_teammate',
'player19_actor',
'player19_keeper',
'player19_location_x',
'player19_location_y',
'player20_teammate',
'player20_actor',
'player20_keeper',
'player20_location_x',
'player20_location_y',
'player21_teammate',
'player21_actor',
'player21_keeper',
'player21_location_x',
'player21_location_y',
'seconds',
'deltaT',
'zone',
'zone_s',
'zone_deltax',
'zone_deltay',
'zone_sg',
'zone_thetag',
'zone_x',
'zone_y',


All features in event df
'id',
'match_id',
'period',
'timestamp',
'type',
'possession',
'possession_team',
'duration',
'pass',
'freeze_frame',
'location_x',
'location_y',
'pass_type',
'pass_cross',
'act',
'player0_teammate',
'player0_actor',
'player0_keeper',
'player0_location_x',
'player0_location_y',
'player1_teammate',
'player1_actor',
'player1_keeper',
'player1_location_x',
'player1_location_y',
'player2_teammate',
'player2_actor',
'player2_keeper',
'player2_location_x',
'player2_location_y',
'player3_teammate',
'player3_actor',
'player3_keeper',
'player3_location_x',
'player3_location_y',
'player4_teammate',
'player4_actor',
'player4_keeper',
'player4_location_x',
'player4_location_y',
'player5_teammate',
'player5_actor',
'player5_keeper',
'player5_location_x',
'player5_location_y',
'player6_teammate',
'player6_actor',
'player6_keeper',
'player6_location_x',
'player6_location_y',
'player7_teammate',
'player7_actor',
'player7_keeper',
'player7_location_x',
'player7_location_y',
'player8_teammate',
'player8_actor',
'player8_keeper',
'player8_location_x',
'player8_location_y',
'player9_teammate',
'player9_actor',
'player9_keeper',
'player9_location_x',
'player9_location_y',
'player10_teammate',
'player10_actor',
'player10_keeper',
'player10_location_x',
'player10_location_y',
'player11_teammate',
'player11_actor',
'player11_keeper',
'player11_location_x',
'player11_location_y',
'player12_teammate',
'player12_actor',
'player12_keeper',
'player12_location_x',
'player12_location_y',
'player13_teammate',
'player13_actor',
'player13_keeper',
'player13_location_x',
'player13_location_y',
'player14_teammate',
'player14_actor',
'player14_keeper',
'player14_location_x',
'player14_location_y',
'player15_teammate',
'player15_actor',
'player15_keeper',
'player15_location_x',
'player15_location_y',
'player16_teammate',
'player16_actor',
'player16_keeper',
'player16_location_x',
'player16_location_y',
'player17_teammate',
'player17_actor',
'player17_keeper',
'player17_location_x',
'player17_location_y',
'player18_teammate',
'player18_actor',
'player18_keeper',
'player18_location_x',
'player18_location_y',
'player19_teammate',
'player19_actor',
'player19_keeper',
'player19_location_x',
'player19_location_y',
'player20_teammate',
'player20_actor',
'player20_keeper',
'player20_location_x',
'player20_location_y',
'player21_teammate',
'player21_actor',
'player21_keeper',
'player21_location_x',
'player21_location_y',
'seconds',
'deltaT',
'zone_dist_0',
'zone_dist_1',
'zone_dist_2',
'zone_dist_3',
'zone_dist_4',
'zone_dist_5',
'zone_dist_6',
'zone_dist_7',
'zone_dist_8',
'zone_dist_9',
'zone_dist_10',
'zone_dist_11',
'zone_dist_12',
'zone_dist_13',
'zone_dist_14',
'zone_dist_15',
'zone_dist_16',
'zone_dist_17',
'zone_dist_18',
'zone_dist_19',
'zone_degree_0',
'zone_degree_1',
'zone_degree_2',
'zone_degree_3',
'zone_degree_4',
'zone_degree_5',
'zone_degree_6',
'zone_degree_7',
'zone_degree_8',
'zone_degree_9',
'zone_degree_10',
'zone_degree_11',
'zone_degree_12',
'zone_degree_13',
'zone_degree_14',
'zone_degree_15',
'zone_degree_16',
'zone_degree_17',
'zone_degree_18',
'zone_degree_19',
'zone',
'zone_s',
'zone_deltax',
'zone_deltay',
'zone_sg',
'zone_thetag',
'zone_x',
'zone_y',

'''