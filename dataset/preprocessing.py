import numpy as np
import pandas as pd
import pdb
import time
import argparse
import ast 
from tqdm import tqdm
import os

#preprocessing the event_df.csv as in (Yeung et al. 2023)

parser = argparse.ArgumentParser()
parser.add_argument("--event_df_path","-d", help="event_df.csv path",default="/home/c_yeung/workspace6/python/statsbomb_conference_2023/data/event_df.csv")
parser.add_argument("--out_path","-o", help="output folder path",default="/home/c_yeung/workspace6/python/statsbomb_conference_2023/data/")
parser.add_argument("--sample_df","-s", help="output a sample of event_df.csv",default=True, required=False)
parser.add_argument("--testing","-t", help="test the code on reduced size df",default=False, required=False)
args = parser.parse_args()
#path for the event_df.csv and out_path for the output
data_path = args.event_df_path
out_path = args.out_path
if not os.path.exists(out_path):
    os.makedirs(out_path)

#load the event_df.csv
df = pd.read_csv(data_path) #n=2022200, required 2min to load
print("df loaded")

if args.testing:
    df=df.head(10000)


# filter out unwanted columns
required_col=["id",                     
"match_id",                  
"period",                  
"timestamp",              
"type",                    
"possession",              
"possession_team",                                     
"duration",                              
"location",
"pass",                       
"freeze_frame"]           
# "injury_stoppage", "tactics",   dropped since all nan after group and filter action type

required_df=df[required_col]

#drop matches with own goal
print("# of matches before dropping matches with OG : ",len(required_df.match_id.unique())) #n=579
drop_match_list1=required_df[required_df["type"] == "{'id': 25, 'name': 'Own Goal For'}"].match_id.unique()
drop_match_list2=required_df[required_df["type"] == "{'id': 20, 'name': 'Own Goal Against'}"].match_id.unique()
drop_match_list=np.concatenate((drop_match_list1,drop_match_list2),axis=0)
required_df=required_df[~required_df.match_id.isin(drop_match_list)]
print("# of matches after dropping matches with OG : ",len(required_df.match_id.unique())) #n=519

#process coordinate data
#convert the location column to x,y
required_df["location"] = required_df["location"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)     #convert string representation of lists to actual lists
required_df["location_x"] = required_df["location"].apply(lambda coords: coords[0] if isinstance(coords, list) else None)
required_df["location_y"] = required_df["location"].apply(lambda coords: coords[1] if isinstance(coords, list) else None)
required_df.drop(columns=["location"],inplace=True) #drop the location column
print("location_x and location_y created")

#group and filter action type
drop_action_list=[  "{'id': 35, 'name': 'Starting XI'}",
                    "{'id': 42, 'name': 'Ball Receipt*'}",
                    "{'id': 17, 'name': 'Pressure'}",
                    "{'id': 22, 'name': 'Foul Committed'}",
                    "{'id': 21, 'name': 'Foul Won'}", 
                    "{'id': 2, 'name': 'Ball Recovery'}",
                    "{'id': 6, 'name': 'Block'}",
                    "{'id': 10, 'name': 'Interception'}", 
                    "{'id': 23, 'name': 'Goal Keeper'}",
                    "{'id': 39, 'name': 'Dribbled Past'}",
                    "{'id': 38, 'name': 'Miscontrol'}",
                    "{'id': 3, 'name': 'Dispossessed'}", 
                    "{'id': 40, 'name': 'Injury Stoppage'}",
                    "{'id': 19, 'name': 'Substitution'}",
                    "{'id': 36, 'name': 'Tactical Shift'}",
                    "{'id': 27, 'name': 'Player Off'}",
                    "{'id': 26, 'name': 'Player On'}", 
                    "{'id': 28, 'name': 'Shield'}",
                    "{'id': 24, 'name': 'Bad Behaviour'}",
                    "{'id': 8, 'name': 'Offside'}",
                    "{'id': 34, 'name': 'Half End'}",       
                    "{'id': 25, 'name': 'Own Goal For'}",
                    "{'id': 41, 'name': 'Referee Ball-Drop'}"]
print("# of row before droping unwanted action type : ",len(required_df)) #n=1814942
required_df=required_df[~required_df.type.isin(drop_action_list)] #drop the action type in drop_action_list

## create column for ground_action_type "act"
pass_action_list=[  "{'id': 18, 'name': 'Half Start'}",       
                    "{'id': 9, 'name': 'Clearance'}"]

dribble_action_list=[   "{'id': 43, 'name': 'Carry'}", 
                        "{'id': 4, 'name': 'Duel'}",
                        "{'id': 14, 'name': 'Dribble'}",
                        "{'id': 33, 'name': '50/50'}"]

shot_action_list=[  "{'id': 16, 'name': 'Shot'}",
                    "{'id': 37, 'name': 'Error'}"]

"{'id': 30, 'name': 'Pass'}"

def convert_to_dict(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None

required_df["pass"]=required_df["pass"].apply(convert_to_dict)
required_df["pass_type"]=required_df["pass"].apply(lambda x: str(x["type"]) if isinstance(x, dict) and "type" in x else None)
required_df["pass_cross"]=required_df["pass"].apply(lambda x: x["cross"] if isinstance(x, dict) and "cross" in x else None)
required_df.loc[required_df.type.isin(pass_action_list),"act"] = "pass"   
required_df.loc[required_df.type.isin(dribble_action_list),"act"] = "dribble"   
required_df.loc[required_df.type.isin(shot_action_list),"act"] = "shot"   
required_df.loc[(required_df.type=="{'id': 30, 'name': 'Pass'}")&(required_df.pass_type!="{'id': 61, 'name': 'Corner'}")&(required_df.pass_cross!=True),"act"] = "pass"   
required_df.loc[(required_df.type=="{'id': 30, 'name': 'Pass'}")&((required_df.pass_type=="{'id': 61, 'name': 'Corner'}")|(required_df.pass_cross==True)),"act"] = "cross" 


#edit the features for "{'id': 18, 'name': 'Half Start'}"
##remove the first half start
drop_row_1=[]
required_df=required_df.reset_index(drop=True)
for match_id in required_df.match_id.unique():
    match_df = required_df[required_df.match_id==match_id]
    poss_df = match_df[match_df.type=="{'id': 18, 'name': 'Half Start'}"]
    for period in poss_df.period.unique():
        period_df = poss_df[poss_df.period==period]
        if len(period_df.index)<2:
            #edit row 1 in required_df
            required_df.loc[period_df.index[0],"possession"] = required_df.loc[period_df.index[1],"possession"]+1
            required_df.loc[period_df.index[0],"possession_team"] = required_df.loc[period_df.index[1]+1,"possession_team"]
            required_df.loc[period_df.index[0],"location_x"] = 60
            required_df.loc[period_df.index[0],"location_y"] = 40
            if args.testing: #should have 2 half start for each half
                pdb.set_trace()
        else:
            drop_row_1.append(period_df.index[0])
            #edit row 2 in required_df
            required_df.loc[period_df.index[1],"possession"] = required_df.loc[period_df.index[1],"possession"]+1
            required_df.loc[period_df.index[1],"possession_team"] = required_df.loc[period_df.index[1]+1,"possession_team"]
            required_df.loc[period_df.index[1],"location_x"] = 60
            required_df.loc[period_df.index[1],"location_y"] = 40
#drop all index in drop_row_1
required_df=required_df.drop(index=drop_row_1)
print("# of row after droping unwanted action type : ",len(required_df)) #n=

#remove event after 45 min for each half
    #turn timestamp from str into time(hh:mm:ss.msmsms)
required_df["timestamp"] = pd.to_datetime(required_df["timestamp"], format="%H:%M:%S.%f").dt.time
    #drop after 45 minute
required_df = required_df[~(required_df['timestamp'] > pd.to_datetime("00:45:00.000", format='%H:%M:%S.%f').time())]
print("# of row after droping after 45 and 90 minute for each half : ",len(required_df)) #n= 928560

#add possession end 
##reset required_df index
required_df=required_df.reset_index(drop=True)
add_poss_end_df = []
for match_id in tqdm(required_df.match_id.unique()):
    possession_count = 1
    for poss_id in required_df[required_df.match_id==match_id].possession.unique():
        for period_id in required_df[required_df.match_id==match_id].period.unique():
            poss_df = required_df[(required_df.match_id==match_id) & (required_df.possession==poss_id)& (required_df.period==period_id)]
            if len(poss_df.period.unique())==1:
                poss_df.loc[:,"possession"] = possession_count
                add_poss_end_df.append(poss_df)
                row=poss_df.iloc[-1]
                possession_end_row = pd.DataFrame(columns=required_df.columns,index=range(1))
                possession_end_row["act"]="end"
                possession_end_row["id"]=str(row["match_id"])+"-"+str(row["period"])+"-"+str(row["possession"])
                possession_end_row["match_id"]=row["match_id"]
                possession_end_row["period"]=row["period"]
                possession_end_row["timestamp"]=row["timestamp"]
                possession_end_row["possession"]=row["possession"]
                possession_end_row["possession_team"]=row["possession_team"]
                possession_end_row["duration"]=0
                possession_end_row["location_x"]=row["location_x"]
                possession_end_row["location_y"]=row["location_y"]
                possession_count += 1
                add_poss_end_df.append(possession_end_row)

#concate the add_poss_end_df_list
add_poss_end_df = pd.concat(add_poss_end_df, ignore_index=True)
#turn the add_poss_end_df_list into one df
required_df = add_poss_end_df
print("# of row after adding poss_end : ",len(required_df)) #n= 928560


#extract the freeze frame data
required_df=required_df.reset_index(drop=True)
required_df["freeze_frame"]=required_df["freeze_frame"].apply(convert_to_dict) #list of dict
columns_dict = {}
for i in range(22):
    columns_dict[f"player{i}_teammate"] = None
    columns_dict[f"player{i}_actor"] = None
    columns_dict[f"player{i}_keeper"] = None
    columns_dict[f"player{i}_location_x"] = None
    columns_dict[f"player{i}_location_y"] = None

# Create a temporary DataFrame with the new columns
new_columns_df = pd.DataFrame(columns_dict, index=required_df.index)
# Concatenate the temporary DataFrame with the existing DataFrame
required_df = pd.concat([required_df, new_columns_df], axis=1)


def process_freeze_frame(row):
    freezeframe_data = row['freeze_frame']
    if freezeframe_data is None:
        return row

    num_players = len(freezeframe_data)
    for i in range(num_players):
        player_data = freezeframe_data[i]
        row[f"player{i}_teammate"] = player_data.get("teammate")
        row[f"player{i}_actor"] = player_data.get("actor") #the player performing the action
        row[f"player{i}_keeper"] = player_data.get("keeper")
        row[f"player{i}_location_x"] = player_data.get("location")[0]
        row[f"player{i}_location_y"] = player_data.get("location")[1]

    return row

required_df = required_df.apply(process_freeze_frame, axis=1)

print("freeze frame data extracted")

#convert the xy coordinate to pitch coordinate
required_df["location_x"] = required_df["location_x"]*105/120
required_df["location_y"] = required_df["location_y"]*68/80
for i in range(22):
    #if x>120 or y>80, set to 120 or 80 and if x<0 or y<0, set to 0
    required_df.loc[required_df[f"player{i}_location_x"]>120,f"player{i}_location_x"] = 120
    required_df.loc[required_df[f"player{i}_location_y"]>80,f"player{i}_location_y"] = 80
    required_df.loc[required_df[f"player{i}_location_x"]<0,f"player{i}_location_x"] = 0
    required_df.loc[required_df[f"player{i}_location_y"]<0,f"player{i}_location_y"] = 0
    required_df[f"player{i}_location_x"] = required_df[f"player{i}_location_x"]*105/120
    required_df[f"player{i}_location_y"] = required_df[f"player{i}_location_y"]*68/80
print("location_x and location_y converted to pitch coordinate")

# #calculate the distance to goal
# goal_x = 120*105/120
# goal_y = 40*68/80
# required_df["Dist2Goal"] = np.sqrt((required_df["location_x"]-goal_x)**2+(required_df["location_y"]-goal_y)**2)
# #calculate the angle to goal
# required_df["Ang2Goal"] = np.abs(np.arctan2((required_df["location_y"]-goal_y),(required_df["location_x"]-goal_x)))
# print("distance and angle to goal calculated")

#Start second half time at 60 minutes and convert the time to second

required_df['seconds'] = required_df['timestamp'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1000000)
required_df.loc[required_df.period==2,"seconds"] = required_df.loc[required_df.period==2,"seconds"]+3600
print("second half time start at 60 minutes")

#calculate interevent time
required_df["deltaT"] = required_df["seconds"].diff()

#limit the maximum delatT to 60s
required_df.loc[required_df.deltaT>60,"deltaT"] = 60

#zone the pitch (create fuzzy c mean cluster for the coordinate, m=2 (how fuzzy))
centroid_x=[ 8.5 , 25.25, 41.75, 58.25, 74.75, 91.5,8.5 , 25.25, 41.75, 58.25, 74.75, 
   91.5,33.5, 66.5,33.5, 66.5,33.5, 66.5,8.5,91.5] #from 0 to 100
centroid_y=[89.45, 89.45, 89.45, 89.45, 89.45, 89.45,10.55, 10.55, 10.55, 10.55, 10.55, 10.55,
   71.05, 71.05,50., 50.,28.95, 28.95, 50.,50.] #from 0 to 100
# need to multiply by 1.05 (x) and 0.68 (y) to get the pitch coordinate
centroid_x=[x*105/100 for x in centroid_x]
centroid_y=[y*68/100 for y in centroid_y]

#cal the diff to the centroid
required_df=required_df.reset_index(drop=True)
diff_dict = {}
for i in range(20):
    dist=((required_df["location_x"]-centroid_x[i])**2+(required_df["location_y"]-centroid_y[i])**2)**0.5
    name="zone_dist_"+str(i)
    diff_dict[name]=dist 
new_columns_df = pd.DataFrame(diff_dict, index=required_df.index)
required_df = pd.concat([required_df, new_columns_df], axis=1)

#get the cluster degree to zone centroid
degree_dict = {}
for i in range(20):
    degree=0
    for j in range(20):
        degree+=(required_df["zone_dist_"+str(i)]/required_df["zone_dist_"+str(j)])**2
    degree=1/degree
    name="zone_degree_"+str(i)
    degree_dict[name]=degree
new_columns_df = pd.DataFrame(degree_dict, index=required_df.index)
required_df = pd.concat([required_df, new_columns_df], axis=1)

print("zone centroid calculated")

# create the hard label for Juego de Posición (position game)

required_df["zone"]=required_df[['zone_degree_0','zone_degree_1',
'zone_degree_2', 'zone_degree_3', 'zone_degree_4', 'zone_degree_5',
'zone_degree_6', 'zone_degree_7', 'zone_degree_8', 'zone_degree_9',
'zone_degree_10', 'zone_degree_11', 'zone_degree_12', 'zone_degree_13',
'zone_degree_14', 'zone_degree_15', 'zone_degree_16', 'zone_degree_17',
'zone_degree_18', 'zone_degree_19']].idxmax(axis=1).str.replace("zone_degree_", '')
required_df["zone"]=pd.to_numeric(required_df["zone"])

print("zone label created")

# create features
'''
 'zone_s', distance since previous event
 'zone_deltay', change in zone distance in x 
 'zone_deltax', change in zone distance in y
 'zone_sg',  distance to the center of opponent goal from the zone
 'zone_thetag' angle from the center of opponent goal 
'''

required_df=required_df.reset_index(drop=True)
features_dict = {}
features_dict["zone_s"]=None
features_dict["zone_deltax"]=None
features_dict["zone_deltay"]=None
features_dict["zone_sg"]=None
features_dict["zone_thetag"]=None
new_columns_df = pd.DataFrame(features_dict, index=required_df.index)
required_df = pd.concat([required_df, new_columns_df], axis=1)

goal_x = 120*105/120
goal_y = 40*68/80

#create zone_x and zone_y with the centroid coordinate
# Create a dictionary to map zone values to centroid coordinates
zone_to_centroid_x = {zone: x for zone, x in enumerate(centroid_x)}
zone_to_centroid_y = {zone: y for zone, y in enumerate(centroid_y)}
# Map zone values to centroid coordinates using pd.Series.map
required_df["zone_x"] = required_df["zone"].map(zone_to_centroid_x)
required_df["zone_y"] = required_df["zone"].map(zone_to_centroid_y)

# Calculate zone_deltay and zone_deltax using vectorized operations
required_df["zone_deltax"] = required_df["zone_x"].diff().fillna(0)
required_df["zone_deltay"] = required_df["zone_y"].diff().fillna(0)
# Calculate zone_s using vectorized operation
required_df["zone_s"] = np.sqrt(required_df["zone_deltax"] ** 2 + required_df["zone_deltay"] ** 2)
# Calculate zone_sg, zone_thetag using vectorized operations
required_df["zone_sg"] = np.sqrt((required_df["zone_x"] - goal_x) ** 2 + (required_df["zone_y"] - goal_y) ** 2)
required_df["zone_thetag"] = np.abs(np.arctan2(required_df["zone_y"] - goal_y,required_df["zone_x"] - goal_x))

print("features created")

#remove the interevent time for first row of each match
# Identify the first row for each 'match_id'
first_rows_mask = required_df.groupby(['match_id',"period"]).cumcount() == 0
# Set the 'deltaT',zone_s,zone_deltax,zone_deltay, value to 0 for the first row of each 'match_id'
required_df.loc[first_rows_mask, 'deltaT'] = 0
required_df.loc[first_rows_mask, 'zone_s'] = 0
required_df.loc[first_rows_mask, 'zone_deltax'] = 0
required_df.loc[first_rows_mask, 'zone_deltay'] = 0

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

if args.sample_df:
    required_df=required_df.reset_index(drop=True)
    required_df.head(10000).to_csv(out_path+"preprocessed_event_reduced.csv",index=False) 
    # pdb.set_trace()

#save the required_df
required_df=required_df.reset_index(drop=True)
required_df.to_csv(out_path+"preprocessed_event_df.csv",index=False)

print("-------preprocessing done-------")
'''
Required features
    #current features for the NMSTPP model

    ##input_vars
    'act': action type
    'deltaT': interevent time
    'zone_s': dist from the previous zone centroid to the current zone centroid
    'zone_deltay': change in y direction of the zone centroid
    'zone_deltax': change in x direction of the zone centroid
    'zone_sg': dist from the mid point of opposing team's goal line to the current zone centroid
    'zone_thetag': angle from the mid point of opposing team's goal line to the current zone centroid
    'zone': zone of the football pitch following Juego de posición (position game)

    ##target_vars 
    'deltaT': interevent time
    'zone': zone of the football pitch following Juego de posición (position game)
    'act': the action type

    #new features to be processed
    "freeze_frame" the location and is teammate of players in the freeze frame

Action grouping 


#Pass 
"{'id': 18, 'name': 'Half Start'}",
"{'id': 30, 'name': 'Pass'}" and other then cross,
"{'id': 9, 'name': 'Clearance'}",


#Dribble
"{'id': 43, 'name': 'Carry'}", 
"{'id': 4, 'name': 'Duel'}",
"{'id': 14, 'name': 'Dribble'}",
"{'id': 33, 'name': '50/50'}",

#Cross
"{'id': 30, 'name': 'Pass'}" and "pass" type ={'id': 61, 'name': 'Corner'} cross=True

#Shot       
"{'id': 16, 'name': 'Shot'}",
"{'id': 37, 'name': 'Error'}",

#possesion end

And after all action in a possession
df.loc[chgpos_idxn2,"act"] = "_"   #action type
df.loc[chgpos_idxn2,"deltaT"] = 0  #inter event time
df.loc[chgpos_idxn2,"s"] = 0       #distance to goal
df.loc[chgpos_idxn2,"theta"] = 0.5 #angle to gaol


# N/A (removed)
"{'id': 41, 'name': 'Referee Ball-Drop'}",
"{'id': 35, 'name': 'Starting XI'}",
"{'id': 42, 'name': 'Ball Receipt*'}",
"{'id': 17, 'name': 'Pressure'}",
"{'id': 22, 'name': 'Foul Committed'}",
"{'id': 21, 'name': 'Foul Won'}", 
"{'id': 2, 'name': 'Ball Recovery'}",
"{'id': 6, 'name': 'Block'}",
"{'id': 10, 'name': 'Interception'}", 
"{'id': 23, 'name': 'Goal Keeper'}",
"{'id': 39, 'name': 'Dribbled Past'}",
"{'id': 38, 'name': 'Miscontrol'}",
"{'id': 3, 'name': 'Dispossessed'}", 
"{'id': 40, 'name': 'Injury Stoppage'}",
"{'id': 19, 'name': 'Substitution'}",
"{'id': 36, 'name': 'Tactical Shift'}",
"{'id': 27, 'name': 'Player Off'}",
"{'id': 26, 'name': 'Player On'}", 
"{'id': 28, 'name': 'Shield'}",
"{'id': 24, 'name': 'Bad Behaviour'}",
"{'id': 8, 'name': 'Offside'}"],
"{'id': 34, 'name': 'Half End'}",       #consider the possesion end for the half
"{'id': 25, 'name': 'Own Goal For'}", #remove match that have own goal 

n=60
    [3837241, 3805086, 3805112, 3837239, 3837550, 3837414, 3805217,
       3805323, 3837529, 3837305, 3805211, 3805212, 3805053, 3805021,
       3837266, 3837467, 3837394, 3837329, 3837282, 3837379, 3805075,
       3805176, 3805026, 3837345, 3805351, 3837477, 3837231, 3805191,
       3837364, 3837265, 3837479, 3837230, 3837547, 3837374, 3837247,
       3805098, 3837242, 3805355, 3805061, 3837404, 3805197, 3805223,
       3805137, 3805179, 3805201, 3805105, 3837234, 3837316, 3837272,
       3805360, 3837445, 3837421, 3837573, 3837578, 3837608, 3805227,
       3805339, 3837275, 3808862, 3837559]

       
"{'id': 20, 'name': 'Own Goal Against'}", #should be the same as own goal for

       array([3837241, 3805086, 3805112, 3837239, 3837550, 3837414, 3805217,
       3805323, 3837529, 3837305, 3805211, 3805212, 3805053, 3805021,
       3837266, 3837467, 3837394, 3837329, 3837282, 3837379, 3805075,
       3805176, 3805026, 3837345, 3805351, 3837477, 3837231, 3805191,
       3837364, 3837265, 3837479, 3837230, 3837547, 3837374, 3837247,
       3805098, 3837242, 3805355, 3805061, 3837404, 3805197, 3805223,
       3805137, 3805179, 3805201, 3805105, 3837234, 3837316, 3837272,
       3805360, 3837445, 3837421, 3837573, 3837578, 3837608, 3805227,
       3805339, 3837275, 3808862, 3837559])

'''

'''
df columns required
id                      #unique id
index                   #remain for now indexing the event number in a match
period                  #remain 1st half and 2nd half, remove extra time and penalty (drop 3,4,5 here and remove event after 45 and 90 minute for each half))) 
timestamp               #the record to millisecond
type                    #action type
possession              #indicate a unique possession
possession_team         #team in possession
play_pattern            #how the possession started
team                    #id and name of the team performing the action
duration                #duration of the action
tactics                 #formation of the team
#player                 # player performing the action
#position               # player role of the player performing the action
location                #location x,y of the action 
match_id                #match id
freeze_frame            #freeze frame data
injury_stoppage         #should remove the injury stoppage n=499

All columns in df
id
index
period
timestamp
minute
second
type
possession
possession_team
play_pattern
obv_for_after
obv_for_before
obv_for_net
obv_against_after
obv_against_before
obv_against_net
obv_total_net
team
duration
tactics
related_events
player
position
location
pass
carry
under_pressure
duel
ball_receipt
counterpress
clearance
off_camera
interception
shot
goalkeeper
dribble
out
50_50
block
foul_won
foul_committed
ball_recovery
substitution
match_id
visible_area
freeze_frame
line_breaking_pass
num_defenders_on_goal_side_of_actor
distance_to_nearest_defender
ball_receipt_in_space
ball_receipt_exceeds_distance
visible_player_counts
distances_from_edge_of_visible_area
injury_stoppage
miscontrol
bad_behaviour
player_off
half_start
half_end

'''

'''
Teams

array(["{'id': 25, 'name': 'Southampton'}",
       "{'id': 36, 'name': 'Manchester City'}",
       "{'id': 35, 'name': 'Brighton & Hove Albion'}",
       "{'id': 46, 'name': 'Wolverhampton Wanderers'}",
       "{'id': 29, 'name': 'Everton'}",
       "{'id': 22, 'name': 'Leicester City'}",
       "{'id': 56, 'name': 'Norwich City'}",
       "{'id': 38, 'name': 'Tottenham Hotspur'}",
       "{'id': 40, 'name': 'West Ham United'}",
       "{'id': 101, 'name': 'Leeds United'}",
       "{'id': 31, 'name': 'Crystal Palace'}",
       "{'id': 24, 'name': 'Liverpool'}", "{'id': 33, 'name': 'Chelsea'}",
       "{'id': 39, 'name': 'Manchester United'}",
       "{'id': 28, 'name': 'AFC Bournemouth'}",
       "{'id': 93, 'name': 'Brentford'}", "{'id': 23, 'name': 'Watford'}",
       "{'id': 59, 'name': 'Aston Villa'}",
       "{'id': 1, 'name': 'Arsenal'}", "{'id': 34, 'name': 'Burnley'}",
       "{'id': 37, 'name': 'Newcastle United'}",
       "{'id': 55, 'name': 'Fulham'}",
       "{'id': 43, 'name': 'Nottingham Forest'}"], dtype=object)
'''
