import pandas as pd
import numpy as np
import argparse
from mplsoccer import Radar, FontManager, grid
import matplotlib.pyplot as plt
import os
import math
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

#ref https://mplsoccer.readthedocs.io/en/latest/gallery/radar/plot_radar.html

parser = argparse.ArgumentParser()
parser.add_argument('--sequnces_pattern_df_path', type=str, default='analysis/sequences_pattern.csv')
parser.add_argument('--out_path', type=str, default='analysis/fig_clust')
parser.add_argument('--with_other','-wo', default=False, action='store_true')
args = parser.parse_args()

if args.with_other:
        args.out_path = os.path.join(args.out_path,'/wo/')

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

# Load df
df = pd.read_csv(args.sequnces_pattern_df_path)
df['HPUS+'] = df.apply(lambda row: row['HPUS'] if row['attack'] else 0, axis=1)




# parameter names of the statistics we want to show
if args.with_other:
    params = ["Pass Based","Dribble Based","Alternating Pass-Dribble Based","Own Half Based","Opponent Wing Based","Others"]
else:
    params = ["Pass Based","Dribble Based","Alternating Pass-Dribble Based","Own Half Based","Opponent Wing Based"]



URL1 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
        'SourceSerifPro-Regular.ttf')
serif_regular = FontManager(URL1)
URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
        'SourceSerifPro-ExtraLight.ttf')
serif_extra_light = FontManager(URL2)
URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/'
        'RubikMonoOne-Regular.ttf')
rubik_regular = FontManager(URL3)
URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
robotto_thin = FontManager(URL4)
URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
        'RobotoSlab%5Bwght%5D.ttf')
robotto_bold = FontManager(URL5)

def get_value(dataframe,with_other=False):
    df1=dataframe[dataframe["pass_based"]==1]
    count1=df1.groupby('match_id')['HPUS'].agg([('HPUS', 'sum'), ('row_count', 'count')])
    df1=df1.groupby('match_id')[['HPUS', 'HPUS+']].sum()
    df2=dataframe[dataframe["dirbble_based"]==1]
    count2=df2.groupby('match_id')['HPUS'].agg([('HPUS', 'sum'), ('row_count', 'count')])
    df2=df2.groupby('match_id')[['HPUS', 'HPUS+']].sum()
    df3=dataframe[dataframe["pass_and_dirbble_based"]==1]
    count3=df3.groupby('match_id')['HPUS'].agg([('HPUS', 'sum'), ('row_count', 'count')])
    df3=df3.groupby('match_id')[['HPUS', 'HPUS+']].sum()
    df4=dataframe[dataframe["wing_based"]==1]
    count4=df4.groupby('match_id')['HPUS'].agg([('HPUS', 'sum'), ('row_count', 'count')])
    df4=df4.groupby('match_id')[['HPUS', 'HPUS+']].sum()
    df5=dataframe[dataframe["own_half_based"]==1]
    count5=df5.groupby('match_id')['HPUS'].agg([('HPUS', 'sum'), ('row_count', 'count')])
    df5=df5.groupby('match_id')[['HPUS', 'HPUS+']].sum()
    if with_other:
        df6=dataframe[dataframe["other"]==1]
        count6=df6.groupby('match_id')['HPUS'].agg([('HPUS', 'sum'), ('row_count', 'count')])
        df6=df6.groupby('match_id')[['HPUS', 'HPUS+']].sum()
        max_value= [df1.HPUS.max(),df2.HPUS.max(),df3.HPUS.max(),df4.HPUS.max(),df5.HPUS.max(),df6.HPUS.max()]
        min_value= [df1.HPUS.min(),df2.HPUS.min(),df3.HPUS.min(),df4.HPUS.min(),df5.HPUS.min(),df6.HPUS.min()]
        average_value= [df1.HPUS.mean(),df2.HPUS.mean(),df3.HPUS.mean(),df4.HPUS.mean(),df5.HPUS.mean(),df6.HPUS.mean()]
        max_value_plus= [df1["HPUS+"].max(),df2["HPUS+"].max(),df3["HPUS+"].max(),df4["HPUS+"].max(),df5["HPUS+"].max(),df6["HPUS+"].max()]
        min_value_plus= [df1["HPUS+"].min(),df2["HPUS+"].min(),df3["HPUS+"].min(),df4["HPUS+"].min(),df5["HPUS+"].min(),df6["HPUS+"].min()]
        average_value_plus= [df1["HPUS+"].mean(),df2["HPUS+"].mean(),df3["HPUS+"].mean(),df4["HPUS+"].mean(),df5["HPUS+"].mean(),df6["HPUS+"].mean()]
        length= [count1.row_count.mean(),count2.row_count.mean(),count3.row_count.mean(),count4.row_count.mean(),count5.row_count.mean(),count6.row_count.mean()]
    else:
        max_value= [df1.HPUS.max(),df2.HPUS.max(),df3.HPUS.max(),df4.HPUS.max(),df5.HPUS.max()]
        min_value= [df1.HPUS.min(),df2.HPUS.min(),df3.HPUS.min(),df4.HPUS.min(),df5.HPUS.min()]
        average_value= [df1.HPUS.mean(),df2.HPUS.mean(),df3.HPUS.mean(),df4.HPUS.mean(),df5.HPUS.mean()]
        max_value_plus= [df1["HPUS+"].max(),df2["HPUS+"].max(),df3["HPUS+"].max(),df4["HPUS+"].max(),df5["HPUS+"].max()]
        min_value_plus= [df1["HPUS+"].min(),df2["HPUS+"].min(),df3["HPUS+"].min(),df4["HPUS+"].min(),df5["HPUS+"].min()]
        average_value_plus= [df1["HPUS+"].mean(),df2["HPUS+"].mean(),df3["HPUS+"].mean(),df4["HPUS+"].mean(),df5["HPUS+"].mean()]
        length=[count1.row_count.mean(),count2.row_count.mean(),count3.row_count.mean(),count4.row_count.mean(),count5.row_count.mean()]
    return max_value,min_value,average_value,max_value_plus,min_value_plus,average_value_plus,length

def get_value_opp(dataframe,team,with_other=False):
    match_list=dataframe[dataframe["possession_team"]==team].match_id.unique()
    dataframe=dataframe[dataframe["match_id"].isin(match_list)]
    dataframe=dataframe[dataframe["possession_team"]!=team]
    return get_value(dataframe,with_other)

# man_city=get_value(df[df["possession_team"]=="{'id': 36, 'name': 'Manchester City'}"],args.with_other)
# man_city_opp=get_value_opp(df,"{'id': 36, 'name': 'Manchester City'}",args.with_other)
# arsenal=get_value(df[df["possession_team"]=="{'id': 1, 'name': 'Arsenal'}"],args.with_other)
# arsenal_opp=get_value_opp(df,"{'id': 1, 'name': 'Arsenal'}",args.with_other)
# man_utd=get_value(df[df["possession_team"]=="{'id': 39, 'name': 'Manchester United'}"],args.with_other)
# man_utd_opp=get_value_opp(df,"{'id': 39, 'name': 'Manchester United'}",args.with_other)



def round_down_to_nearest(number, rounding_value):
    return math.floor(number / rounding_value) * rounding_value

def round_up_to_nearest(number, rounding_value):
    return math.ceil(number / rounding_value) * rounding_value

arsenal_color = '#EF0107'
man_city_color = '#6CADDF'
man_utd_color = '#FFB6C1'
edgecolor_line='#6495ED'

team_dict={}

for team_i in df.possession_team.unique():
    team_i_value=get_value(df[df["possession_team"]==team_i],args.with_other)
    team_i_opp_value=get_value_opp(df,team_i,args.with_other)
    team_dict[team_i]=[team_i_value,team_i_opp_value]


for i in range(len(params)):
    sns.set()
    plt.figure(figsize=(8, 6))
    for key, value in team_dict.items():
        # pdb.set_trace()
        x=value[0][6][i]
        y=value[0][2][i]
        sns.scatterplot(x=[x], y=[y], s=100)
        key=eval(key)
        plt.text(x, y, key["name"], fontsize=6, ha='left', va='bottom')

    plt.title(f"Tactics: {params[i]}")
    plt.ylabel("HPUS")
    plt.xlabel("Number of Possession")
    plt.savefig(args.out_path+f'/{params[i]}_scat.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    for key, value in team_dict.items():
        # pdb.set_trace()
        x=value[0][6][i]
        y=value[0][5][i]
        sns.scatterplot(x=[x], y=[y], s=100)
        key=eval(key)
        plt.text(x, y, key["name"], fontsize=6, ha='left', va='bottom')

    plt.title(f"Tactics: {params[i]}")
    plt.ylabel("HPUS+")
    plt.xlabel("Number of Possession")
    plt.savefig(args.out_path+f'/{params[i]}_scat_plus.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    clustering_points=[]
    plt.figure(figsize=(8, 6))
    for key, value in team_dict.items():
        # pdb.set_trace()
        x=value[0][6][i]
        y=value[0][5][i]/value[0][2][i]
        clustering_points.append([x,y])
        sns.scatterplot(x=[x], y=[y], s=100)
        #turn key from string to dict
        key=eval(key)
        plt.text(x, y, key["name"], fontsize=6, ha='left', va='bottom')

    plt.title(f"Tactics: {params[i]}")
    plt.ylabel("HPUS Ratio")
    plt.xlabel("Number of Possession")
    plt.savefig(args.out_path+f'/{params[i]}_scat_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    clustering_points=np.array(clustering_points)
    bandwidth = estimate_bandwidth(clustering_points, quantile=0.2, n_samples=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(clustering_points)
    cluster_centers = ms.cluster_centers_
    labels = ms.labels_
    n_clusters = len(np.unique(labels))
    plt.figure(figsize=(8, 6))
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, color='gray', marker='X')
    plt.scatter(clustering_points[:, 0], clustering_points[:, 1], c=labels, cmap='rainbow')
    for j in range(len(team_dict)):
        key=list(team_dict.keys())[j]
        key=eval(key)
        x=clustering_points[j][0]
        y=clustering_points[j][1]
        plt.text(x, y, key["name"], fontsize=6, ha='left', va='bottom')
    plt.title(f'Tactics: {params[i]}, Mean Shift Clustering')
    plt.ylabel('HPUS Ratio')
    plt.xlabel('Number of Possession')
    plt.savefig(args.out_path+f'/{params[i]}_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()     
#     pdb.set_trace()


    plt.figure(figsize=(8, 6))
    for key, value in team_dict.items():
        # pdb.set_trace()
        x=value[1][6][i]
        y=value[1][2][i]
        sns.scatterplot(x=[x], y=[y], s=100)
        key=eval(key)
        plt.text(x, y, key["name"], fontsize=6, ha='left', va='bottom')

    plt.title(f"Tactics: {params[i]}")
    plt.ylabel("HPUS")
    plt.xlabel("Number of Possession")
    plt.savefig(args.out_path+f'/{params[i]}_scat_opp.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    for key, value in team_dict.items():
        # pdb.set_trace()
        x=value[1][6][i]
        y=value[1][5][i]
        sns.scatterplot(x=[x], y=[y], s=100)
        key=eval(key)
        plt.text(x, y, key["name"], fontsize=6, ha='left', va='bottom')

    plt.title(f"Tactics: {params[i]}")
    plt.xlabel("HPUS+")
    plt.ylabel("Number of Possession")
    plt.savefig(args.out_path+f'/{params[i]}_scat_plus_opp.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    for key, value in team_dict.items():
        # pdb.set_trace()
        x=value[1][6][i]
        y=value[1][5][i]/value[1][2][i]
        sns.scatterplot(x=[x], y=[y], s=100)
        #turn key from string to dict
        key=eval(key)
        plt.text(x, y, key["name"], fontsize=6, ha='left', va='bottom')

    plt.title(f"Tactics: {params[i]}")
    plt.ylabel("HPUS Ratio")
    plt.xlabel("Number of Possession")
    plt.savefig(args.out_path+f'/{params[i]}_scat_ratio_opp.png', dpi=300, bbox_inches='tight')
    plt.close()


