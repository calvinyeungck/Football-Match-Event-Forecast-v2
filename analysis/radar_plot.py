import pandas as pd
import numpy as np
import argparse
from mplsoccer import Radar, FontManager, grid
import matplotlib.pyplot as plt
import os
import math
import pdb
#ref https://mplsoccer.readthedocs.io/en/latest/gallery/radar/plot_radar.html

parser = argparse.ArgumentParser()
parser.add_argument('--sequnces_pattern_df_path', type=str, default='/home/c_yeung/workspace6/python/statsbomb_conference_2023/script/analysis/sequences_pattern.csv')
parser.add_argument('--out_path', type=str, default='/home/c_yeung/workspace6/python/statsbomb_conference_2023/script/analysis/fig_radar_final/')
parser.add_argument('--with_other','-wo', default=True, action='store_true')
args = parser.parse_args()

if args.with_other:
        args.out_path = os.path.join(args.out_path,'wo/')

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

man_city=get_value(df[df["possession_team"]=="{'id': 36, 'name': 'Manchester City'}"],args.with_other)
man_city_opp=get_value_opp(df,"{'id': 36, 'name': 'Manchester City'}",args.with_other)
arsenal=get_value(df[df["possession_team"]=="{'id': 1, 'name': 'Arsenal'}"],args.with_other)
arsenal_opp=get_value_opp(df,"{'id': 1, 'name': 'Arsenal'}",args.with_other)
man_utd=get_value(df[df["possession_team"]=="{'id': 39, 'name': 'Manchester United'}"],args.with_other)
man_utd_opp=get_value_opp(df,"{'id': 39, 'name': 'Manchester United'}",args.with_other)

team={}
for team_i in df.possession_team.unique():
    team_i_value=get_value(df[df["possession_team"]==team_i],args.with_other)
    team_i_value_opp=get_value_opp(df,team_i,args.with_other)
    team[team_i]=[team_i_value,team_i_value_opp]    
    

def round_down_to_nearest(number, rounding_value):
    return math.floor(number / rounding_value) * rounding_value

def round_up_to_nearest(number, rounding_value):
    return math.ceil(number / rounding_value) * rounding_value

arsenal_color = '#EF0107'
man_city_color = '#6CADDF'
man_utd_color = '#FFB6C1'
edgecolor_line='#6495ED'

# plot radar
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal[2]+man_city[2]+man_utd[2]),10)
upper_bound = round_up_to_nearest(max(arsenal[2]+man_city[2]+man_utd[2]),10)

low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar1, vertices1 = radar.draw_radar_solid(man_city[2], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar2, vertices2 = radar.draw_radar_solid(arsenal[2], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar3, vertices3 = radar.draw_radar_solid(man_utd[2], ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Manchester United', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Average HPUS per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot.png', dpi=300, bbox_inches='tight')
plt.close()




# plot radar
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal[5]+man_city[5]+man_utd[5]),10)
upper_bound = round_up_to_nearest(max(arsenal[5]+man_city[5]+man_utd[5]),10)

low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar1, vertices1 = radar.draw_radar_solid(man_city[5], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar2, vertices2 = radar.draw_radar_solid(arsenal[5], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar3, vertices3 = radar.draw_radar_solid(man_utd[5], ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Manchester United', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Average HPUS+ per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_plus.png', dpi=300, bbox_inches='tight')
plt.close()





# plot radar
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal[6]+man_city[6]+man_utd[6]),10)
upper_bound = round_up_to_nearest(max(arsenal[6]+man_city[6]+man_utd[6]),10)


low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar1, vertices1 = radar.draw_radar_solid(man_city[6], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar2, vertices2 = radar.draw_radar_solid(arsenal[6], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar3, vertices3 = radar.draw_radar_solid(man_utd[6], ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Manchester United', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Average Possession per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_len.png', dpi=300, bbox_inches='tight')
plt.close()



lower_bound = round_down_to_nearest(min(arsenal_opp[2]+man_city_opp[2]+man_utd_opp[2]),10)
upper_bound = round_up_to_nearest(max(arsenal_opp[2]+man_city_opp[2]+man_utd_opp[2]),10)

low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar3, vertices3 = radar.draw_radar_solid(man_utd_opp[2], ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar1, vertices1 = radar.draw_radar_solid(man_city_opp[2], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar2, vertices2 = radar.draw_radar_solid(arsenal_opp[2], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Manchester United', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Opponent Average HPUS per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_opp.png', dpi=300, bbox_inches='tight')
plt.close()




# plot radar
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal_opp[5]+man_city_opp[5]+man_utd_opp[5]),10)
upper_bound = round_up_to_nearest(max(arsenal_opp[5]+man_city_opp[5]+man_utd_opp[5]),10)
low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar3, vertices3 = radar.draw_radar_solid(man_utd_opp[5], ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar2, vertices2 = radar.draw_radar_solid(arsenal_opp[5], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar1, vertices1 = radar.draw_radar_solid(man_city_opp[5], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Manchester United', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Opponent Average HPUS+ per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_plus_opp.png', dpi=300, bbox_inches='tight')
plt.close()





# plot radar
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal_opp[6]+man_city_opp[6]+man_utd_opp[6]),10)
upper_bound = round_up_to_nearest(max(arsenal_opp[6]+man_city_opp[6]+man_utd_opp[6]),10)


low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar3, vertices3 = radar.draw_radar_solid(man_utd_opp[6], ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar1, vertices1 = radar.draw_radar_solid(man_city_opp[6], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar2, vertices2 = radar.draw_radar_solid(arsenal_opp[6], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})



ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Manchester United', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Opponent Average Possession per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_len_opp.png', dpi=300, bbox_inches='tight')
plt.close()



man_utd_color = '#98FB98'
# plot radar
average_team=[0]*len(params)
count=0
for key,value in team.items():
        average_team=[x + y for x, y in zip(average_team, team[key][0][2])]
        count+=1
average_team=[x/count for x in average_team]
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal[2]+man_city[2]+average_team),10)
upper_bound = round_up_to_nearest(max(arsenal[2]+man_city[2]+average_team),10)

low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar1, vertices1 = radar.draw_radar_solid(man_city[2], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar2, vertices2 = radar.draw_radar_solid(arsenal[2], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})



radar3, vertices3 = radar.draw_radar_solid(average_team, ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Average Team', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Average HPUS per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_avg.png', dpi=300, bbox_inches='tight')
plt.close()




# plot radar
average_team=[0]*len(params)
count=0
for key,value in team.items():
        average_team=[x + y for x, y in zip(average_team, team[key][0][5])]
        count+=1
average_team=[x/count for x in average_team]
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal[5]+man_city[5]+average_team),10)
upper_bound = round_up_to_nearest(max(arsenal[5]+man_city[5]+average_team),10)

low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar1, vertices1 = radar.draw_radar_solid(man_city[5], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar2, vertices2 = radar.draw_radar_solid(arsenal[5], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})



radar3, vertices3 = radar.draw_radar_solid(average_team, ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Average Team', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Average HPUS+ per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_plus_avg.png', dpi=300, bbox_inches='tight')
plt.close()





# plot radar
average_team=[0]*len(params)
count=0
for key,value in team.items():
        average_team=[x + y for x, y in zip(average_team, team[key][0][6])]
        count+=1
average_team=[x/count for x in average_team]
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal[6]+man_city[6]+average_team),10)
upper_bound = round_up_to_nearest(max(arsenal[6]+man_city[6]+average_team),10)


low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar1, vertices1 = radar.draw_radar_solid(man_city[6], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar2, vertices2 = radar.draw_radar_solid(arsenal[6], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})



radar3, vertices3 = radar.draw_radar_solid(average_team, ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Average Team', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Average Possession per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_len_avg.png', dpi=300, bbox_inches='tight')
plt.close()





# plot radar
average_team=[0]*len(params)
count=0
for key,value in team.items():
        average_team=[x + y for x, y in zip(average_team, team[key][1][2])]
        count+=1
average_team=[x/count for x in average_team]
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal_opp[2]+man_city_opp[2]+average_team),10)
upper_bound = round_up_to_nearest(max(arsenal_opp[2]+man_city_opp[2]+average_team),10)

low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar3, vertices3 = radar.draw_radar_solid(average_team, ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})
radar2, vertices2 = radar.draw_radar_solid(arsenal_opp[2], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})
radar1, vertices1 = radar.draw_radar_solid(man_city_opp[2], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Average Team', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Opponent Average HPUS per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_avg_opp.png', dpi=300, bbox_inches='tight')
plt.close()




# plot radar
average_team=[0]*len(params)
count=0
for key,value in team.items():
        average_team=[x + y for x, y in zip(average_team, team[key][1][5])]
        count+=1
average_team=[x/count for x in average_team]
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal_opp[5]+man_city_opp[5]+average_team),10)
upper_bound = round_up_to_nearest(max(arsenal_opp[5]+man_city_opp[5]+average_team),10)

low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')


radar3, vertices3 = radar.draw_radar_solid(average_team, ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})
radar2, vertices2 = radar.draw_radar_solid(arsenal_opp[5], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})
radar1, vertices1 = radar.draw_radar_solid(man_city_opp[5], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Average Team', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Opponent Average HPUS+ per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_plus_avg_opp.png', dpi=300, bbox_inches='tight')
plt.close()





# plot radar
average_team=[0]*len(params)
count=0
for key,value in team.items():
        average_team=[x + y for x, y in zip(average_team, team[key][1][6])]
        count+=1
average_team=[x/count for x in average_team]
# The lower and upper boundaries for the statistics
lower_bound = round_down_to_nearest(min(arsenal_opp[6]+man_city_opp[6]+average_team),10)
upper_bound = round_up_to_nearest(max(arsenal_opp[6]+man_city_opp[6]+average_team),10)


low =  [lower_bound]*len(params)
high = [upper_bound]*len(params)

# Add anything to this list where having a lower number is better
# this flips the statistic
lower_is_better = ['Miscontrol']

radar = Radar(params, low, high,
              round_int=[False]*len(params),
              num_rings=4,  
              ring_width=1, center_circle_radius=1)


fig, ax = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

# plot radar
radar.setup_axis(ax=ax['radar'], facecolor='None')  # format axis as a radar

# fig, ax = radar.setup_axis()
rings_inner = radar.draw_circles(ax=ax['radar'], facecolor='lightblue', edgecolor='lightblue')

radar3, vertices3 = radar.draw_radar_solid(average_team, ax=ax['radar'],
                                           kwargs={'facecolor': man_utd_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})



radar2, vertices2 = radar.draw_radar_solid(arsenal_opp[6], ax=ax['radar'],
                                           kwargs={'facecolor': arsenal_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})

radar1, vertices1 = radar.draw_radar_solid(man_city_opp[6], ax=ax['radar'],
                                           kwargs={'facecolor': man_city_color,
                                                   'alpha': 0.6,
                                                   'edgecolor': edgecolor_line,
                                                   'lw': 3})



ax['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
           c=man_city_color, edgecolors=man_city_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
           c=arsenal_color, edgecolors=arsenal_color, marker='o', s=150, zorder=2)
ax['radar'].scatter(vertices3[:, 0], vertices3[:, 1],
           c=man_utd_color, edgecolors=man_utd_color, marker='o', s=150, zorder=2)

title1_text = ax['title'].text(0.01, -0.60, 'Manchester City', fontsize=36, color=man_city_color,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.20, 'Arsenal', fontsize=36,color=arsenal_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.01, -1.80, 'Average Team', fontsize=36,color=man_utd_color,
                                fontproperties=robotto_bold.prop,ha='left', va='center')
title2_text = ax['title'].text(0.50, 0.60, 'Opponent Average Possession per Match', fontsize=48,color='black',
                                fontproperties=robotto_bold.prop,ha='center', va='center')


range_labels = radar.draw_range_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=ax['radar'], fontsize=36, fontproperties=robotto_bold.prop)
fig.set_facecolor('#F0F0F0')
#save 
plt.savefig(args.out_path + 'radar_plot_len_avg_opp.png', dpi=300, bbox_inches='tight')
plt.close()