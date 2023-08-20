import pandas as pd
import os
import json
from datetime import datetime
import argparse
import spmf
import itertools
import pdb

# Suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

def timestamp_to_seconds(timestamp):
    try:
        # Check if the timestamp is in the format 'HH:MM:SS.SSSSSS'
        dt = datetime.strptime(timestamp, '%H:%M:%S.%f')
        return (dt - dt.replace(microsecond=0)).total_seconds()
    except ValueError:
        pass

    try:
        # Check if the timestamp is in the format 'MM:SS.SSSSSS'
        dt = datetime.strptime(timestamp, '%M:%S.%f')
        return (dt - dt.replace(microsecond=0)).total_seconds()
    except ValueError:
        pass

    try:
        # Check if the timestamp is in the format 'SS.SSSSSS'
        dt = datetime.strptime(timestamp, '%S.%f')
        return dt.second + dt.microsecond / 1e6
    except ValueError:
        # If none of the above formats match, consider it as just seconds
        try:
            return float(timestamp)
        except ValueError:
            # If the timestamp is not a valid number, return NaN
            return float('nan')

def dict_to_tuple(d):
    return tuple(d.items())

def group_concurrent_act_zones(group):
    result = []
    current_list = []
    current_timestamp = None

    for _, row in group.iterrows():
        timestamp = timestamp_to_seconds(row['timestamp'])
        act_zone = row['act_zone']

        if current_timestamp is None or abs(timestamp - current_timestamp) == 0:
            current_list.append(act_zone)
        else:
            if current_list:
                result.append(current_list)
            current_list = [act_zone]

        current_timestamp = timestamp

    if current_list:
        result.append(current_list)

    return result

def group_concurrent_acts(group):
    # group = group.sort_values('timestamp')

    # Convert timestamp to seconds
    # group['timestamp_sec'] = group['timestamp'].apply(timestamp_to_seconds)

    # Initialize variables
    # grouped_acts = []
    current_group = []
    # last_timestamp = None

    for _, row in group.iterrows():
        # timestamp_sec = row['timestamp_sec']
        act = row['act']
        if act!='end':
        # if last_timestamp is None or last_timestamp == timestamp_sec:
            current_group.append([act])
        # else:
        #     if current_group:
        #         grouped_acts.append(current_group)
        #     current_group = [act]

        # last_timestamp = timestamp_sec

    # if current_group:
    # grouped_acts.append(current_group)

    return current_group

# Create a function to encode individual elements within a list
def encode_nested_list_act_zone(nested_list):
    if isinstance(nested_list, list):
        return [encode_nested_list_act_zone(act_zone_element) if isinstance(act_zone_element, list) else act_zone_mapping[act_zone_element] for act_zone_element in nested_list]
    else:
        return act_zone_mapping[nested_list]

def encode_nested_list_act(nested_list):
    if isinstance(nested_list, list):
        return [encode_nested_list_act(act_element) if isinstance(act_element, list) else act_mapping[act_element] for act_element in nested_list]
    else:
        return act_mapping[nested_list]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess StatsBomb data and create SPMF input files.')
    parser.add_argument('-t', '--possession_team', type=str, help='Subset to consider data for the specified possession team.')
    parser.add_argument('-i', '--input_dir', required=False, type=str, default="/home/c_yeung/workspace6/python/statsbomb_conference_2023/data/analysis_df.csv", help='The directory where the train_df2.csv is located.')
    parser.add_argument('-o', '--output_dir', required=False, type=str, default=os.path.join(os.getcwd(), 'sequence_data_action_analysis'), help='The directory where you want to the output files to be written out to.')
    parser.add_argument('--seq_mining_output_dir', required=False, type=str, default=os.path.join(os.getcwd(), 'spmf_output_action_analysis'), help='The directory where you want to the output files to be written out to.')
    parser.add_argument('-a', '--algorithm', required=False, default="CM-SPADE", choices=["CM-SPADE", "PrefixSpan"], type=str, help='SPMF algorithm to apply: CM-SPADE or PrefixSpan')
    parser.add_argument('-s', '--min_sup', required=False, type=float, default=0.5, help='minimum support threshold (%). Specifies the minimum support required for a pattern to be considered frequent. E.g., if set to 0.05, only patterns that appear in at least 5% of the sequences will be considered frequent.')
    parser.add_argument('-l', '--max_length', required=False, type=int, default=100, help='maximum pattern length to be considered by the CM-SPADE algorithm. This parameter sets a limit on the length of the patterns that the algorithm will consider. E.g., if set to 5, the algorithm will only consider patterns of up to five items in length.')
    parser.add_argument('--seq_type', required=False, type=str, default='action', help='actionzone or action')
    args = parser.parse_args()

    # ---------------------- Transform train_df2.csv data into sequences in SPMF format ---------------------- 

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.seq_mining_output_dir):
        os.makedirs(args.seq_mining_output_dir)

    df = pd.read_csv(args.input_dir, low_memory=False)

    # sort_order = ['match_id', 'period', 'timestamp']
    # df_sorted = df.sort_values(by=sort_order)
    df_sorted = df.copy()

    df_subset = df_sorted[['match_id', 'period', 'timestamp', 'possession', 'possession_team', 'act', 'zone']]

    # # Preprocess the 'possession_team' column by replacing single quotes with double quotes
    # df_subset['possession_team'] = df_subset['possession_team'].str.replace("'", "\"")

    # Convert JSON-like strings to dictionaries
    # df_subset['possession_team'] = df_subset['possession_team'].apply(json.loads)

    # # Filter the DataFrame based on the specified 'possession_team'
    # if args.possession_team is not None:
    #     df_subset = df_subset[df_subset['possession_team'].apply(lambda x: x['name']) == args.possession_team]

    # if args.seq_type == 'actionzone':
    #     # Add a new column that combines 'act' and 'zone' with '-'
    #     df_subset['act_zone'] = df_subset['act'] + '-' + df_subset['zone'].astype(str)


    # # Convert the 'possession_team' column to tuples
    # df_subset['possession_team_tuple'] = df_subset['possession_team'].apply(dict_to_tuple)

    if args.seq_type == 'actionzone':
        # Group by 'match_id' and 'possession' and aggregate 'act_zone' values using the custom function
        sequences = df_subset.groupby(['match_id', 'possession_team_tuple', 'possession']).apply(group_concurrent_act_zones).reset_index(name='act_zone')
    elif args.seq_type == 'action':
        # Group by 'match_id' and 'possession' and aggregate 'act' values using the custom function
        sequences = df_subset.groupby(['match_id', 'possession','possession_team']).apply(group_concurrent_acts).reset_index(name='act')
        

    # Convert the 'possession_team_tuple' back to a dictionary
    # sequences['possession_team'] = sequences['possession_team_tuple'].apply(lambda x: dict(x))
    # sequences.drop(columns=['possession_team_tuple'], inplace=True)

    # Save to sequences.csv file without the index
    sequences.to_csv(os.path.join(args.output_dir, 'sequences.csv'), index=False)



    # Flatten the nested lists in the 'act_zone' column
    if args.seq_type == 'actionzone':
        sequences['act_zone_flat'] = sequences['act_zone'].apply(lambda x: [item for sublist in x for item in sublist] if isinstance(x, list) else x)
    elif args.seq_type == 'action':
        sequences['act_flat'] = sequences['act'].apply(lambda x: [item for sublist in x for item in sublist] if isinstance(x, list) else x)

    # Create the encoding mapping dictionary
    if args.seq_type == 'actionzone':
        act_zone_mapping = {}
        for act_zone in sequences['act_zone_flat']:
            if isinstance(act_zone, list):
                for act_zone_element in act_zone:
                    if act_zone_element not in act_zone_mapping:
                        act_zone_mapping[act_zone_element] = len(act_zone_mapping)
            else:
                if act_zone not in act_zone_mapping:
                    act_zone_mapping[act_zone] = len(act_zone_mapping)

    elif args.seq_type == 'action':
        act_mapping = {'pass': 1, 'dribble': 2, 'end': 3, 'shot': 4, 'cross': 5}
        # for act in sequences['act_flat']:
        #     if isinstance(act, list):
        #         for act_element in act:
        #             if act_element not in act_mapping:
        #                 act_mapping[act_element] = len(act_mapping)
        #     else:
        #         if act not in act_mapping:
        #             act_mapping[act] = len(act_mapping)
        
    # Create a new DataFrame with act_zone_encoded
    sequences_encoded = sequences.copy()
    if args.seq_type == 'actionzone':
        sequences_encoded['act_zone_encoded'] = sequences['act_zone'].apply(encode_nested_list_act_zone)
    elif args.seq_type == 'action':
        sequences_encoded['act_encoded'] = sequences['act'].apply(encode_nested_list_act)

    # Create the encoding_mapping DataFrame
    if args.seq_type == 'actionzone':
        encoding_mapping = pd.DataFrame(list(act_zone_mapping.items()), columns=['act_zone', 'act_zone_encoded'])
    elif args.seq_type == 'action':
        encoding_mapping = pd.DataFrame(list(act_mapping.items()), columns=['act', 'act_encoded'])

    # Convert the 'possession_team_tuple' back to a dictionary
    # sequences_encoded['possession_team'] = sequences_encoded['possession_team_tuple'].apply(lambda x: dict(x))
    # sequences_encoded.drop(columns=['possession_team_tuple'], inplace=True)

    # Save to sequences_encoded.csv file without the index and only the necessary columns
    if args.seq_type == 'actionzone':
        sequences_encoded[['match_id', 'possession', 'possession_team', 'act_zone_encoded']].to_csv(os.path.join(args.output_dir, 'sequences_encoded.csv'), index=False)
    elif args.seq_type == 'action':
        sequences_encoded[['match_id', 'possession', 'possession_team', 'act_encoded']].to_csv(os.path.join(args.output_dir, 'sequences_encoded.csv'), index=False)

    # Save to encoding_mapping.csv file without the index
    encoding_mapping.to_csv(os.path.join(args.output_dir, 'encoding_mapping.csv'), index=False)

    # Convert to the format required by SPMF
    def convert_to_file_format(dataset, output_file):
        with open(output_file, 'w') as f:
            for sequence in dataset:
                for itemset in sequence:
                    itemset_str = ' '.join(str(item) for item in itemset)
                    f.write(itemset_str + ' -1 ')
                f.write('-2\n')

    output_file = 'seq_spmf_format.txt' # file is called contextPrefixSpan.txt on the SPMF website/documentation
    if args.seq_type == 'actionzone':
        convert_to_file_format(sequences_encoded['act_zone_encoded'], os.path.join(args.output_dir, output_file))
    elif args.seq_type == 'action':
        convert_to_file_format(sequences_encoded['act_encoded'], os.path.join(args.output_dir, output_file))

    # ---------------------- Run sequential pattern mining using SPMF ---------------------- 
    # Create the SPMF instance and run the algorithm
    if args.possession_team is not None:
        output_filename = os.path.join(args.seq_mining_output_dir, f"{args.possession_team}_{args.algorithm}_{args.min_sup}_{args.max_length}.txt")
    elif args.possession_team is None:
        output_filename = os.path.join(args.seq_mining_output_dir, f"{args.algorithm}_{args.min_sup}_{args.max_length}.txt")

    if args.algorithm == "CM-SPADE":
        # the output_dir from above is now the input directory
        spmf_instance = spmf.Spmf(args.algorithm, input_filename=os.path.join(args.output_dir, output_file),
                                output_filename=output_filename, arguments=[args.min_sup, "", args.max_length])
    elif args.algorithm == "PrefixSpan":
        spmf_instance = spmf.Spmf(args.algorithm, input_filename=os.path.join(args.output_dir, output_file),
                                output_filename=output_filename, arguments=[args.min_sup, args.max_length])

    spmf_instance.run()

    # Load the result as a pandas DataFrame and save it to CSV
    result_df = spmf_instance.to_pandas_dataframe(pickle=True)
    # print(result_df)

    if args.possession_team is not None:
        output_csv_file = os.path.join(args.seq_mining_output_dir, f"{args.possession_team}_{args.algorithm}_{args.min_sup}_{args.max_length}.csv")
    elif args.possession_team is None:
        output_csv_file = os.path.join(args.seq_mining_output_dir, f"{args.algorithm}_{args.min_sup}_{args.max_length}.csv")
    result_df.to_csv(output_csv_file, index=False)