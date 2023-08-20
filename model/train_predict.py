import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from dataloader_fc import CustomDataset
from model_fc import NMSTPP
import torch.optim as optim
from train_fc import model_train,cost_function,model_test
import torch
import torch.nn as nn
import pdb
import os
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sys

def get_features(dataset,seq_len,scaler1=None,scaler2=None,freeze_frame_features=False):
    #specify input features and output features
    index_features=['id','match_id','possession']
    class_features1=['act']
    class_features2=['zone']
    scale_feature1=['deltaT','zone_s','zone_sg','zone_thetag'] 
    scale_feature2=['zone_deltax','zone_deltay']
    teammate_list = [f'player{i}_teammate' for i in range(22)]
    actor_list = [f'player{i}_actor' for i in range(22)]
    keeper_list = [f'player{i}_keeper' for i in range(22)]
    location_x_list = [f'player{i}_location_x' for i in range(22)]
    location_y_list = [f'player{i}_location_y' for i in range(22)]

    #get the dataset with only the features wanted

    if freeze_frame_features:
        features=index_features+class_features1+class_features2+scale_feature1+scale_feature2+teammate_list+actor_list+keeper_list+location_x_list+location_y_list
        dataset=dataset[features]
    else:
        features=index_features+class_features1+class_features2+scale_feature1+scale_feature2
        dataset=dataset[features]

    #features scaling
    if scaler1 is None:
        scaler1 = MinMaxScaler(feature_range=(0,1))
        scaler1.fit(dataset[scale_feature1])
    if scaler2 is None:
        scaler2 = MinMaxScaler(feature_range=(-1,1))
        scaler2.fit(dataset[scale_feature2])

    dataset.loc[:, scale_feature1] = scaler1.fit_transform(dataset[scale_feature1])
    dataset.loc[:, scale_feature2] = scaler2.fit_transform(dataset[scale_feature2]) 
    
    if freeze_frame_features:
        dataset.loc[:, location_x_list] = dataset[location_x_list] / 105
        dataset.loc[:, location_y_list] = dataset[location_y_list] / 68
    
    #encode features
    # act_decode_dict={0:'pass', 1:'dribble', 2:'end', 3:'shot', 4:'cross'}
    act_encode_dict = {'pass': 0, 'dribble': 1, 'end': 2, 'shot': 3, 'cross': 4}
    dataset.loc[:, 'act'] = dataset['act'].replace(act_encode_dict)
    if freeze_frame_features:
        dataset.loc[:, teammate_list] = dataset[teammate_list].fillna(False).astype(int)
        dataset.loc[:, actor_list] = dataset[actor_list].fillna(False).astype(int)
        dataset.loc[:, keeper_list] = dataset[keeper_list].fillna(False).astype(int)
        dataset.loc[:, location_x_list] = dataset[location_x_list].fillna(0)
        dataset.loc[:, location_y_list] = dataset[location_y_list].fillna(0)
    
    #define valid slice flag
    dataset.loc[:, "valid_slice_flag"]=False
    dataset.loc[:, "prediction_flag"]=False
    for _, group in dataset.groupby('match_id'):
        group_len = len(group)
        if group_len<seq_len:
            continue
        group_first_row_index=group.index[0]
        dataset.loc[group_first_row_index:group_first_row_index+group_len-seq_len-1,"valid_slice_flag"]=True
        dataset.loc[group_first_row_index+seq_len:group_first_row_index+group_len-1,"prediction_flag"]=True  
    return dataset,features,scaler1,scaler2

def get_weight(dataset,pass_weight=1.,dribble_weight=1.,end_weight=1.,shot_weight=1.,cross_weight=1.):
    zone=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    action=[0,1,2,3,4]

    weight_action_class=compute_class_weight(
                                            class_weight = "balanced",
                                            classes = action,
                                            y = dataset.act                                                    
                                        )

    weight_zone_class=compute_class_weight(
                                            class_weight = "balanced",
                                            classes = zone,
                                            y = dataset.zone                                                  
                                        )
    print('action weight before scaling',weight_action_class)
    #Adjust the weight of the classes
    # act_decode_dict={0: 'pass', 1: 'dribble', 2: 'end', 3: 'shot', 4: 'cross'}
    weight_action_class[0]=weight_action_class[0]*pass_weight
    weight_action_class[1]=weight_action_class[1]*dribble_weight
    weight_action_class[2]=weight_action_class[2]*end_weight
    weight_action_class[3]=weight_action_class[3]*shot_weight
    weight_action_class[4]=weight_action_class[4]*cross_weight
    print('action weight after scaling',weight_action_class)
    weight_action_class = torch.tensor(weight_action_class)
    weight_zone_class = torch.tensor(weight_zone_class)
    weight_deltaT=torch.tensor([1.])
    return weight_deltaT, weight_action_class, weight_zone_class

def worker_init_fn(worker_id):
    np.random.seed(seed_value + worker_id)
    torch.manual_seed(seed_value + worker_id)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_df_path","-d", help="train_df2.csv path",default="data/train_df2.csv")
    parser.add_argument("--valid_df_path","-v", help="valid_df2.csv path",default="data/valid_df2.csv")
    parser.add_argument("--test_df_path","-t", help="test_df2.csv path",default="data/test_df2.csv")
    parser.add_argument("--prediction_df_path","-p", help="analysis_df.csv path",default="data/analysis_df.csv")
    parser.add_argument('--output_dir', '-o', help='output directory', default='result/')
    parser.add_argument("--testing","-test", help="test the code on reduced size df", action='store_true', default=False)
    parser.add_argument("--freeze_frame_features","-ff", help="use freeze frame features", action='store_true', default=False)
    parser.add_argument("--prediction","-pred", help="use prediction", action='store_true', default=False)
    parser.add_argument("--parameter","-param", help="use parameter", action='store_true', default=False)
    parser.add_argument("--parameter_path","-param_path", help="parameter path",default="model/param.pt")
    parser.add_argument("--gat","-gat", help="use Graph Attention Network based Graph Neural Network for freeze_frame_features ", action='store_true', default=False)
    parser.add_argument('--other_lin1_out','-h1',help='hyperparameter other_lin1_out',default=6, type=int)
    parser.add_argument('--other_lin2_out','-h2',help='hyperparameter other_lin2_out',default=10, type=int)
    parser.add_argument('--hidden_dim','-hd',help='hyperparameter hidden_dim',default=1024, type=int)
    parser.add_argument('--gat_hidden_dim','-gat_hd',help='list of hyperparameter gat_hidden_dim',default=[110,55,20], type=int, nargs='+')
    parser.add_argument('--gat_activation_function','-gat_af',help='activation function for the GNN',default='relu', type=str)
    parser.add_argument('--end_weight','-ew',help='weight for end action',default=1.5, type=float)
    args = parser.parse_args()


    #load data  
    if args.testing:
        #replace train_df to train_df_reduced
        path=str(args.train_df_path)
        path=path.replace("train_df.csv","train_df2_reduced.csv")
        train_df=pd.read_csv(path)
        train_df=train_df.iloc[:10000,:]
        valid_df=pd.read_csv(args.valid_df_path).iloc[:1000,:]
    else:
        train_df=pd.read_csv(args.train_df_path)
        valid_df=pd.read_csv(args.valid_df_path)
        test_df=pd.read_csv(args.test_df_path)
        if args.prediction:
            prediction_df=pd.read_csv(args.prediction_df_path)
            prediction_out_df=prediction_df.copy()
    print('data loaded')

    #get the required features
    seq_len=40
    train_df=train_df.reset_index(drop=True) #reset index
    valid_df=valid_df.reset_index(drop=True) #reset index
    train_df,features,scaler1,scaler2=get_features(train_df,seq_len,scaler1=None,scaler2=None,freeze_frame_features=args.freeze_frame_features)
    valid_df,_,_,_=get_features(valid_df,seq_len,scaler1=scaler1,scaler2=scaler2,freeze_frame_features=args.freeze_frame_features)
    test_df,_,_,_=get_features(test_df,seq_len,scaler1=scaler1,scaler2=scaler2,freeze_frame_features=args.freeze_frame_features)
    if args.prediction:
        prediction_df,_,_,_=get_features(prediction_df,seq_len,scaler1=scaler1,scaler2=scaler2,freeze_frame_features=args.freeze_frame_features)
 
    #specify input features and output features
    index_features=['id','match_id','possession',"valid_slice_flag",'prediction_flag']
    input_features=[item for item in features if item not in index_features]
    target_features=['deltaT','zone','act']

    #Specify loss function weighting
    weight_deltaT, weight_action, weight_zone=get_weight(train_df,pass_weight=1.,dribble_weight=1.,end_weight=args.end_weight,shot_weight=1.,cross_weight=1.)
    
    #data loader
    if args.testing:
        batch_size=1000
    else:
        batch_size=100

    num_workers=4

    
    train_dataset=CustomDataset(train_df,seq_len,input_features,target_features,"valid_slice_flag")
    train_loader=DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,drop_last=True, worker_init_fn=worker_init_fn)
    valid_dataset=CustomDataset(valid_df,seq_len,input_features,target_features,"valid_slice_flag")
    valid_loader=DataLoader(valid_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False,drop_last=False)
    test_dataset=CustomDataset(test_df,seq_len,input_features,target_features,"valid_slice_flag")
    test_loader=DataLoader(test_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False,drop_last=False)
    if args.prediction:
        prediction_dataset=CustomDataset(prediction_df,seq_len,input_features,target_features,"valid_slice_flag")
        prediction_loader=DataLoader(prediction_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False,drop_last=False)

    print('data loader ready')
    print('-----training start-----')
    #model
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',device)
    #fixed hyperparameters
    action_emb_in=5
    zone_emb_in=20
    other_lin1_in=6
    other_lin2_in=22*5
    scale_grad_by_freq=True
    mutihead_attention=1 
    #tunable hyperparameters (encoding)
    action_emb_out=5 #fixed in this study
    zone_emb_out=20 #fixed in this study
    other_lin1_out=6 #[1, 6, 12] 6 is the default
    other_lin2_out=10 #[10, 20, 30] 10 is the default
    hidden_dim=1024 #[16, 64, 256, 1024, 2048] 1024 is the default
    gat_act=nn.ReLU()
    #featureset dependent hyperparameters
    input_features_len=action_emb_out+zone_emb_out+other_lin1_out+other_lin2_out
    print("input_features_len",input_features_len)

    


    print(args.gat_hidden_dim)
    model=NMSTPP(device, args.freeze_frame_features, action_emb_in, action_emb_out,zone_emb_in,zone_emb_out,other_lin1_in,other_lin1_out,other_lin2_in,other_lin2_out,scale_grad_by_freq
                 ,input_features_len, mutihead_attention, hidden_dim).to(device)

    optimiser  = optim.Adam(model.parameters(),lr=0.01,eps=1e-16)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,factor=.1,patience=3,verbose=False)
    #train 
    #trainning parameters
    if args.testing:
        epochs=2
    else:
        epochs=50
    deltaT_weight=10
    zone_weight=1
    action_weight=1
    weight_action_class=weight_action
    weight_zone_class=weight_zone
    early_stop_count_max=5
    if not args.parameter:
        trainloss_df,best_model_params,best_epoch=model_train(epochs,device,optimiser,model,train_loader,valid_loader,deltaT_weight,zone_weight,action_weight,weight_action_class,weight_zone_class,early_stop_count_max,scheduler=scheduler)

        #save model and trainloss_df
        path=str(args.output_dir)+"/prediction/"
        if not os.path.exists(path):
            os.makedirs(path)
        hidden_dim_str=str(args.gat_hidden_dim).replace(" ", "_")
        hidden_dim_str=hidden_dim_str.replace("[", "")
        hidden_dim_str=hidden_dim_str.replace("]", "")
        hidden_dim_str=hidden_dim_str.replace(",", "")
        time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        trainloss_df.to_csv(path+f"time_{time}_ew_{args.end_weight}_train_loss.csv",index=False)
        path=path+f"time_{time}_ew_{args.end_weight}.pt"
        torch.save(best_model_params,path)
        print('-----training end-----')
    else:
        #load model
        best_model_params=torch.load(args.parameter_path)


    if args.prediction:
        if not os.path.exists(str(args.output_dir)+"/prediction/"):
            os.makedirs(str(args.output_dir)+"/prediction/")
        #load the best model
        model.load_state_dict(best_model_params)
        #test
        print('-----prediction start-----')
        loss_rollingmean, lossRMSE_rollingmean, lossCEL_zone_rollingmean,lossCEL_action_rollingmean,pred_list,target_list=model_test(prediction_loader, model, optimiser,deltaT_weight,zone_weight,action_weight,weight_action_class,weight_zone_class,device,trainning=False)

        prediction_value= pred_list.cpu().detach().numpy()
        # turn into df
        pred_df=pd.DataFrame(prediction_value)
        pred_df.columns=['pred_deltaT','pred_zone1','pred_zone2','pred_zone3','pred_zone4','pred_zone5',
                        'pred_zone6','pred_zone7','pred_zone8','pred_zone9','pred_zone10','pred_zone11','pred_zone12',
                        'pred_zone13','pred_zone14','pred_zone15','pred_zone16','pred_zone17','pred_zone18',
                        'pred_zone19','pred_zone20','pred_action1','pred_action2','pred_action3','pred_action4','pred_action5']
        pred_df[pred_df['pred_deltaT']<=0]=0
        pred_id=prediction_df[prediction_df.prediction_flag==True]["id"].copy()
        # pdb.set_trace()
        pred_df['id']=pred_id.values
        #merge with the prediction_out_df
        prediction_out_df=pd.merge(prediction_out_df,pred_df,on='id',how='left')
        #save the prediction_out_df
        prediction_out_df.to_csv(str(args.output_dir)+"/prediction/prediction_df.csv",index=False)
        print('-----prediction end-----')