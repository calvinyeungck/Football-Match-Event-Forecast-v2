import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pdb
import time

def model_train(epochs,device,optimiser,model,train_loader,valid_loader,deltaT_weight,zone_weight,action_weight,weight_action_class,weight_zone_class,early_stop_count_max=5,scheduler=None):
    torch.cuda.empty_cache(); import gc; gc.collect()
    trainloss_df_list=[]
    early_stop_count=0
    early_stop_count_max=early_stop_count_max
    early_stop_treshold=np.inf
    for t in range(epochs):
        torch.cuda.empty_cache(); import gc; gc.collect()
        print(f"Epoch {t}\n-------------------------------")
        trn_L, trn_MSEL, trn_CEL_zone,trn_CEL_action= epoch(train_loader, model, optimiser,deltaT_weight,zone_weight,action_weight,weight_action_class,weight_zone_class,device,trainning=True)  
        
        with torch.no_grad():
            torch.cuda.empty_cache(); import gc; gc.collect()
            val_L, val_MSEL, val_CEL_zone, val_CEL_action=epoch(valid_loader, model, optimiser,deltaT_weight,zone_weight,action_weight,weight_action_class,weight_zone_class,device,trainning=False)

            epochloss = pd.DataFrame({
                                        "epoch": [t],
                                        "trn_L": [trn_L.cpu().detach().numpy()],
                                        "trn_CEL_zone": [trn_CEL_zone.cpu().detach().numpy()],
                                        "trn_CEL_action": [trn_CEL_action.cpu().detach().numpy()],
                                        "trn_MSEL": [trn_MSEL.cpu().detach().numpy()],
                                        "val_L": [val_L.cpu().detach().numpy()],
                                        "val_CEL_zone": [val_CEL_zone.cpu().detach().numpy()],
                                        "val_CEL_action": [val_CEL_action.cpu().detach().numpy()],
                                        "val_MSEL": [val_MSEL.cpu().detach().numpy()]
                                    })
        trainloss_df_list.append(epochloss)
        if scheduler is not None:
            scheduler.step(val_L)

        if val_L < early_stop_treshold:
            early_stop_count=0
            best_model_params = model.state_dict()
            best_epoch=t
            early_stop_treshold=val_L
        else:
            early_stop_count=early_stop_count+1

        if early_stop_count>early_stop_count_max:
            print("-----Early stop-----")
            break


    trainloss_df=pd.concat(trainloss_df_list,ignore_index=True)
    return trainloss_df,best_model_params,best_epoch


def epoch(dataloader, model, optimiser,deltaT_weight,zone_weight,action_weight,weight_action_class,weight_zone_class,device,trainning=False):
    if trainning:
        model.train()     #turn training off if (val or test)
    else:
        model.eval()
    loss_rollingmean = 0
    lossRMSE_rollingmean = 0
    lossCEL_zone_rollingmean = 0
    lossCEL_action_rollingmean = 0
    start_time = time.time()
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        Loss,RMSE_deltaT,CEL_zone,CEL_action = cost_function(Y,pred,deltaT_weight,zone_weight,action_weight,weight_action_class,weight_zone_class,device)
        loss_rollingmean = loss_rollingmean+(Loss-loss_rollingmean)/(1+batch)
        lossRMSE_rollingmean = lossRMSE_rollingmean+(RMSE_deltaT-lossRMSE_rollingmean)/(1+batch)
        lossCEL_zone_rollingmean =  lossCEL_zone_rollingmean+(CEL_zone-lossCEL_zone_rollingmean)/(1+batch)
        lossCEL_action_rollingmean = lossCEL_action_rollingmean+(CEL_action-lossCEL_action_rollingmean)/(1+batch)
       
        
        if trainning:
            if batch%100==0:
                print("batch: ",batch," Loss: ",loss_rollingmean.item()," RMSE_deltaT: ",RMSE_deltaT.item()," CEL_zone: ",CEL_zone.item()," CEL_action: ",CEL_action.item())
            optimiser.zero_grad()
            Loss.backward()
            optimiser.step()
    end_time = time.time()
    time_required=(end_time-start_time)/ 60
    print('Time:',time_required,'Training: ',trainning ,'epoch loss: ',loss_rollingmean.item(),' epoch RMSE_deltaT: ',lossRMSE_rollingmean.item(),' epoch CEL_zone: ',lossCEL_zone_rollingmean.item(),' epoch CEL_action: ',lossCEL_action_rollingmean.item())
    return loss_rollingmean, lossRMSE_rollingmean, lossCEL_zone_rollingmean,lossCEL_action_rollingmean

def cost_function(y,y_head,deltaT_weight,zone_weight,action_weight,weight_action_class,weight_zone_class,device):
    y_deltaT=y[:,0].float()
    y_zone=y[:,1].long()
    y_action=y[:,2].long()
    y_head_deltaT=y_head[:,0].float() 
    y_head_zone=y_head[:,1:21]
    y_head_action=y_head[:,21:]
    if weight_action_class is not None:
        weight_action_class=weight_action_class.float().to(device)
    CEL_action = nn.CrossEntropyLoss(weight=weight_action_class,reduction ="none")
    Yhat_CEL_action  = torch.mean(CEL_action(y_head_action,y_action)) 
    if weight_zone_class is not None:
        weight_zone_class=weight_zone_class.float().to(device)
    CEL_zone = nn.CrossEntropyLoss(weight=weight_zone_class,reduction="none")
    Yhat_CEL_zone  = torch.mean(CEL_zone( y_head_zone,y_zone)) 
    Yhat_RMSE_deltaT= torch.mean((y_deltaT-y_head_deltaT)**2)**0.5
    Loss=  Yhat_RMSE_deltaT*deltaT_weight +Yhat_CEL_zone*zone_weight+Yhat_CEL_action*action_weight
    return Loss,Yhat_RMSE_deltaT,Yhat_CEL_zone,Yhat_CEL_action


def model_test(dataloader, model, optimiser,deltaT_weight,zone_weight,action_weight,weight_action_class,weight_zone_class,device,trainning=False):
    with torch.no_grad():
        model.eval()
        loss_rollingmean = 0
        lossRMSE_rollingmean = 0
        lossCEL_zone_rollingmean = 0
        lossCEL_action_rollingmean = 0
        start_time = time.time()
        pred_list=[]
        target_list=[]
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            Loss,RMSE_deltaT,CEL_zone,CEL_action = cost_function(Y,pred,deltaT_weight,zone_weight,action_weight,weight_action_class,weight_zone_class,device)
            loss_rollingmean = loss_rollingmean+(Loss-loss_rollingmean)/(1+batch)
            lossRMSE_rollingmean = lossRMSE_rollingmean+(RMSE_deltaT-lossRMSE_rollingmean)/(1+batch)
            lossCEL_zone_rollingmean =  lossCEL_zone_rollingmean+(CEL_zone-lossCEL_zone_rollingmean)/(1+batch)
            lossCEL_action_rollingmean = lossCEL_action_rollingmean+(CEL_action-lossCEL_action_rollingmean)/(1+batch)
            pred_list.append(pred)
            target_list.append(Y)
        pred_list=torch.cat(pred_list,dim=0)
        target_list=torch.cat(target_list,dim=0)
        end_time = time.time()
        time_required=(end_time-start_time)/ 60
        print('Time:',time_required,'Training: ',trainning ,'epoch loss: ',loss_rollingmean.item(),' epoch RMSE_deltaT: ',lossRMSE_rollingmean.item(),' epoch CEL_zone: ',lossCEL_zone_rollingmean.item(),' epoch CEL_action: ',lossCEL_action_rollingmean.item())
        return loss_rollingmean, lossRMSE_rollingmean, lossCEL_zone_rollingmean,lossCEL_action_rollingmean,pred_list,target_list
