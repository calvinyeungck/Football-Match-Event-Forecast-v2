import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np


class NMSTPP(nn.Module):
    def __init__(self, device, freeze_frame_features, action_emb_in, action_emb_out,zone_emb_in,zone_emb_out,other_lin1_in,other_lin1_out,other_lin2_in,other_lin2_out,scale_grad_by_freq
                 ,input_features_len, mutihead_attention, hidden_dim, gat=False,gat_act=nn.ReLU(),gat_hidden_dim=[1,1,1]):  #pick up all specification vars from the global environment
        super(NMSTPP, self).__init__()    
        self.device = device
        self.freeze_frame_features = freeze_frame_features
        # for action one-hot
        self.emb_act = nn.Embedding(action_emb_in,action_emb_out,scale_grad_by_freq=scale_grad_by_freq)
        # for zone one-hot
        self.emb_zone = nn.Embedding(zone_emb_in,zone_emb_out,scale_grad_by_freq=scale_grad_by_freq)
        # for continuous features
        self.lin0 = nn.Linear(other_lin1_in,other_lin1_out,bias=True) 
        self.lin1 = nn.Linear(other_lin2_in,other_lin2_out,bias=True)
        self.gat=gat
        if self.gat:
            self.gat_layer = GAT(in_dim=5, out_dim=5, n_heads=1)
            self.gat_act = gat_act
            self.gat_lin0 = nn.Linear(22*5,gat_hidden_dim[0],bias=True)
            self.gat_lin1 = nn.Linear(gat_hidden_dim[0],gat_hidden_dim[1],bias=True)
            self.gat_lin2 = nn.Linear(gat_hidden_dim[1],gat_hidden_dim[2],bias=True)
            self.gat_dropout = nn.Dropout(0.1)


        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_features_len,nhead=mutihead_attention,batch_first=True,dim_feedforward=hidden_dim).to(self.device)

        self.lin_relu = nn.Linear(input_features_len,input_features_len)
        self.lin_deltaT = nn.Linear(input_features_len,1)
        self.lin_zone = nn.Linear(input_features_len+1,20)
        self.lin_action = nn.Linear(input_features_len+1+20,5)
        self.NN_deltaT = nn.ModuleList()
        self.NN_zone = nn.ModuleList()
        self.NN_action = nn.ModuleList()
        for num_layer_deltaT in range(1):
            self.NN_deltaT.append(nn.Linear(input_features_len,input_features_len))
        for num_layer_zone in range(1):
            self.NN_zone.append(nn.Linear(input_features_len+1,input_features_len+1))
        for num_layer_action in range(2):
            self.NN_action.append(nn.Linear(input_features_len+1+20,input_features_len+1+20))

        print(self)        

    def positional_encoding(self, src):
        pos_encoding = torch.zeros_like(src)
        seq_len = pos_encoding.shape[0]
        d_model = pos_encoding.shape[1]
        for i in range(d_model):
            for pos in range(seq_len):
                if i % 2 == 0:
                    pos_encoding[pos,i] = np.sin(pos/100**(2*i/d_model))
                else:
                    pos_encoding[pos,i] = np.cos(pos/100**(2*i/d_model))
        return pos_encoding.float()

    def forward(self, X):

        feed_action=X[:,:,0]
        feed_zone=X[:,:,1]
        feed_other_deltaT=X[:,:,2:8]
        if self.freeze_frame_features:
            feed_freeze_frame=X[:,:,8:]
        
        X_act = self.emb_act(feed_action.int())
        X_zone = self.emb_zone(feed_zone.int())
        feed_other_deltaT= self.lin0(feed_other_deltaT.float())
        if self.freeze_frame_features:
            if self.gat:
                #convert to (batch_size, num_nodes, out_dim)
                feed_freeze_frame=feed_freeze_frame.float()
                batch_size, seq_len, _ = feed_freeze_frame.size()

                gat_outputs = torch.zeros(batch_size, seq_len, 22* 5, dtype=feed_freeze_frame.dtype, device=feed_freeze_frame.device)
                #loop over the sequence length
                for i in range(seq_len):
                    #convert seq i into (batch_size, num_nodes, out_dim)
                    temp = feed_freeze_frame[:, i, :]
                    temp = temp.reshape(batch_size, 22, 5)
                    temp = self.gat_layer(temp)
                    temp = temp.reshape(batch_size, 22 * 5)
                    # Store the result in the new tensor
                    gat_outputs[:, i, :] = temp
                if self.gat_act!="None":
                    feed_freeze_frame = gat_outputs
                    feed_freeze_frame = self.gat_act(self.gat_lin0(feed_freeze_frame))
                    feed_freeze_frame = self.gat_dropout(feed_freeze_frame)
                    feed_freeze_frame = self.gat_act(self.gat_lin1(feed_freeze_frame))
                    feed_freeze_frame = self.gat_dropout(feed_freeze_frame)
                    feed_freeze_frame = self.gat_act(self.gat_lin2(feed_freeze_frame))
                else:
                    feed_freeze_frame = gat_outputs
                    feed_freeze_frame = self.gat_lin0(feed_freeze_frame)
                    feed_freeze_frame = self.gat_lin1(feed_freeze_frame)
                    feed_freeze_frame = self.gat_lin2(feed_freeze_frame)
            else:
                feed_freeze_frame= self.lin1(feed_freeze_frame.float())
                
        if self.freeze_frame_features:
            X_cat = torch.cat((X_act,X_zone,feed_other_deltaT,feed_freeze_frame),2)
        else:
            X_cat = torch.cat((X_act,X_zone,feed_other_deltaT),2)

        X_cat = X_cat.float()
     
        src = X_cat+ self.positional_encoding(X_cat).to(self.device)
     
        src=src.float()
        
        X_cat_seqnet = self.encoder_layer(src)
        x_relu=self.lin_relu(X_cat_seqnet[:,-1,:])

        model_deltaT=x_relu
        for layer in self.NN_deltaT[:]:
            model_deltaT=layer(model_deltaT)
        model_deltaT=self.lin_deltaT(model_deltaT)
        #take the absolute value of deltaT
        # model_deltaT=torch.abs(model_deltaT)

        features_zone=torch.cat((model_deltaT, x_relu),1)
        model_zone=features_zone
        for layer in self.NN_zone[:]:
            model_zone=layer(model_zone)
        model_zone=self.lin_zone(model_zone)
        #take the softmax of zone
        # model_zone=torch.softmax(model_zone,dim=1)

        features_action=torch.cat((model_zone,model_deltaT, x_relu),1)
        model_action=features_action
        for layer in self.NN_action[:]:
            model_action=layer(model_action)
        model_action=self.lin_action(model_action)
        #take the softmax of action
        # model_action=torch.softmax(model_action,dim=1)

        out=torch.cat((model_deltaT,model_zone,model_action),1)

        return out

class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads):
        super(GAT, self).__init__()
        self.n_heads = n_heads
        self.out_dim=out_dim
        self.linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(n_heads)
        ])
        self.attentions = nn.ParameterList([
            nn.Parameter(torch.zeros(size=(2*out_dim, 1))) for _ in range(n_heads)
        ])

    def forward(self, x):
        batch_size, num_nodes, input_dim = x.size()
        outputs = []
        for i in range(self.n_heads):
            h_out = self.linears[i](x)  #calculate the W weight matrix for each head
            a_input = torch.cat([h_out.repeat(1, 1, num_nodes).view(batch_size, num_nodes*num_nodes, -1),
                                 h_out.repeat(1, num_nodes, 1)], dim=-1).view(batch_size, num_nodes, num_nodes, 2*self.out_dim) #create the concatenated vector for node i and node j
            e = F.leaky_relu(torch.matmul(a_input, self.attentions[i]).squeeze(3), negative_slope=0.2) #mutiply with the "a" vector and apply the leaky relu
            attention = e
            attention = F.softmax(attention, dim=-1) #apply the softmax to the attention scores such that they sum to 1
            attention = F.dropout(attention, p=0.6, training=self.training) 
            h_prime = torch.matmul(attention, h_out)
            outputs.append(h_prime)

        outputs = torch.stack(outputs, dim=1)
        h_out = torch.mean(outputs, dim=1)
        return h_out