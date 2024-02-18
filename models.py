import torch.nn as nn
import torch
import math as m
import torch.nn.functional as F
import numpy as np
import copy
import math
import pprint
# ccc = torch.tensor(np.transpose(cc,(0,2,1)),dtype=torch.float32)
import os
import time

from spikingjelly.activation_based import functional, layer, surrogate, neuron

torch.set_printoptions(threshold=10_000)
#%%
tau_global = 1. / (1. - 0.5)

class NormalCNN(nn.Module):
    def __init__(self):
        super(NormalCNN,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=9, 
                  stride=2, padding=4, padding_mode='zeros')
        torch.nn.init.normal_(self.conv1.weight, mean = 0, std = m.sqrt(1/32))
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, 
                  stride=2, padding=2, padding_mode='zeros')
        torch.nn.init.normal_(self.conv2.weight, mean = 0, std = m.sqrt(1/32))
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, 
                  stride=2, padding=2, padding_mode='zeros')
        torch.nn.init.normal_(self.conv3.weight, mean = 0, std = m.sqrt(1/32))
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, 
                  stride=2, padding=1, padding_mode='zeros')
        torch.nn.init.normal_(self.conv4.weight, mean = 0, std = m.sqrt(1/32))
        self.maxpool = nn.MaxPool1d(kernel_size=8,stride=2)
        self.fc1 = nn.Linear(1024,49)
        
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(64)
        self.norm3 = nn.BatchNorm1d(64)
        self.norm4 = nn.BatchNorm1d(64)
        
    def forward(self,x):
        #print(x.shape)
        # 16 64 500
        out =  self.maxpool(F.relu(self.norm1(self.conv1(x))))
        # 16 64 122
        out = F.relu(self.norm2(self.conv2(out)))
        # 16 64 61
        out = F.relu(self.norm3(self.conv3(out)))
        # 16 64 31
        out = F.relu(self.norm4(self.conv4(out)))
        # 16 64 16
        out = torch.flatten(out,1)
        # 16 1024
        #print(out.shape)
        #out = self.fc1(out)
        #print(out.shape)
        #print(sum(output_list))
        return out
    
class transCNN(nn.Module):
    def __init__(self):
        super(transCNN,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=500, out_channels=512, kernel_size=3, 
                  stride=1, padding=1, padding_mode='zeros')
        torch.nn.init.normal_(self.conv1.weight, mean = 0, std = m.sqrt(1/32))
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, 
                  stride=1, padding=1, padding_mode='zeros')
        torch.nn.init.normal_(self.conv2.weight, mean = 0, std = m.sqrt(1/32))
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, 
                  stride=1, padding=1, padding_mode='zeros')
        torch.nn.init.normal_(self.conv3.weight, mean = 0, std = m.sqrt(1/32))
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, 
                  stride=1, padding=1, padding_mode='zeros')
        torch.nn.init.normal_(self.conv4.weight, mean = 0, std = m.sqrt(1/32))
        self.maxpool = nn.MaxPool1d(kernel_size=8,stride=2)
        self.fc1 = nn.Linear(1024,52)
        
        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(256)
        self.norm4 = nn.BatchNorm1d(256)
        
    def forward(self,x):
        #print(x.shape)
        out = self.norm1(self.conv1(x))
        #print(out.shape)
        #print(x.shape)
        #print(x.shape)
        #print(x.shape)
        out = self.norm2(self.conv2(out))
        #print(out.shape)
        out = self.norm3(self.conv3(out))
        #print(out.shape)
        out = self.norm4(self.conv4(out))
        #print(out.shape)
        out = torch.flatten(out,1)
        #print(out.shape)
        #out = self.fc1(out)
        #print(out.shape)
        #print(out.shape)
        #print(sum(output_list))
        return out
    
class DF(nn.Module):
    def __init__(self):
        super(DF, self).__init__()
        self.cnn = NormalCNN()
        self.tcnn = transCNN()
        
        self.fc1 = nn.Linear(3584,512)
        self.fc2 = nn.Linear(512,52)
    def forward(self,x,y):
        #input (batch, 10, 500)
        x = self.cnn(x)
        y = self.tcnn(y)
        out = torch.cat((x,y),1)
        #print(x.shape)
        out = F.relu(self.fc1(out))
        out = F.dropout(out,0.2)
        #x = F.relu(self.fcc(x))
        #x = F.dropout(x,0.2)
        #x = F.relu(self.fcc2(x))
        #x = F.dropout(x,0.2)
        x = F.relu(self.fc2(out))
        
        return out
    


class SNN(nn.Module):
    def __init__(self):
        super(SNN,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=64, kernel_size=9, 
                  stride=2, padding=4, padding_mode='zeros')
        torch.nn.init.normal_(self.conv1.weight, mean = 0, std = m.sqrt(1/32))
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, 
                  stride=2, padding=2, padding_mode='zeros')
        torch.nn.init.normal_(self.conv2.weight, mean = 0, std = m.sqrt(1/32))
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, 
                  stride=2, padding=2, padding_mode='zeros')
        torch.nn.init.normal_(self.conv3.weight, mean = 0, std = m.sqrt(1/32))
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, 
                  stride=2, padding=1, padding_mode='zeros')
        torch.nn.init.normal_(self.conv4.weight, mean = 0, std = m.sqrt(1/32))
        self.lif1 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.lif3 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.lif4 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.maxpool = nn.MaxPool1d(kernel_size=8,stride=2)
        self.fc1 = nn.Linear(1024,52)
        
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(64)
        self.norm3 = nn.BatchNorm1d(64)
        self.norm4 = nn.BatchNorm1d(64)
        
    def forward(self,x):
        output_list = []
        #print(x.shape)
        x =  self.maxpool(self.lif1(self.norm1(self.conv1(x))))
        #print(x.shape)
        #print(x.shape)
        #print(x.shape)
        for t in range(100):
            out = self.lif2(self.norm2(self.conv2(x)))
            out = self.lif3(self.norm3(self.conv3(out)))
            out = self.lif4(self.norm4(self.conv4(out)))
            out = torch.flatten(out,1)
            out = self.fc1(out)
            #print(out.shape)
            output_list.append(out)
        #print(sum(output_list))
        return output_list
        
        
    

class CNNmodel(nn.Module):
    def __init__(self,stride=2, padding_mode='zeros'):
        super(CNNmodel,self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=64, kernel_size=9, 
                  stride=stride, padding=4, padding_mode='zeros')
        torch.nn.init.normal_(self.conv1.weight, mean = 0, std = m.sqrt(1/32))
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, 
                  stride=stride, padding=2, padding_mode='zeros')
        torch.nn.init.normal_(self.conv2.weight, mean = 0, std = m.sqrt(1/32))
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, 
                  stride=stride, padding=2, padding_mode='zeros')
        torch.nn.init.normal_(self.conv3.weight, mean = 0, std = m.sqrt(1/32))
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, 
                  stride=stride, padding=1, padding_mode='zeros')
        torch.nn.init.normal_(self.conv4.weight, mean = 0, std = m.sqrt(1/32))
        
        self.norm1 = nn.BatchNorm1d(64,eps=1e-6,momentum=0.05)
        self.norm2 = nn.BatchNorm1d(64,eps=1e-6,momentum=0.05)
        self.norm3 = nn.BatchNorm1d(64,eps=1e-6,momentum=0.05)
        self.norm4 = nn.BatchNorm1d(64,eps=1e-6,momentum=0.05)
        
        self.maxpool = nn.MaxPool1d(kernel_size=8,stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(2)
        
    def forward(self,x):
        #input (batch, 10, 20) = (batch, emg_ch, unit_time)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        # (batch, 64, 10) = (batch_size, num_filters, 10)
        #print(x.shape)
        
        #x = F.dropout(x,0.2)
        x = self.maxpool(x)
        #x = self.avgpool(x)
        # (batch, 64,2)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        # (batch, 64, 1)
        #print(x.shape)
        x = F.relu(self.conv3(x))
        # (batch, 64, 1)
        #print(x.shape)
        x = F.relu(self.conv4(x))
        # (batch, 64, 1)
        #print(x.shape)
        x= torch.flatten(x,1)
        # (batch, 64)
        #print(x.shape)
        
        return x    

class Bi_LSTMModel(nn.Module):
    def __init__(self):
        super(Bi_LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = 200

        # Number of hidden layers
        self.num_layers = 2

        # batch_first=True causes input/output tensors to be of shape
        # (z, seq_dim, feature_dim) (25,10,200)
        # (seq_length, batch, feature dim) (25, 32, 64)
        self.lstm = nn.LSTM(64, self.hidden_dim, self.num_layers, dropout=0.2093,bidirectional=True, batch_first=True)
        #self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, dropout=0.2093,bidirectional=True, batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool1d(20)

    def forward(self, x):
        
        #print(x.shape)
        # input = (25, batch, 64)
        
        
        # Initialize hidden state with zeros
        # (4,batch,200)
        h0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_dim,device=x.device).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_dim,device=x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        #print(cn)
        
        #(batch, 25, 400)
        #print(out.shape)
        
        #out = self.avgpool(out)
        out = torch.flatten(out,1)
        ##print(out.shape)
        #print('e')
        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        #out = self.fc(out[:, -1, :]).squeeze()
        # out.size() --> 100, 10
        return out
    
class EMGhandnet(nn.Module):
    def __init__(self):
        super(EMGhandnet, self).__init__()
        self.cnn = CNNmodel()
        self.lstm = Bi_LSTMModel()
        
        self.fc1 = nn.Linear(10000,512)
        self.fc2 = nn.Linear(512,52)
    def forward(self,x):
        #input (batch, 25,10, 20)
        temp = [ self.cnn(x[:,t,:,:]) for t in range(x.size(1))]
        """
        for t in range(x.size(1)):
            temp.append(self.cnn(x[:,t,:,:]))
        """
        x = torch.stack(temp,1)
        # (batch,time, features) = (batch, 25, 64)
        #print(x.shape)
        x = self.lstm(x)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.2)
        #x = F.relu(self.fcc(x))
        #x = F.dropout(x,0.2)
        #x = F.relu(self.fcc2(x))
        #x = F.dropout(x,0.2)
        x = F.relu(self.fc2(x))
        
        return x
    
#%% transformer
"config"
CONFIG = {
    'd_embed' : 500,
    'input_dim' : 500,
    'hid_dim':500,
    'n_layers':8,
    'n_heads':5,
    'pf_dim':1024,
    'dropout':0.2,
    'device':'cuda:0',
    'max_length' : 50,
    }


class TFModel(nn.Module):
    def __init__(self, input_dim=CONFIG['input_dim'], hid_dim=CONFIG['hid_dim'], n_layers=CONFIG['n_layers'],
                 n_heads=CONFIG['n_heads'], pf_dim=CONFIG['pf_dim'], dropout=CONFIG['dropout'],
                 device=CONFIG['device'], max_length = CONFIG['max_length']):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer() 
                                     for _ in range(n_layers)])
        self.seg_embedding = nn.Embedding(10, 20)
        self.dropout = nn.Dropout(0.2)
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.posenc = PositionalEncoding()
        self.linear1 = nn.Linear(5000,52)
        self.linear2 = nn.Linear(2500,768)
        self.linear3 = nn.Linear(768,52)
        self.dropout2 = nn.Dropout(0.5)
        
    def forward(self, src):
        #src = [batch size, src len]
        # = [batch_size = 16, src_len = 25]
        #src_mask = [batch size, 1, 1, src len]
        # = [ batch_size = 16, 1, 1, src_len = 25]
        #print(src.shape)
        # [ 16, 25, 200] without embedding
        # [ 16, 1, 1, 25]
        #seg = self.seg_embedding(torch.arange(0,10,1).to(self.device)).flatten()
        batch_size = src.shape[0]
        # 16
        src_len = src.shape[1]
        # 25
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos = [batch size, src len]
        src = self.dropout((src * self.scale) + self.posenc(pos))
        #src = self.dropout(src * self.scale)
        #src = (src * self.scale) + self.posenc(pos)
        #print(self.posenc(pos)[1])
        #print(self.posenc(pos).shape)
        #print(src.shape)
        
        #src = [batch size, src len, hid dim]
        
        for elayer in self.layers:
            src = elayer(src)
            
        #src = [batch size, src len, hid dim]
        feature = src.flatten(1)
        src = self.linear1(feature)
        #src = self.dropout2(F.gelu(self.linear1(feature)))
        #src = self.dropout2(F.gelu(self.linear2(src)))
        #src = self.linear3(src)
        #print(src.shape)
        return src ,feature

class EncoderLayer(nn.Module):
    def __init__(self,hid_dim=CONFIG['hid_dim'], n_heads=CONFIG['n_heads'],
                 pf_dim=CONFIG['pf_dim'], dropout=CONFIG['dropout'], device=CONFIG['device']):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim=CONFIG['hid_dim'], n_heads=CONFIG['n_heads'],
                 dropout=CONFIG['dropout'], device=CONFIG['device']):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(F.gelu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_embed=CONFIG['d_embed'], max_len=CONFIG['max_length'], device=CONFIG['device']):
        super(PositionalEncoding, self).__init__()
        self.device=device
        encoding = torch.zeros(max_len, d_embed)
        # 25, 200
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)
        # [1,25,200]


    def forward(self, x):
        _, seq_len= x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        #out = x + pos_embed
        return pos_embed.to(self.device)


#%% transformer2
"config"
TF2_CONFIG = {
    'd_embed' : 288,
    'input_dim' : 288,
    'hid_dim':288,
    'n_layers':8,
    'n_heads':4,
    'pf_dim':1024,
    'dropout':0.2,
    'device':'cuda:0',
    'max_length' : 25,
    }


class TFModel2(nn.Module):
    def __init__(self, input_dim=TF2_CONFIG['input_dim'], hid_dim=TF2_CONFIG['hid_dim'], n_layers=TF2_CONFIG['n_layers'],
                 n_heads=TF2_CONFIG['n_heads'], pf_dim=TF2_CONFIG['pf_dim'], dropout=TF2_CONFIG['dropout'],
                 device=TF2_CONFIG['device'], max_length = TF2_CONFIG['max_length']):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer3() 
                                     for _ in range(n_layers)])
        self.seg_embedding = nn.Embedding(10, 20)
        self.dropout = nn.Dropout(0.2)
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.posenc = PositionalEncoding3()
        self.linear1 = nn.Linear(7200,49)
        self.linear2 = nn.Linear(2048,49)
        self.linear3 = nn.Linear(1024,49)
        self.dropout2 = nn.Dropout(0.5)
        
    def forward(self, src):
        #src = [batch size, src len]
        # = [batch_size = 16, src_len = 25]
        #src_mask = [batch size, 1, 1, src len]
        # = [ batch_size = 16, 1, 1, src_len = 25]
        #print(src.shape)
        # [ 16, 25, 200] without embedding
        # [ 16, 1, 1, 25]
        #seg = self.seg_embedding(torch.arange(0,10,1).to(self.device)).flatten()
        #print(src.shape)
        batch_size = src.shape[0]
        # 16
        src_len = src.shape[1]
        # 25
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos = [batch size, src len]
        src = self.dropout((src * self.scale) + self.posenc(pos))
        #src = self.dropout(src * self.scale)
        #src = (src * self.scale) + self.posenc(pos)
        #print(self.posenc(pos)[1])
        #print(self.posenc(pos).shape)
        #print(src.shape)
        
        #src = [batch size, src len, hid dim]
        
        for elayer in self.layers:
            src = elayer(src)
            
        #src = [batch size, src len, hid dim]
        feature = src.flatten(1)
        src = self.linear1(feature)
        #src = self.dropout2(F.gelu(self.linear1(feature)))
        #src = self.dropout2(F.gelu(self.linear2(src)))
        #src = self.linear2(src)
        #src = self.linear3(src)
        #src = self.linear3(src)
        #print(src.shape)
        return src ,feature

class EncoderLayer3(nn.Module):
    def __init__(self,hid_dim=TF2_CONFIG['hid_dim'], n_heads=TF2_CONFIG['n_heads'],
                 pf_dim=TF2_CONFIG['pf_dim'], dropout=TF2_CONFIG['dropout'], device=TF2_CONFIG['device']):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer3(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer3(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src
    
class MultiHeadAttentionLayer3(nn.Module):
    def __init__(self, hid_dim=TF2_CONFIG['hid_dim'], n_heads=TF2_CONFIG['n_heads'],
                 dropout=TF2_CONFIG['dropout'], device=TF2_CONFIG['device']):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
    
class PositionwiseFeedforwardLayer3(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(F.gelu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
    
class PositionalEncoding3(nn.Module):
    def __init__(self, d_embed = TF2_CONFIG['d_embed'], max_len=TF2_CONFIG['max_length'], device=TF2_CONFIG['device']):
        super(PositionalEncoding3, self).__init__()
        self.device=device
        encoding = torch.zeros(max_len, d_embed)
        # 25, 200
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)
        # [1,25,200]


    def forward(self, x):
        _, seq_len= x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        #out = x + pos_embed
        return pos_embed.to(self.device)
