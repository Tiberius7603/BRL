from torch.utils.data import Dataset
import numpy as np
import torch
import copy


class Nina1Dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        target_row = self.dataframe.iloc[idx]
        data = target_row['normalized'][0:500]
        means = target_row['normalized_mean']
        #means = target_row['total_means']
        "Zero-Padding"
        if len(data)<500:
            data = np.concatenate((data,np.zeros(((500-len(data)),10))),axis=0)
        #mean = np.mean(data, axis =0)
        #std = np.mean(data, axis=0)
        #data = (data-mean)/std
        "Division data by time-segment"
        #noised = data + 0.05*np.random.normal(-means,stds,data.shape)
        noised = data + 1*np.random.uniform(-means,means,data.shape)
        noised_data = torch.tensor(np.transpose(noised.reshape((20,25,10)),(0,2,1)),dtype=torch.float)
        noised_data = noised_data.flatten(1)
        input_data = torch.tensor(np.transpose(data.reshape((20,25,10)),(0,2,1)),dtype=torch.float)
        input_data = input_data.flatten(1)
        
        label = torch.tensor(target_row['stimulus'],dtype=torch.long)
        #print(label)
        
        return {'input_data': input_data,
                'label': label,
                'noised': noised_data}
    
class Nina2(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        target_row = self.dataframe.iloc[idx]
        #print(target_row)
        data = target_row['normalized'][0:600]
        means = target_row['normalized_mean']
        #std = target_row['normalized_std']
        "Zero-Padding"
        if len(data)<600:
            data = np.concatenate((data,np.zeros(((600-len(data)),12))),axis=0)
        
        #data = ((data-min_v)/(max_v-min_v)*3-1.5)
        #data = 4*(data-min_v)/(max_v-min_v)
        "Division data by time-segment"
        noised = data + 0.01*np.random.uniform(-means,means,data.shape)
        #noised = data + 0.5*np.random.normal(means,std,data.shape)
        noised_data = torch.tensor(np.transpose(noised.reshape((25,24,12)),(0,2,1)),dtype=torch.float)
        noised_data = noised_data.flatten(1)
        input_data = torch.tensor(np.transpose(data.reshape((25,24,12)),(0,2,1)),dtype=torch.float)
        input_data = input_data.flatten(1)
        label = torch.tensor(target_row['stimulus'],dtype=torch.long)
        
        return {'input_data': input_data,
                'label': label,
                'noised': noised_data}  
