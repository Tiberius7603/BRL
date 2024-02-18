import torch
from torch import optim
from torch.utils.data import DataLoader
import emgdataset
from tqdm import tqdm
import gc
import numpy as np
import os
import random
#from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import models
import methods
import matplotlib.pyplot as plt
import json
from torchsummaryX import summary
#%% seed
def seed_everything(seed: int = 21):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

        

#%% Argparse
parser = argparse.ArgumentParser(description='Using for training on emg dataset')

parser.add_argument('--model-name', type=str,  choices=["EMGhandnet","BERT","SNN","CNN","DF","TF","UCI","TF2"],
                    default="EMGhandnet", help="training model")
parser.add_argument('--type',type=str,choices=['default','sub-wise','leave-one-out'],default = 'default')
parser.add_argument('--epochs', type=int,  help='an integer for numbers of epochs', default = 90)
parser.add_argument('--learning-rate', type=float,  help='an float number of learning rate', default = 1e-2)
parser.add_argument('--batch-size', type=int,  help='integer for size of mini batch', default = 16)
parser.add_argument('--method', type=str, choices=["default", "adv","encoder","SNN","DF"],default="default", help="training method")
parser.add_argument('--dataset', type=int, default=1, help="Ninaprodb dataset type")
parser.add_argument('--train-dir', type=str,  help='dir of train_data', default = './ninaprodb2train3.pkl')
parser.add_argument('--test-dir', type=str,  help='dir of test_data', default = './ninaprodb2test3.pkl')

args = parser.parse_args()


#%% pickle 불러오기





#%% Main
def main(args: argparse.Namespace):
    "set up"
    Dataset = {1:emgdataset.Nina1Dataset,2:emgdataset.Nina2}[args.dataset]
    train_step = {"default": methods.train_step,"adv": methods.adversarial_train_step,
                  "SNN": methods.SNN_train_step}[args.method]
    model = ({"EMGhandnet": models.EMGhandnet,"SNN":models.SNN,
              "CNN":models.NormalCNN, "TF":models.TFModel, "TF2":models.TFModel2}[args.model_name]()).to(device) 
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    train = pd.read_pickle(args.train_dir)
    eval_data = pd.read_pickle(args.test_dir)
    best_loss = 5e10
    ba = 0
    train_dataset = Dataset(train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    eval_dataset = Dataset(eval_data)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    
    
    if args.method == "SNN":
        optimizer = optim.SGD(model.parameters(),lr = learning_rate,momentum=0.9,weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= 0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= 1e-7)
            
    
    "training"
    if (args.type == "sub-wise"):
        f= open("sub-wise_resultDB2adv_jen.txt","w+")
        acc_ls = []
        for i in range(40):
            model = ({"EMGhandnet": models.EMGhandnet,"SNN":models.SNN,
                      "CNN":models.NormalCNN, "TF":models.TFModel, "TF2":models.TFModel2}[args.model_name]()).to(device) 
            learning_rate = args.learning_rate
            epochs = args.epochs
            batch_size = args.batch_size
            train = pd.read_pickle(f"./DB2_subject{i+1}_train.pkl")
            eval_data = pd.read_pickle(f"./DB2_subject{i+1}_test.pkl")
            train_dataset = Dataset(train)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
            eval_dataset = Dataset(eval_data)
            eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
            train_loss_list = []
            val_loss_list = []
            train_accuracy_list = []
            val_accuracy_list = []
            best_loss = 5e10
            ba = 0
            best_correct = 0
            
            if args.method == "SNN":
                optimizer = optim.SGD(model.parameters(),lr = learning_rate,momentum=0.9,weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= 0)
            else:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=5e-5)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= 1e-7)
                
            for epoch in range(epochs):   
                gc.collect()
                total_loss, total_val_loss = 0, 0
                total_acc, total_val_acc, total_correct = 0, 0, 0
                
                #for name, param in model.named_parameters():
                    #print(name, param.requires_grad)
                print(f"learning_rate : {scheduler.get_lr()[0]}")
                tqdm_dataset = tqdm(train_dataloader)
                training = True
                for batch,batch_item in enumerate(tqdm_dataset):
        
                    batch_loss, batch_acc, correct= train_step(batch_item, epoch, batch, training, model, optimizer, device)
                    total_loss += batch_loss.item()
                    total_acc += batch_acc.item()
                    
                    tqdm_dataset.set_postfix({
                        'Epoch': epoch + 1,
                        'Loss': '{:06f}'.format(batch_loss.item()),
                        'Train Loss' : '{:06f}'.format(total_loss/(batch+1)),
                        'Train ACC' : '{:03f}'.format(total_acc/(batch+1)),
                    })
                    
                train_loss_list.append(total_loss/(batch+1))
                train_accuracy_list.append(total_acc/(batch+1))
                tqdm_dataset = tqdm(eval_dataloader)
                training = False
                for batch, batch_item in enumerate(tqdm_dataset):
                    batch_loss, batch_acc, correct = train_step(batch_item, epoch, batch, training,model, optimizer, device)
                    total_val_loss += batch_loss.item()
                    total_val_acc += batch_acc
                    total_correct += correct
                    tqdm_dataset.set_postfix({
                        'Epoch': epoch + 1,
                        'Loss': '{:06f}'.format(batch_loss.item()),
                        'Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                        'Val ACC' : '{:03f}'.format(total_val_acc/(batch+1)),
                    })
                scheduler.step()
                val_loss_list.append(total_val_loss/(batch+1))
                val_accuracy_list.append(total_val_acc.cpu()/(batch+1))
                
                if best_loss>(total_val_loss/(batch+1)):
                    best_loss = total_val_loss/(batch+1)
                    best_acc = total_val_acc/(batch+1)
                    best_epoch = epoch+1
                if ba < (total_val_acc/(batch+1)):
                    ba = total_val_acc/(batch+1)
                    ba_loss = total_val_loss/(batch+1)
                    ba_epoch = epoch+1
                if best_correct < total_correct:
                    bc_epoch = epoch+1
                    best_correct = total_correct
            acc_ls.append(ba)
            print(acc_ls)
            f.write(f"subject{i+1}'s best_acc : {ba:.3f} ba_epoch : {ba_epoch} best_loss_ac : {best_acc:.3f} best_epoch : {best_epoch} best_correct : {best_correct} bc_epoch : {bc_epoch}\n")
    elif (args.type == "leave-one-out"):
        for i in range(27):
            f= open(f"leave-one-out_sub{i+1}_result.txt","w+")
            model = ({"EMGhandnet": models.EMGhandnet,"SNN":models.SNN,
                      "CNN":models.NormalCNN, "TF":models.TFModel, "TF2":models.TFModel2}[args.model_name]()).to(device)
            learning_rate = args.learning_rate
            epochs = args.epochs
            batch_size = args.batch_size
            temp_list = []
            for j in range(27):
                if j != i:
                    temp = pd.read_pickle(f"./subject{j+1}_all.pkl")
                    temp_list.append(temp)
            train= pd.concat(temp_list)
            eval_data = pd.read_pickle(f"./subject{i+1}_all.pkl")
            train_dataset = Dataset(train)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
            eval_dataset = Dataset(eval_data)
            eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
            train_loss_list = []
            val_loss_list = []
            train_accuracy_list = []
            val_accuracy_list = []
            best_loss = 5e10
            ba = 0
            best_correct = 0
            
            if args.method == "SNN":
                optimizer = optim.SGD(model.parameters(),lr = learning_rate,momentum=0.9,weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= 0)
            else:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=5e-5)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= 1e-7)
                
            for epoch in range(epochs):   
                gc.collect()
                total_loss, total_val_loss = 0, 0
                total_acc, total_val_acc, total_correct = 0, 0, 0
                
                #for name, param in model.named_parameters():
                    #print(name, param.requires_grad)
                print(f"learning_rate : {scheduler.get_lr()[0]}")
                tqdm_dataset = tqdm(train_dataloader)
                training = True
                for batch,batch_item in enumerate(tqdm_dataset):
        
                    batch_loss, batch_acc, correct= train_step(batch_item, epoch, batch, training, model, optimizer, device)
                    total_loss += batch_loss.item()
                    total_acc += batch_acc.item()
                    
                    tqdm_dataset.set_postfix({
                        'Epoch': epoch + 1,
                        'Loss': '{:06f}'.format(batch_loss.item()),
                        'Train Loss' : '{:06f}'.format(total_loss/(batch+1)),
                        'Train ACC' : '{:03f}'.format(total_acc/(batch+1)),
                    })
                    
                train_loss_list.append(total_loss/(batch+1))
                train_accuracy_list.append(total_acc/(batch+1))
                tqdm_dataset = tqdm(eval_dataloader)
                training = False
                for batch, batch_item in enumerate(tqdm_dataset):
                    batch_loss, batch_acc, correct = train_step(batch_item, epoch, batch, training,model, optimizer, device)
                    total_val_loss += batch_loss.item()
                    total_val_acc += batch_acc
                    total_correct += correct
                    tqdm_dataset.set_postfix({
                        'Epoch': epoch + 1,
                        'Loss': '{:06f}'.format(batch_loss.item()),
                        'Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                        'Val ACC' : '{:03f}'.format(total_val_acc/(batch+1)),
                    })
                scheduler.step()
                val_loss_list.append(total_val_loss/(batch+1))
                val_accuracy_list.append(total_val_acc.cpu()/(batch+1))
                
                if best_loss>(total_val_loss/(batch+1)):
                    best_loss = total_val_loss/(batch+1)
                    best_acc = total_val_acc/(batch+1)
                    best_epoch = epoch+1
                if ba < (total_val_acc/(batch+1)):
                    ba = total_val_acc/(batch+1)
                    ba_loss = total_val_loss/(batch+1)
                    ba_epoch = epoch+1
                if best_correct < total_correct:
                    bc_epoch = epoch+1
                    best_correct = total_correct
            f.write(f"subject{i+1}'s best_acc : {ba:.3f} ba_epoch : {ba_epoch} best_loss_ac : {best_acc:.3f} best_epoch : {best_epoch} best_correct : {best_correct} bc_epoch : {bc_epoch}\n")
        
        print('hello')
    else:
        #model.load_state_dict(torch.load('./pretrain.ckpt'))
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
        for epoch in range(epochs):   
            gc.collect()
            total_loss, total_val_loss = 0, 0
            total_acc, total_val_acc, total_correct = 0, 0, 0
            best_correct = 0
            
            #for name, param in model.named_parameters():
                #print(name, param.requires_grad)
            print(f"learning_rate : {scheduler.get_lr()[0]}")
            tqdm_dataset = tqdm(train_dataloader)
            training = True
            for batch,batch_item in enumerate(tqdm_dataset):
    
                batch_loss, batch_acc, _= train_step(batch_item, epoch, batch, training, model, optimizer, device)
                total_loss += batch_loss.item()
                total_acc += batch_acc.item()
                
                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Loss': '{:06f}'.format(batch_loss.item()),
                    'Train Loss' : '{:06f}'.format(total_loss/(batch+1)),
                    'Train ACC' : '{:06f}'.format(total_acc/(batch+1)),
                })
                
            train_loss_list.append(total_loss/(batch+1))
            train_accuracy_list.append(total_acc/(batch+1))
            tqdm_dataset = tqdm(eval_dataloader)
            training = False
            for batch, batch_item in enumerate(tqdm_dataset):
                batch_loss, batch_acc, correct= train_step(batch_item, epoch, batch, training,model, optimizer, device)
                total_val_loss += batch_loss.item()
                total_val_acc += batch_acc
                total_correct += correct
                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Loss': '{:06f}'.format(batch_loss.item()),
                    'Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                    'Val ACC' : '{:06f}'.format(total_val_acc/(batch+1)),
                })
            scheduler.step()
            val_loss_list.append(total_val_loss/(batch+1))
            val_accuracy_list.append(total_val_acc.cpu()/(batch+1))
            
            if best_loss>(total_val_loss/(batch+1)):
                best_loss = total_val_loss/(batch+1)
                best_acc = total_val_acc/(batch+1)
                best_epoch = epoch+1
            if ba < (total_val_acc/(batch+1)):
                ba = total_val_acc/(batch+1)
                ba_loss = total_val_loss/(batch+1)
                ba_epoch = epoch+1
            if best_correct < total_correct:
                bc_epoch = epoch+1
                best_correct = total_correct
        
        plt.plot(train_loss_list,c='b',label = 'train_loss')
        plt.plot(val_loss_list,c='r', label = 'test_loss')
        plt.legend()
        plt.savefig(f'{args.model_name}_{args.method}_loss', dpi=1200)
        plt.clf()
        plt.plot(train_accuracy_list,c='b',label = 'train_accuracy')
        plt.plot(val_accuracy_list,c='r',label = 'test_accuracy')
        plt.legend()
        
        plt.savefig(f'{args.model_name}_{args.method}_acc', dpi=1200)

    
    print(f"best_loss : {best_loss} best_loss_ac : {best_acc} best_epoch : {best_epoch}")
    print(f"ba_loss : {ba_loss} best_ac : {ba} ba_epoch : {ba_epoch}")
    


if __name__ == "__main__":
    exit(main(parser.parse_args()))
