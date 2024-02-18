import torch
import torch.nn.functional as F
import torch.nn as nn
from spikingjelly.activation_based.functional import reset_net
from torchsummaryX import summary
#%% functions
criterion = nn.CrossEntropyLoss()

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def correct_function(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    return correct.float()

def calculate_accuracy2(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
#%%
def inference(batch_item,batch,model,device):
    inputs = batch_item[0].to(device)
    labels = batch_item[1].to(device)
    
    model.eval()
    with torch.no_grad(): output = model(x = inputs.float())
    loss = criterion(output, labels)
    accuracy = calculate_accuracy(output, labels)
        
    return loss, accuracy

#%% default
def train_step(batch_item, epoch, batch, training,model, optimizer, device):
    inputs = batch_item['input_data'].to(device)
    labels = batch_item['label'].to(device)
    criterion = nn.CrossEntropyLoss()
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(): output, _= model(inputs)
        loss = criterion(output, labels)
        #print(labels)
        #print(torch.argmax(output,dim=1))
        accuracy = calculate_accuracy(output, labels)
        correct = correct_function(output, labels)
        #print(output.argmax(1),labels)
        #print(output)
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad(): output, _= model(inputs)
        #print(labels)
        #print(torch.argmax(output,dim=1))
        correct = correct_function(output, labels)
        loss = criterion(output, labels)
        accuracy = calculate_accuracy(output, labels)
        
    return loss, accuracy, correct


#%% adversarial
def _get_symm_kl(noised_output, output):
    return (
        F.kl_div(
            F.log_softmax(noised_output,dim=1),
            0.5*(F.softmax(output,dim=1)+F.softmax(noised_output,dim=1)),
            None,
            None,
            "sum",
        )
        + F.kl_div(
            F.log_softmax(output,dim=1),
            0.5*(F.softmax(noised_output,dim=1)+F.softmax(output,dim=1)),
            None,
            None,
            "sum",
        )
    ) /(2*noised_output.size(0))

"""
def _get_symm_kl(noised_output, output):
    return (
        F.kl_div(
            F.log_softmax(noised_output,dim=1),
            F.softmax(output,dim=1),
            None,
            None,
            "sum",
        )
        + F.kl_div(
            F.log_softmax(output,dim=1),
            F.softmax(noised_output,dim=1),
            None,
            None,
            "sum",
        )
    ) / noised_output.size(0)
"""
def adversarial_train_step(batch_item, epoch, batch, training,model, optimizer, device):
    inputs = batch_item['input_data'].to(device)
    labels = batch_item['label'].to(device)
    noised_inputs = batch_item['noised'].to(device)
    #noise_sampler = torch.distributions.uniform.Uniform(-0.06*(1e-1), 0.06*1e-1)
    
    #noise_sampler = torch.distributions.normal.Normal(0.06*(1e-1), 0.11*1e-1)
    
        
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(): output,_= model(inputs)
        #noise = noise_sampler.sample(sample_shape=inputs.shape).to(inputs)
        #noised_inputs = inputs.detach().clone() + noise
        with torch.cuda.amp.autocast(): noised_output, _= model(noised_inputs)
        symm_kl = _get_symm_kl(noised_output, output)
        #print(output.shape)
        loss = criterion(output, labels)
        loss = loss + 10*symm_kl
        correct = correct_function(output, labels)
        accuracy = calculate_accuracy(output, labels)
        loss.backward()
        
        optimizer.step()
    else:
        model.eval()
        with torch.cuda.amp.autocast(): output,_ = model(inputs)
        #noise = noise_sampler.sample(sample_shape=inputs.shape).to(inputs)
        #noised_inputs = inputs.detach().clone() + noise
        with torch.cuda.amp.autocast(): noised_output,_ =  model(noised_inputs)
        symm_kl = _get_symm_kl(noised_output, output)
        loss = criterion(output, labels)
        loss = loss + 10*symm_kl
        accuracy = calculate_accuracy(output, labels)
        correct = correct_function(output, labels)
        
    return loss, accuracy, correct

#%%
def SNN_train_step(batch_item, epoch, batch, training,model, optimizer, device):
    inputs = batch_item['input_data'].to(device)
    labels = batch_item['label'].to(device)
    criterion = nn.CrossEntropyLoss()
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(): output_list = model(x=inputs)
        loss = criterion(sum(output_list)/100, labels)
        accuracy = calculate_accuracy(sum(output_list)/100, labels)
        #print(output.argmax(1),labels)
        loss.backward()
        reset_net(model)
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad(): output_list = model(x = inputs)
        loss = criterion(sum(output_list)/100, labels)
        accuracy = calculate_accuracy(sum(output_list)/100, labels)
        reset_net(model)
        
    return loss, accuracy