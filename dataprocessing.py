from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%% multi file
dir_path = r'C:\Users\owner\projects\emg\datas\db1'
file_pathlist = []
dir_list = []
file_type = []
for (root, directories, files) in os.walk(dir_path):
    for d in directories:
        d_path = os.path.join(root,d)
        dir_list.append(d)
        print(d)
        
    for file in files:
        file_path = os.path.join(root,file)
        file_pathlist.append(file_path)
        if ("E3" in file):
            file_type.append("E3")
        elif ("E2" in file):
            file_type.append("E2")
        else:
            file_type.append("E1")
        print(file_path)

#%% stimulus별로 data를 쪼갬.
path= r'C:\Users\owner\Downloads\nina1\s1\S1_A1_E1.mat'
res = loadmat(path)
emg = res['emg']
train = pd.DataFrame(columns=['emg','stimulus','repetition','subject','rerepetition','restimulus'])
test = pd.DataFrame(columns=['emg','stimulus','repetition','subject','rerepetition','restimulus'])



now = 100;
start=0;
end = 0;
idx=-1;
for j,file_path in enumerate(file_pathlist):
    res = loadmat(file_path)
    for i in range(len(res['emg'])-1):
        if(res['stimulus'][i][0] != now):
            now = res['stimulus'][i][0]
            start= i
        if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (2,5,7) )):
            idx+=1
            train.loc[idx,'emg']=res['emg'][start:i]
            if (file_type[j]=="E3"):
                now = now+29
            elif (file_type[j]=="E2"):
                now = now+12
            train.loc[idx,'stimulus']=now-1
            train.loc[idx,'repetition']= f'{res["repetition"][i][0]}'
            train.loc[idx,'subject'] = f'subject{int(j/3)+1}'    
            train.loc[idx,'rerepetition']= f'{res["rerepetition"][i][0]}'
            train.loc[idx,'restimulus']= f'{res["restimulus"][i][0]-1}'    
        
train.to_pickle('ninaprodb1train.pkl')

now = 100;
start=0;
end = 0;
idx=-1;
for j,file_path in enumerate(file_pathlist):
    res = loadmat(file_path)
    for i in range(len(res['emg'])-1):
        if(res['stimulus'][i][0] != now):
            now = res['stimulus'][i][0]
            start= i
        if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (1,3,4,6,8,9,10) )):
            idx+=1
            test.loc[idx,'emg']=res['emg'][start:i]
            if (file_type[j]=="E3"):
                now = now+29
            elif (file_type[j]=="E2"):
                now = now+12
            test.loc[idx,'stimulus']=now-1
            test.loc[idx,'repetition']= f'{res["repetition"][i][0]}'
            test.loc[idx,'subject'] = f'subject{int(j/3)+1}'    
            test.loc[idx,'rerepetition']= f'{res["rerepetition"][i][0]}'
            test.loc[idx,'restimulus']= f'{res["restimulus"][i][0]-1}'    
        

test.to_pickle('ninaprodb1test.pkl')



#%% 
path= r'C:\Users\owner\Downloads\nina1\s1\S1_A1_E1.mat'
res = loadmat(path)
emg = res['emg']
train = pd.DataFrame(columns=['emg','stimulus','repetition','subject','rerepetition','restimulus'])
test = pd.DataFrame(columns=['emg','stimulus','repetition','subject','rerepetition','restimulus'])



now = 100;
start=0;
end = 0;
idx=-1;
for i in range(len(res['emg'])-1):
    if(res['stimulus'][i][0] != now):
        now = res['stimulus'][i][0]
        start= i
    if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (2,5,7) )):
        idx+=1
        train.loc[idx,'emg']=res['emg'][start:i]

        train.loc[idx,'stimulus']=now-1
        train.loc[idx,'repetition']= f'{res["repetition"][i][0]}'
        train.loc[idx,'rerepetition']= f'{res["rerepetition"][i][0]}'
        train.loc[idx,'restimulus']= f'{res["restimulus"][i][0]}'    

train.to_pickle('subject1train.pkl')

for i in range(len(res['emg'])-1):
    if(res['stimulus'][i][0] != now):
        now = res['stimulus'][i][0]
        start= i
    if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (1,3,4,6,8,9,10) )):
        idx+=1
        test.loc[idx,'emg']=res['emg'][start:i]

        test.loc[idx,'stimulus']=now-1
        test.loc[idx,'repetition']= f'{res["repetition"][i][0]}'
        test.loc[idx,'rerepetition']= f'{res["rerepetition"][i][0]}'
        test.loc[idx,'restimulus']= f'{res["restimulus"][i][0]}'    
test.to_pickle('subject1test.pkl')
#%% wavelet
    
#test = emg.transpose()[0]pywt.wavelist('coif')
    
# reshape best (10,25,20) else (10,20,25) (10,500,1) ( 10,250,2)

def WT(df, col, wavelet='db5', thresh=0.63):
    signal = df[col].values
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def WT2(df, col, wavelet='sym8', thresh=0.63):
    signal = df[col]
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def WT3( wavelet='sym8', thresh=0.63):
    signal = test
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal
 
f = WT3()


#%% multi file
dir_path = r'C:\Users\owner\projects\emg\datas\db1'
file_pathlist = []
dir_list = []
file_type = []
for (root, directories, files) in os.walk(dir_path):
    for d in directories:
        d_path = os.path.join(root,d)
        dir_list.append(d)
        print(d)
        
    for file in files:
        file_path = os.path.join(root,file)
        file_pathlist.append(file_path)
        if ("E3" in file):
            file_type.append("E3")
        elif ("E2" in file):
            file_type.append("E2")
        else:
            file_type.append("E1")
        print(file_path)

#%% stimulus별로 data를 쪼갬.
path= r'C:\Users\owner\projects\emg\datas\db4\s1\S1_E2_A1.mat'
res = loadmat(path)
emg = res['emg']
train = pd.DataFrame(columns=['emg','stimulus','repetition','subject','rerepetition','restimulus'])



now = 100;
start=0;
end = 0;
idx=-1;
for j,file_path in enumerate(file_pathlist):
    res = loadmat(file_path)
    for i in range(len(res['emg'])-1):
        if(res['stimulus'][i][0] != now):
            now = res['stimulus'][i][0]
            start= i
        if(res['stimulus'][i+1][0] != now and now !=0):
            idx+=1
            train.loc[idx,'emg']=res['emg'][start:i]
            if (file_type[j]=="E3"):
                now = now+29
            elif (file_type[j]=="E2"):
                now = now+12
            train.loc[idx,'stimulus']=now-1
            train.loc[idx,'repetition']= f'{res["repetition"][i][0]}'
            train.loc[idx,'subject'] = f'subject{int(j/3)+1}'    
            train.loc[idx,'rerepetition']= f'{res["rerepetition"][i][0]}'
            train.loc[idx,'restimulus']= f'{res["restimulus"][i][0]}'    
