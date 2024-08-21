from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.getcwd())
from utils.tools import StandardScaler

dataset = pd.read_csv('/home/ws5/Desktop/Code_Base_Genesys/Implemented_ML_Papers/Informer/Informer2020/data/OBD_ADMA/Informer_dataset_file_fivehourdataset.csv')
data_path ="/home/ws5/Desktop/Code_Base_Genesys/Implemented_ML_Papers/Informer/Informer2020/data/OBD_ADMA/Informer_dataset_file_fivehourdataset.csv"

#Writing the dataset class for the regression model to extract the data.
class adma_obd_dataset(Dataset):
    def __init__(self,data_path, scale=True, flag='train'):
        self.data_path = data_path
        self.scale = scale
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.read_data()

    def read_data(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(data_path)

        #adding noise in test dataset
        noise_level = 0
        start_idx = int(0.825 * len(df_raw))
        for column in df_raw.columns:
            if column not in ['Correvit_slip_angle_COG_corrvittiltcorrected', 'INSTimestamp_ADMA']:
                df_raw.loc[start_idx:, column] = df_raw.loc[start_idx:, column] + np.random.normal(0, noise_level,
                                                                                                     len(df_raw) - start_idx)

        cols = list(df_raw.columns)
        cols.remove('INSTimestamp_ADMA')
        cols.remove('INS_time_sec')
        cols.remove('side_slip_angle_COG')

        df_raw = df_raw[cols]


        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2) #for domain adaption dataset.
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[cols]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

        else:
            data = df_data.values
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        seq_x = self.data_x[s_begin]
        seq_y = self.data_x[s_begin:s_begin+5]
        return seq_x,seq_y

    def __len__(self):
        return len(self.data_x)-5

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

model = nn.Sequential(
nn.Linear(9,45,bias=True),
    nn.ReLU(),
    nn.Linear(45,15,bias=True),
    nn.ReLU(),
    nn.Linear(15,5,bias=True),
    nn.ReLU(),
    nn.Linear(5,5,bias=True)
 )

#model = nn.Sequential(
    #nn.Linear(9,20,bias=True),
    #nn.ReLU(),
    #nn.Linear(20,5,bias=True)
#)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.09, weight_decay=0.01)
#optimizer = optim.Adam(model.parameters(),lr=0.0001)

n_epochs = 10
batch_size = 500


#Dataloaders
train_loader = DataLoader(adma_obd_dataset(data_path, flag='train'), batch_size=batch_size,shuffle=True)
vali_loader = DataLoader(adma_obd_dataset(data_path, flag='val'), batch_size=batch_size,shuffle=True)
test_loader = DataLoader(adma_obd_dataset(data_path, flag='test'), batch_size=batch_size,shuffle=False)

history = []

for epoch in range(n_epochs):
    print(epoch)
    iter_count = 0
    train_loss = []
    model.train()
    for i, (batch_x, batch_y) in enumerate(train_loader):
        iter_count += 1
        pred = model(batch_x[:,:-1].float())
        true = batch_y[:, :, -1].float()
        loss = criterion(pred, true)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = []
    for i , (batch_x, batch_y) in enumerate(vali_loader):
        pred = model(batch_x[:, :-1].float())
        true = batch_y[:,:,-1].float()
        loss = criterion(pred, true)
        val_loss.append(loss.item())
    val_loss = np.average(val_loss)
    history.append(val_loss)
    print(val_loss)
    model.train()

plt.plot(history)
plt.show()
plt.close()

preds = []
trues = []
model.eval()
test_dataset = adma_obd_dataset(data_path, flag='test')# To speed up the process
for i, (batch_x,batch_y) in enumerate(test_loader):
    pred = model(batch_x[:, :-1].float())
    true = batch_y[:, :,-1].float()

    # This part of the code was modified to reproject the data back to slip angles (inversion of the conversion process)
    true_rescaled = test_dataset.inverse_transform((true[:, :])).unsqueeze(-1)
    pred_rescaled = test_dataset.inverse_transform((pred[:, :])).unsqueeze(-1)
    preds.append(pred_rescaled.detach().cpu().numpy())
    trues.append(true_rescaled.detach().cpu().numpy())

preds_i = np.array(preds[:-1]) #Leaving out the last batch due to dimensiong issues
trues_i = np.array(trues[:-1])
preds_f = np.array(preds[-1])
trues_f = np.array(trues[-1])
print('test shape:', preds_i.shape, trues_i.shape)
preds_i = preds_i.reshape(-1, preds_i.shape[-2], preds_i.shape[-1])
trues_i = trues_i.reshape(-1, trues_i.shape[-2], trues_i.shape[-1])
preds_f = preds_f.reshape(-1, preds_f.shape[-2], preds_f.shape[-1])
trues_f = trues_f.reshape(-1, trues_f.shape[-2], trues_f.shape[-1])

preds_combined = np.concatenate((preds_i, preds_f), axis=0)
trues_combined = np.concatenate((trues_i, trues_f), axis=0)

np.save('results/Regression/pred_regression_fivehours.npy', preds_combined)
np.save('results/Regression/true_regression_fivehours.npy', trues_combined)
mean_absolute_error = np.mean(np.abs(trues_combined - preds_combined))
print('test shape:', preds_combined.shape, trues_combined.shape)
print('mean-absolute-error', mean_absolute_error)
print('max-absolute-error', np.max(np.abs(trues_combined - preds_combined)))

# Visualization in tensorboard
pred_first_elements = preds_combined [:, 0,:]
true_first_elements = trues_combined[:, 0,:]
plt.figure(figsize=(10, 6))
plt.plot(pred_first_elements, label='Regression/MLP prediction', color="b")
plt.plot(true_first_elements, label='Ground truth', color="g")
plt.xlabel('Time stamps')
plt.ylabel('Slip angle in degrees')
plt.title('True vs regression models')
plt.legend()
plt.show()
plt.close()
