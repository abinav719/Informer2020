"""In the results analysis file it is not obvious.
The entire fusion architecture is as follows:
The main informer is run in prediction mode to extract the values for train and val (Informer values and variance)
Then in this script the corresponding KMB1,KMB2 and its variance is computed
This computed dataset acts as the training dataset for the neural network model for learning fusion network
The model trained on this makes predictions on the test dataset.
Now the testing is decoupled so the fusion_test_dataset is used to generate values.
"""
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
sys.path.append(os.path.abspath("/home/ws5/Desktop/Code_Base_Genesys/Implemented_ML_Papers/Informer/Informer2020/Abinav_exp"))
from Kalman_filter import *
from scipy.signal import butter, filtfilt


file = "results/informer_OBD_ADMA_ftMS_sl100_ll25_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0_Uncertainty=Student-t_fivehours_new"
dataset = pd.read_csv("data/OBD_ADMA/Informer_dataset_file_fivehourdataset_new.csv")

data_pred = np.load(f'./{file}/fusion_dataset_pred.npy')
data_true = np.load(f'./{file}/fusion_dataset_true.npy')
data_variance = np.load(f'./{file}/fusion_dataset_variance_pred.npy')
pred_first_elements = data_pred[:, 0, :]
true_first_elements = data_true[:, 0, :]
variance_first_elements = data_variance[:, 0, :]
"""Check this code block whether things are matching"""
#This is for matching and is crucial check that the true_first_elements and acutal correvit slip angle in dataset
#matching with each other.
dataset = dataset[100:len(pred_first_elements)+100]
dataset = dataset.reset_index(drop=True)
#Still there might be some mismatch between the two due to some padding in the time series (check it here)
#index = np.where(true_first_elements[:] == dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].iloc[0])[0].item()

index = 0
dataset["true_informer"] = np.nan
true_first_elements_1d = true_first_elements[:, 0].flatten()
dataset["true_informer"]=true_first_elements_1d
#Check whether this is correct and true_informer matches with correvit slip angle in dataset
dataset["pred_informer"] = np.nan
pred_first_elements_1d = pred_first_elements[:, 0].flatten()
dataset["pred_informer"] = pred_first_elements_1d

# #variance block
dataset["variance_informer"] = np.nan
variance_first_elements_1d = variance_first_elements[:, 0].flatten()
dataset["variance_informer"]=variance_first_elements_1d

#Entire code block commented out to read
#Now need to create variance and values for the KMB model
# print('KMB1')
# dataset["SW_pos_obd"] = dataset["SW_pos_obd"] - 10#6#10 #Correvit -10,5,10(final)
# negative_play = 5
# positive_play = 15  #For new datasets check whether 5 or 10 holds good for positive play.
# mask_steering = ((dataset["SW_pos_obd"] <= 0) & (dataset["SW_pos_obd"] >= -negative_play))
# dataset.loc[mask_steering,'SW_pos_obd'] = 0
# mask_steering = ((dataset["SW_pos_obd"] <= positive_play) & (dataset["SW_pos_obd"] >= 0))
# dataset.loc[mask_steering,'SW_pos_obd'] = 0
# dataset.loc[dataset["SW_pos_obd"] > 0, "SW_pos_obd"] -= positive_play
# dataset.loc[dataset["SW_pos_obd"] < 0, "SW_pos_obd"] += negative_play
# mask_speedyaw = dataset['speedo_obd']>5
# dataset.loc[mask_speedyaw,'Vehicle_slip_obd'] = (np.arctan(np.tan((dataset.loc[mask_speedyaw,"SW_pos_obd"]*np.pi)/(22*180))*0.427))*(180/np.pi) #lh=799,8,l=1873 for IssaK
# # #dataset.loc[mask_speedyaw,'Vehicle_slip_obd'] = 0.427*((dataset.loc[mask_speedyaw,"SW_pos_obd"]*2*np.pi)/(22*360))
# dataset['Yaw_rate_kmb1'] = (dataset['speedo_obd']*(5/18)*np.tan(dataset["SW_pos_obd"]*2*np.pi/(22*360))/1.873)*(180/np.pi)
# dataset.loc[~mask_speedyaw,'Vehicle_slip_obd'] = 0
#
# plt.figure(figsize=(10, 6))
# plt.plot((dataset['Yaw_rate_kmb1']-dataset['yaw_rate']).abs(),label='KMB 1 vs yaw rate')
# plt.plot((dataset['Correvit_slip_angle_COG_corrvittiltcorrected']-dataset['Vehicle_slip_obd']).abs(),label='Correvit vs KMB1')
# plt.legend()
# plt.show()
# plt.close()
#
# #These plots are made to analyse the error between predicted yaw rate vs actual yaw rate to generate confidence for KMB-1
# condition =  ((dataset['speedo_obd'] <= 200) & ( dataset["LatAcc_obd"].abs() <= 30))
# abs_diff_kmb_yaw = (dataset.loc[condition,'Yaw_rate_kmb1'] - dataset.loc[condition,'yaw_rate']).abs()
# bin_edges = np.arange(0, 15+ 0.25, 0.25)
# bin_indices = np.digitize(np.abs(dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]-dataset.loc[condition,'Vehicle_slip_obd']), bins=bin_edges)
#
# #abs_diff_regression_df = (dataset.loc[condition,"regression"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
#
# # Calculate mean absolute error for each bin
# mean_abs_yaw_error_per_bin_KMB_1 = []
# bin_centers = []
# bin_counts = []
# for i in range(1, len(bin_edges)):
#     bin_mask = (bin_indices == i)
#     if np.any(bin_mask):
#         mean_abs_yaw_error_kmb1 = np.mean(abs_diff_kmb_yaw[bin_mask])
#
#         mean_abs_yaw_error_per_bin_KMB_1.append(mean_abs_yaw_error_kmb1)
#
#         bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)
#         bin_counts.append(np.sum(bin_mask))
# # Plot the mean absolute error for each bin
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.plot(bin_centers,mean_abs_yaw_error_per_bin_KMB_1,marker='o', linestyle='-', label='Yaw error KMB1')
# ax1.set_xlabel('Error between Correvit vehicle slip angle and KMB-1 (degrees)')
# ax1.set_ylabel('Yaw rate error from KMB-1')
# ax1.set_title('Yaw rate error between state equations and measurement KMB-1')
# ax1.grid(True)
# ax1.legend(loc='upper left')
#
# ax2 = ax1.twinx()
#
# # Plot the histogram data as a bar plot on the twin axis
# ax2.bar(bin_centers, (bin_counts/sum(bin_counts))*100, width=0.25, color='gray', alpha=0.5, label='Data count per bin')
# ax2.set_ylabel('Data Percentage in bin')
# ax2.set_ylim(0, max((bin_counts/sum(bin_counts))*100)+5)  # Adjust the ylim for better visualization
#
# # Add legend for the histogram
# ax2.legend(loc='upper right')
# plt.show()
# plt.close()
#
# dataset['kmb_1_variance']= (dataset['Yaw_rate_kmb1']-dataset['yaw_rate']).abs()
#
#
# #Tire velocities ( This model is working well :))
# print('kmb-2')
# mask_speedyaw = dataset['speedo_obd']>5
# dataset.loc[mask_speedyaw,'speed_y'] =  ((dataset.loc[mask_speedyaw,"VelFL_obd"]+dataset.loc[mask_speedyaw,"VelRL_obd"]) - (dataset.loc[mask_speedyaw,"VelFR_obd"]+dataset.loc[mask_speedyaw,"VelRR_obd"]))/4
# dataset.loc[mask_speedyaw,'speed_x'] =  (dataset.loc[mask_speedyaw,"VelFR_obd"]+dataset.loc[mask_speedyaw,"VelFL_obd"]+dataset.loc[mask_speedyaw,"VelRR_obd"]+dataset.loc[mask_speedyaw,"VelRL_obd"])/4
# dataset.loc[mask_speedyaw,'KMB_slip2'] = -1*((np.arctan(dataset.loc[mask_speedyaw,'speed_y']/dataset.loc[mask_speedyaw,'speed_x'])*(180/np.pi)))
# dataset.loc[~mask_speedyaw,'KMB_slip2'] = 0
# dataset.loc[~mask_speedyaw,'Vehicle_slip_obd'] = 0
#
# def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
#     nyquist = 0.5 * sample_rate  # Nyquist frequency
#     normal_cutoff = cutoff_freq / nyquist
#     # Get the filter coefficients
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     # Apply the filter to the data using filtfilt
#     y = filtfilt(b, a, data)
#     return y
#
# cutoff_frequency = 2.0  # Desired cutoff frequency (in Hz)
# sampling_rate = 50.0   # Sampling rate of the data (in Hz)
# filter_order = 3
# dataset['KMB_slip2_fitler'] = butter_lowpass_filter(dataset['KMB_slip2'], cutoff_frequency, sampling_rate, filter_order)
# condition =  ((dataset['speedo_obd'] <= 200) & ( dataset["LatAcc_obd"].abs() <= 30))
#
# print("kmb2",((dataset.loc[condition, 'KMB_slip2'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean())
# print("kmb2 butter worth",((dataset.loc[condition, 'KMB_slip2_fitler'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean())
# #print("kmb3",((dataset.loc[condition, 'KMB_3'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean())
# plt.figure(figsize=(10, 6))
# plt.plot(dataset['KMB_slip2'],label="kmb slip 2")
# plt.plot(dataset['Correvit_slip_angle_COG_corrvittiltcorrected'],label='GT - Correvit')
# plt.plot(dataset['KMB_slip2_fitler'],label='kmb slip 2 butter worth')
# plt.legend()
# plt.show()
# plt.close()
#
#
# #Kalman filter for tire velocities - KMB-2:
# x_n_1 = np.array([[np.float(dataset['Correvit_slip_angle_COG_corrvittiltcorrected'].iloc[0])* np.pi / 180],
#                   [np.float(dataset['yaw_rate'].iloc[0])* np.pi / 180]])
# p_n_1 = np.array([[20,0],[0,20]]) #Need to initialize
# kalman_slip = [np.float(dataset['Correvit_slip_angle_COG_corrvittiltcorrected'].iloc[0])]
# kalman_variance = [0]
# for i in range(len(dataset)-1):
#     i=i+1
#     #y_n = np.array([[np.float(dataset["KMB_slip2"].iloc[i])*np.pi/180],[np.float(dataset['yaw_rate'].iloc[i]*np.pi/180)]])
#     y_n = np.array([[np.float(dataset["KMB_slip2"].iloc[i]) * np.pi / 180], [np.float(dataset['yaw_rate'].iloc[i] * np.pi / 180)]])
#     u_n_1 = np.array([[(np.float(dataset["SW_pos_obd"].iloc[i-1])*np.pi)/(i_s*180)],[0]])
#     if dataset['speedo_obd'].iloc[i-1]<1:
#         matrix_a, matrix_b = state_matrix(1)
#     else:
#         matrix_a, matrix_b = state_matrix(np.float(dataset['speedo_obd'].iloc[i-1])/3.6)
#     x_a_prior, p_a_prior = pred_a_prior(matrix_a,matrix_b,matrix_g,x_n_1,u_n_1,p_n_1)
#     innovation, inno_residual, gain = inno_gain_covariance(y_n,matrix_c,x_a_prior,p_a_prior,Cnm)
#     x_post, p_post = update_posterior (x_a_prior,gain,innovation,matrix_c,p_a_prior)
#     kalman_slip.append(float(x_post[0])*180/np.pi)
#     kalman_variance.append(float(p_post[0,0])*180/np.pi)
#     x_n_1 = x_post
#     p_n_1 = p_post
#
# dataset['kalman_slip_kmb2']=kalman_slip
# dataset["kalman_slip_kmb2_variance"] = kalman_variance
# dataset["fused_slip"] = pred_first_elements_1d
# dataset["fused_slip_2"] = pred_first_elements_1d
# dataset.loc[~mask_speedyaw,'kalman_slip_kmb2'] = 0
# plt.figure(figsize=(10, 6))
#
# plt.plot(dataset['Correvit_slip_angle_COG_corrvittiltcorrected'],label='GT - Correvit')
# plt.plot(dataset['kalman_slip_kmb2'],label='kalman slip')
# plt.plot(dataset['KMB_slip2'],label='KMB-slip2')
# plt.legend()
# plt.show()
# plt.close()
#
#
# # Analyzing kalman filter slip and variance
# """1) At very low velocities and standstill the variance shoots up
# Reason - the state equations have divided by v and v2 in it making it shoot up
# The measurement of slip angle from virtual sensor in this area is made to be 0 from the hard condition of velocity
#
# 2) why variance is high
#
# 3) Model is following the state equation more"""
#
# plt.figure(figsize=(10, 6))
# #plt.plot(np.abs(dataset['Correvit_slip_angle_COG_corrvittiltcorrected']-dataset['kalman_slip_kmb2']),label='GT - Correvit')
# plt.plot(dataset["kalman_slip_kmb2_variance"],label="kalman a prior")
# plt.plot(dataset['Correvit_slip_angle_COG_corrvittiltcorrected'],label='GT - Correvit')
# plt.plot(dataset['kalman_slip_kmb2'],label='kalman slip')
# plt.plot(dataset['KMB_slip2'],label='KMB-slip2')
# plt.legend()
# plt.show()
# plt.close()
# print("kmb2_kalman",((dataset.loc[condition, 'kalman_slip_kmb2'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean())
# condition =  ((dataset['speedo_obd'] <= 200) & ( dataset["LatAcc_obd"].abs() <= 30))
# abs_variance_kmb2_kalman = dataset.loc[condition,"kalman_slip_kmb2_variance"]
# bin_edges = np.arange(0, 15+ 0.25, 0.25)
# bin_indices = np.digitize(np.abs(dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]-dataset.loc[condition,'kalman_slip_kmb2']), bins=bin_edges)
#
# #abs_diff_regression_df = (dataset.loc[condition,"regression"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
#
# # Calculate mean absolute error for each bin
# dataset["kalman_slip_kmb2_variance"]  = dataset["kalman_slip_kmb2_variance"]/1e6 #need to remove this line
# mean_abs_variance_per_bin_KMB_2_kalman = []
# bin_centers = []
# bin_counts = []
# for i in range(1, len(bin_edges)):
#     bin_mask = (bin_indices == i)
#     if np.any(bin_mask):
#         mean_abs_variance_kmb2_kalman = np.mean(abs_variance_kmb2_kalman[bin_mask])
#         mean_abs_variance_per_bin_KMB_2_kalman.append(mean_abs_variance_kmb2_kalman)
#         bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)
#         bin_counts.append(np.sum(bin_mask))
# # Plot the mean absolute error for each bin
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.plot(bin_centers,mean_abs_variance_per_bin_KMB_2_kalman,marker='o', linestyle='-', label='KMB2 kalman variance')
# ax1.set_xlabel('Error between Correvit vehicle slip angle and KMB-2 (degrees)')
# ax1.set_ylabel('Variance from KMB-2')
# ax1.set_title('Varaince significance analysis KMB-2')
# ax1.grid(True)
# ax1.legend(loc='upper left')
# ax2 = ax1.twinx()
#
# # Plot the histogram data as a bar plot on the twin axis
# ax2.bar(bin_centers, (bin_counts/sum(bin_counts))*100, width=0.25, color='gray', alpha=0.5, label='Data count per bin')
# ax2.set_ylabel('Data Percentage in bin')
# ax2.set_ylim(0, max((bin_counts/sum(bin_counts))*100)+5)  # Adjust the ylim for better visualization
#
# # Add legend for the histogram
# ax2.legend(loc='upper right')
# plt.show()
# plt.close()
#
# #Reading the dataset (kalman_slip_kmb3_variance is divided by 1e6)
# mask_speedyaw = dataset['speedo_obd']>2
# dataset = dataset[mask_speedyaw].reset_index(drop=True)
# dataset["variance_informer"] = np.sqrt(dataset["variance_informer"])
# columns_fusion = ['kmb_1_variance',"kalman_slip_kmb2_variance","variance_informer",'Vehicle_slip_obd','kalman_slip_kmb2',"pred_informer","Correvit_slip_angle_COG_corrvittiltcorrected"]
# fusion_dataset = dataset[columns_fusion]
# fusion_dataset.to_csv('fusion_network_training_dataset.csv', index=False)
from gmr import GMM
fusion_dataset = pd.read_csv('fusion_network_training_dataset.csv')
fusion_dataset = fusion_dataset[fusion_dataset['variance_informer']<1].reset_index(drop=True) #removing outliers from train
X = torch.tensor(fusion_dataset.drop(columns=["Correvit_slip_angle_COG_corrvittiltcorrected"]).values, dtype=torch.float32)
y = torch.tensor(fusion_dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].values, dtype=torch.float32).view(-1, 1)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
#Loading test dataset
test_dataset = pd.read_csv('data_fusion_2_test.csv')
mae_informer = (test_dataset['informer_val'] - test_dataset['true_val']).abs().mean()
print('mae_informer',mae_informer)
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_test = torch.tensor(test_dataset.drop(columns=['true_val']).values, dtype=torch.float32)
y_test = torch.tensor(test_dataset['true_val'].values, dtype=torch.float32).view(-1, 1)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

#GMM block for regression
# gmm_input = np.concatenate((X_scaled, y_scaled), axis=1)
# gmm = GMM(n_components=10,random_state=0)
# gmm.from_samples(gmm_input)
# gmm_predict = gmm.predict(np.array([6]), np.concatenate((X_test_scaled, y_test_scaled), axis=1))
# y_pred_original_gmm = scaler_y.inverse_transform(gmm_predict)
# y_original = scaler_y.inverse_transform(y_test_tensor.numpy())
# mae_gmm = np.mean(np.abs((y_pred_original_gmm - y_original)))
# gmr = GMMRegressor(n_compoenents=10)
# gmr.fit(X_scaled, y_scaled)
# gmm_predict = gmr.predict(X_test_scaled)
# y_pred_original_gmm = scaler_y.inverse_transform(gmm_predict)
# y_original = scaler_y.inverse_transform(y_test_tensor.numpy())
# mae_gmm = np.mean(np.abs((y_pred_original_gmm - y_original)))
# print(f'Final Mean Absolute Error (GMM-MAE): {mae_gmm:.4f}')

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset_train, batch_size=1000, shuffle=True)

dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(dataset_val, batch_size=1000, shuffle=False)
class SimpleRegressor(nn.Module):
    def __init__(self, input_size, hidden_size_1,hidden_size_2, output_size,dropout_rate=0.3):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
    def forward(self, x):
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc3(x)
        return x

# Model initialization
#Best configuration so far size 10,3 and dropout = 0
input_size = X.shape[1]
hidden_size_1 = 10 #10
hidden_size_2 = 3  #3
output_size = 1
model = SimpleRegressor(input_size, hidden_size_1,hidden_size_2, output_size,dropout_rate=0.1)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
num_epochs = 100

# Early stopping variables
best_val_loss = float('inf')
patience = 10
trigger_times = 0

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_losses = []
        for val_X, val_y in val_loader:
            val_pred = model(val_X)
            val_loss = criterion(val_pred, val_y)
            val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)

    # Print training and validation losses every 10 epochs
    if (epoch + 1) % 5 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.10f}, Validation Loss: {avg_val_loss:.10f}, Patience:{trigger_times}')

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0  # Reset early stopping counter
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered")
            break

    # Step the scheduler
    scheduler.step()

# Put model in evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions_numpy = predictions.numpy()
    y_pred_original = scaler_y.inverse_transform(predictions_numpy)
    y_original = scaler_y.inverse_transform(y_test_tensor.numpy())
    mae = np.mean(np.abs((y_pred_original  - y_original )))
    print(f'Final Mean Absolute Error (MAE): {mae:.4f}')
    y_pred_df = pd.DataFrame(y_pred_original, columns=['best_fusion_val'])

    # Save the DataFrame to CSV
    #y_pred_df.to_csv('y_pred_original_trained_on_train.csv', index=False)



