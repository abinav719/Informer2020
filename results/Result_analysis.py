#result _ file: informer_OBD_ADMA_ftMS_sl100_ll25_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0
#Dataset _ path:
""""The goal of this code is to visualize the slip angle predicted vs ground truth slip angle (Correvit vs ADMA)
Then we compare this OBD based slip angles"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath("/home/ws5/Desktop/Code_Base_Genesys/Implemented_ML_Papers/Informer/Informer2020/Abinav_exp"))
from Kalman_filter import *
from scipy.signal import butter, filtfilt

#file ="results/informer_OBD_ADMA_ftMS_sl100_ll25_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0_None"
#file = "results/informer_OBD_ADMA_ftMS_sl100_ll25_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0_Uncertainty=None_domainadapt"
#file = "results/informer_OBD_ADMA_ftMS_sl100_ll25_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0_Uncertainty=None_fivehours"
#Student t distribution
#file = "results/informer_OBD_ADMA_ftMS_sl100_ll25_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0_Student-t"
file = "results/informer_OBD_ADMA_ftMS_sl100_ll25_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0_Uncertainty=Student-t_fivehours"

#dataset - domainadapt or fivehourdataset
dataset = pd.read_csv("data/OBD_ADMA/Informer_dataset_file_fivehourdataset.csv")

data_pred = np.load(f'./{file}/pred.npy')
data_true = np.load(f'./{file}/true.npy')
data_variance = np.load(f'./{file}/variance_pred.npy')

#data_pred_regression = np.load(f'results/Regression/pred_regression_fivehours.npy')
#data_true_regression = np.load(f'results/Regression/true_regression_fivehours.npy')
#pred_regression_first_elements = data_pred_regression[:-25, 0, :] #-25 here seems right and matches with dataloaders
#true_regression_first_elements = data_true_regression[:-25, 0, :]

pred_first_elements = data_pred[:, 0, :]
true_first_elements = data_true[:, 0, :]
variance_first_elements = data_variance[:, 0, :]
"""Check this code block whether things are matching"""
#Now we extract the test dataset from the dataset to compare things like obd based slip angle
dataset = dataset[-len(pred_first_elements)-30:-30] #-30 here seems right and matches with dataloaders

#Still there might be some mismatch between the two due to some padding in the time series (check it here)
index = np.where(true_first_elements[:] == dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].iloc[0])[0].item()

index = 0
dataset["true_informer"] = np.nan
true_first_elements_1d = true_first_elements[:, 0].flatten()
dataset["true_informer"]=true_first_elements_1d
#Check whether this is correct and true_informer matches with correvit slip angle in dataset
dataset["pred_informer"] = np.nan
pred_first_elements_1d = pred_first_elements[:, 0].flatten()
dataset["pred_informer"] = pred_first_elements_1d

#regression block
#dataset["regression"] = np.nan
#regression_first_elements_1d = pred_regression_first_elements[:, 0].flatten()
#dataset["regression"]= regression_first_elements_1d

# #variance block
dataset["variance_informer"] = np.nan
variance_first_elements_1d = variance_first_elements[:, 0].flatten()
dataset["variance_informer"]=variance_first_elements_1d


#Calculation of OBD based slip value for comparision (KMB-1) based on botsch book.
# counts, bin_edges = np.histogram(dataset["SW_pos_obd"], bins=range(int(dataset["SW_pos_obd"].min()), int(dataset["SW_pos_obd"].max()) + 1))
# bin_info = []
# # Populate the 2D array with bin edges and counts
# for i in range(len(counts)):
#     bin_info.append([bin_edges[i], bin_edges[i+1], counts[i]])
# # Convert to a numpy array for easier manipulation (optional)
# bin_info = np.array(bin_info)
# plt.figure(figsize=(10, 6))
# plt.hist(dataset["SW_pos_obd"], bins=range(int(dataset["SW_pos_obd"].min()), int(dataset["SW_pos_obd"].max()) + 1), edgecolor='black')
# plt.xlabel('Steering Angle (degrees)')
# plt.ylabel('Frequency')
# plt.title('Histogram of Steering Angles')
# plt.grid(True)
# plt.show()

#These values were obtained after careful calibration and checking in the entire dataset.
#Now the values are more mirrored type. -10 for steering obtained from overall steering plots
#Positive and negative bias chosen after various plots.
dataset["SW_pos_obd"] = dataset["SW_pos_obd"] - 10#6#10 #Correvit -10,5,10(final)
negative_play = 5
positive_play = 15  #For new datasets check whether 5 or 10 holds good for positive play.
mask_steering = ((dataset["SW_pos_obd"] <= 0) & (dataset["SW_pos_obd"] >= -negative_play))
dataset.loc[mask_steering,'SW_pos_obd'] = 0
mask_steering = ((dataset["SW_pos_obd"] <= positive_play) & (dataset["SW_pos_obd"] >= 0))
dataset.loc[mask_steering,'SW_pos_obd'] = 0
dataset.loc[dataset["SW_pos_obd"] > 0, "SW_pos_obd"] -= positive_play
dataset.loc[dataset["SW_pos_obd"] < 0, "SW_pos_obd"] += negative_play
mask_speedyaw = dataset['speedo_obd']>5
dataset.loc[mask_speedyaw,'Vehicle_slip_obd'] = (np.arctan(np.tan((dataset.loc[mask_speedyaw,"SW_pos_obd"]*np.pi)/(22*180))*0.427))*(180/np.pi) #lh=799,8,l=1873 for IssaK
# #dataset.loc[mask_speedyaw,'Vehicle_slip_obd'] = 0.427*((dataset.loc[mask_speedyaw,"SW_pos_obd"]*2*np.pi)/(22*360))
dataset['Yaw_rate_kmb1'] = (dataset['speedo_obd']*(5/18)*np.tan(dataset["SW_pos_obd"]*2*np.pi/(22*360))/1.873)*(180/np.pi)
dataset.loc[~mask_speedyaw,'Vehicle_slip_obd'] = 0

plt.figure(figsize=(10, 6))
plt.plot((dataset['Yaw_rate_kmb1']-dataset['yaw_rate']).abs(),label='KMB 1 vs yaw rate')
plt.plot((dataset['Correvit_slip_angle_COG_corrvittiltcorrected']-dataset['Vehicle_slip_obd']).abs(),label='Correvit vs KMB1')
plt.legend()
plt.show()
plt.close()

#These plots are made to analyse the error between predicted yaw rate vs actual yaw rate to generate confidence for KMB-1
condition =  ((dataset['speedo_obd'] <= 20) & ( dataset["LatAcc_obd"].abs() >= 3))
abs_diff_kmb_yaw = (dataset.loc[condition,'Yaw_rate_kmb1'] - dataset.loc[condition,'yaw_rate']).abs()
bin_edges = np.arange(0, 15+ 0.25, 0.25)
bin_indices = np.digitize(np.abs(dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]-dataset.loc[condition,'Vehicle_slip_obd']), bins=bin_edges)

#abs_diff_regression_df = (dataset.loc[condition,"regression"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()

# Calculate mean absolute error for each bin
mean_abs_yaw_error_per_bin_KMB_1 = []
bin_centers = []
bin_counts = []
for i in range(1, len(bin_edges)):
    bin_mask = (bin_indices == i)
    if np.any(bin_mask):
        mean_abs_yaw_error_kmb1 = np.mean(abs_diff_kmb_yaw[bin_mask])

        mean_abs_yaw_error_per_bin_KMB_1.append(mean_abs_yaw_error_kmb1)

        bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)
        bin_counts.append(np.sum(bin_mask))
# Plot the mean absolute error for each bin
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(bin_centers,mean_abs_yaw_error_per_bin_KMB_1,marker='o', linestyle='-', label='Yaw error KMB1')
ax1.set_xlabel('Error between Correvit vehicle slip angle and KMB-1 (degrees)')
ax1.set_ylabel('Yaw rate error from KMB-1')
ax1.set_title('Yaw rate error between state equations and measurement KMB-1')
ax1.grid(True)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()

# Plot the histogram data as a bar plot on the twin axis
ax2.bar(bin_centers, (bin_counts/sum(bin_counts))*100, width=0.25, color='gray', alpha=0.5, label='Data count per bin')
ax2.set_ylabel('Data Percentage in bin')
ax2.set_ylim(0, max((bin_counts/sum(bin_counts))*100)+5)  # Adjust the ylim for better visualization

# Add legend for the histogram
ax2.legend(loc='upper right')
plt.show()
plt.close()


#Computing vehicle slip angle based on OBD data
#mask_speedo = dataset['speedo_obd'] > 20
#dataset.loc[mask_speedo, 'Vehicle_slip_obd'] = np.nan
#mask_middle_nan = (dataset['Vehicle_slip_obd'].shift(-1).isna() & dataset['Vehicle_slip_obd'].shift(1).isna())
#dataset.loc[mask_middle_nan, 'Vehicle_slip_obd'] = np.nan

#calculating slip angle based on yaw rate and lateral acceleration (KMB-2) failure
# mask_speedyaw = dataset['speedo_obd'].notna() & dataset['yaw_rate'].notna() & dataset['LatAcc_obd'].notna() & (dataset['yaw_rate'].abs() > 2)
# dataset.loc[mask_speedyaw, 'speed_y'] = -1*dataset.loc[mask_speedyaw, 'LatAcc_obd'] / ((dataset.loc[mask_speedyaw, 'yaw_rate'])*(np.pi/180))
# dataset.loc[mask_speedyaw, 'speed_x'] = np.sqrt(dataset.loc[mask_speedyaw, 'speedo_obd']**2 - dataset.loc[mask_speedyaw, 'speed_y']**2)
# dataset.loc[mask_speedyaw,'KMB_slip2'] = ((np.arctan(dataset.loc[mask_speedyaw,'speed_y']/dataset.loc[mask_speedyaw,'speed_x'])*(180/np.pi)))
# dataset.loc[~mask_speedyaw,'KMB_slip2'] = 0
# dataset['KMB_slip2'] = np.arctan((dataset['LatAcc_obd']/ ((dataset['speedo_obd']/3.6) * ((dataset['yaw_rate']*np.pi)/180))))*(180/np.pi)


#Caculating slip angle based on yaw rate and vehicle dimensions (lh= 799, lv= 1074, l= 1873) failure
mask_speedyaw = dataset['speedo_obd']>5 & dataset['yaw_rate'].notna() & dataset['LatAcc_obd'].notna() & (dataset['yaw_rate'].abs() > 2)
dataset.loc[mask_speedyaw,'speed_y'] =  (-1*(dataset.loc[mask_speedyaw,'yaw_rate'])*(np.pi/180)*((1.074*0.799)/1.873))*3.6
dataset.loc[mask_speedyaw,'speed_x'] = np.sqrt(dataset.loc[mask_speedyaw, 'speedo_obd']**2 - dataset.loc[mask_speedyaw, 'speed_y']**2)
dataset.loc[mask_speedyaw,'KMB_slip3'] = ((np.arctan(dataset.loc[mask_speedyaw,'speed_y']/dataset.loc[mask_speedyaw,'speed_x'])*(180/np.pi)))
dataset.loc[~mask_speedyaw,'KMB_slip3'] = 0

#Tire velocities ( This model is working well :))
mask_speedyaw = dataset['speedo_obd']>5
dataset.loc[mask_speedyaw,'speed_y'] =  ((dataset.loc[mask_speedyaw,"VelFL_obd"]+dataset.loc[mask_speedyaw,"VelRL_obd"]) - (dataset.loc[mask_speedyaw,"VelFR_obd"]+dataset.loc[mask_speedyaw,"VelRR_obd"]))/4
dataset.loc[mask_speedyaw,'speed_x'] =  (dataset.loc[mask_speedyaw,"VelFR_obd"]+dataset.loc[mask_speedyaw,"VelFL_obd"]+dataset.loc[mask_speedyaw,"VelRR_obd"]+dataset.loc[mask_speedyaw,"VelRL_obd"])/4
dataset.loc[mask_speedyaw,'KMB_slip2'] = -1*((np.arctan(dataset.loc[mask_speedyaw,'speed_y']/dataset.loc[mask_speedyaw,'speed_x'])*(180/np.pi)))
dataset.loc[~mask_speedyaw,'KMB_slip2'] = 0
dataset.loc[~mask_speedyaw,'Vehicle_slip_obd'] = 0

def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the data using filtfilt
    y = filtfilt(b, a, data)
    return y

cutoff_frequency = 2.0  # Desired cutoff frequency (in Hz)
sampling_rate = 50.0   # Sampling rate of the data (in Hz)
filter_order = 3
dataset['KMB_slip2_fitler'] = butter_lowpass_filter(dataset['KMB_slip2'], cutoff_frequency, sampling_rate, filter_order)
condition =  ((dataset['speedo_obd'] <= 200) & ( dataset["LatAcc_obd"].abs() <= 30))

def bicycle_model(params, data):
    # Vehicle parameters
    m = 1050    # Mass of vehicle
    lf = 1.0732 # Distance from CoG to front axle
    lr = 0.7998 # Distance from CoG to rear axle
    T = 0.02    # The prediction is offset by 10 values
    i_s = 22    # Steering wheel ratio

    #Params to be optimized
    c_alpha_f, c_alpha_r, I_z = params

    vel = data["speedo_obd"]/3.6#data["ins_vel_frame_COG_x"]
    correvit_slip = data["Correvit_slip_angle_COG_corrvittiltcorrected"] * np.pi/180
    psi_dot_meas = -1*data["yaw_rate"]* np.pi/180
    steering =  data["SW_pos_obd"] * (np.pi / 180) / i_s

    slip_pred = np.zeros_like(correvit_slip)
    psi_dot_pred = np.zeros_like(psi_dot_meas)

    # Initialize arrays to store the model predictions
    mask_vel = vel > 2
    slip_pred[mask_vel] = (correvit_slip[mask_vel] - (((c_alpha_f + c_alpha_r) / (m * vel[mask_vel])) * T) * correvit_slip[mask_vel] +
                        (((c_alpha_r * lr - c_alpha_f * lf) / (m * vel[mask_vel] * vel[mask_vel])) - 1) * T * psi_dot_meas[mask_vel] + (
                            c_alpha_f / (m * vel[mask_vel])) * T * steering[mask_vel])
    psi_dot_pred[mask_vel]  = ((c_alpha_r * lr - c_alpha_f * lf) / I_z) * T * correvit_slip[mask_vel] + psi_dot_meas[mask_vel] - (
                ((c_alpha_f * lf * lf + c_alpha_r * lr * lr) / (I_z * vel[mask_vel])) * T) * psi_dot_meas[mask_vel] + (
                               c_alpha_f * lf / I_z) * T * steering[mask_vel]
    return slip_pred*(180/np.pi), psi_dot_pred*(180/np.pi)

params = [40000,63321,1500]
dataset['KMB_3'],random_variable = bicycle_model(params,dataset)

print("kmb2",((dataset.loc[condition, 'KMB_slip2'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean())
print("kmb2 butter worth",((dataset.loc[condition, 'KMB_slip2_fitler'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean())
print("kmb3",((dataset.loc[condition, 'KMB_3'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean())
plt.figure(figsize=(10, 6))
plt.plot(dataset['KMB_slip2'],label="kmb slip 2")
plt.plot(dataset['Correvit_slip_angle_COG_corrvittiltcorrected'],label='GT - Correvit')
plt.plot(dataset['KMB_slip2_fitler'],label='kmb slip 2 butter worth')
plt.legend()
plt.show()
plt.close()

#Kalman filter for tire velocities - KMB-2:
x_n_1 = np.array([[np.float(dataset['Correvit_slip_angle_COG_corrvittiltcorrected'].iloc[0])* np.pi / 180],
                  [np.float(dataset['yaw_rate'].iloc[0])* np.pi / 180]])
p_n_1 = np.array([[20,0],[0,20]]) #Need to initialize
kalman_slip = [np.float(dataset['Correvit_slip_angle_COG_corrvittiltcorrected'].iloc[0])]
kalman_variance = [0]
for i in range(len(dataset)-1):
    i=i+1
    #y_n = np.array([[np.float(dataset["KMB_slip2"].iloc[i])*np.pi/180],[np.float(dataset['yaw_rate'].iloc[i]*np.pi/180)]])
    y_n = np.array([[np.float(dataset['KMB_slip2'].iloc[i]) * np.pi / 180], [np.float(dataset['yaw_rate'].iloc[i] * np.pi / 180)]])
    u_n_1 = np.array([[(np.float(dataset["SW_pos_obd"].iloc[i-1])*np.pi)/(i_s*180)],[0]])
    if dataset['speedo_obd'].iloc[i-1]<1:
        matrix_a, matrix_b = state_matrix(1)
    else:
        matrix_a, matrix_b = state_matrix(np.float(dataset['speedo_obd'].iloc[i-1])/3.6)
    x_a_prior, p_a_prior = pred_a_prior(matrix_a,matrix_b,matrix_g,x_n_1,u_n_1,p_n_1)
    innovation, inno_residual, gain = inno_gain_covariance(y_n,matrix_c,x_a_prior,p_a_prior,Cnm)
    x_post, p_post = update_posterior (x_a_prior,gain,innovation,matrix_c,p_a_prior)
    kalman_slip.append(float(x_post[0])*180/np.pi)
    kalman_variance.append(float(p_post[0,0])*180/np.pi)
    x_n_1 = x_post
    p_n_1 = p_post

dataset['kalman_slip_kmb2']=kalman_slip
dataset["kalman_slip_kmb2_variance"] = kalman_variance
dataset["fused_slip"] = pred_first_elements_1d
dataset["fused_slip_2"] = pred_first_elements_1d
dataset.loc[~mask_speedyaw,'kalman_slip_kmb2'] = 0
plt.figure(figsize=(10, 6))

plt.plot(dataset['Correvit_slip_angle_COG_corrvittiltcorrected'],label='GT - Correvit')
plt.plot(dataset['kalman_slip_kmb2'],label='kalman slip')
plt.plot(dataset['KMB_slip2'],label='KMB-slip2')
plt.legend()
plt.show()
plt.close()


# Analyzing kalman filter slip and variance
"""1) At very low velocities and standstill the variance shoots up
Reason - the state equations have divided by v and v2 in it making it shoot up
The measurement of slip angle from virtual sensor in this area is made to be 0 from the hard condition of velocity

2) why variance is high

3) Model is following the state equation more"""

plt.figure(figsize=(10, 6))
#plt.plot(np.abs(dataset['Correvit_slip_angle_COG_corrvittiltcorrected']-dataset['kalman_slip_kmb2']),label='GT - Correvit')
plt.plot(dataset["kalman_slip_kmb2_variance"],label="kalman a prior")
plt.plot(dataset['Correvit_slip_angle_COG_corrvittiltcorrected'],label='GT - Correvit')
plt.plot(dataset['kalman_slip_kmb2'],label='kalman slip')
plt.plot(dataset['KMB_slip2'],label='KMB-slip2')
plt.legend()
plt.show()
plt.close()

condition =  ((dataset['speedo_obd'] <= 20) & ( dataset["LatAcc_obd"].abs() <= 3))
abs_variance_kmb2_kalman = dataset.loc[condition,"kalman_slip_kmb2_variance"]
bin_edges = np.arange(0, 15+ 0.25, 0.25)
bin_indices = np.digitize(np.abs(dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]-dataset.loc[condition,'kalman_slip_kmb2']), bins=bin_edges)

#abs_diff_regression_df = (dataset.loc[condition,"regression"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()

# Calculate mean absolute error for each bin
dataset["kalman_slip_kmb2_variance"]  = dataset["kalman_slip_kmb2_variance"] #need to remove this line
mean_abs_variance_per_bin_KMB_2_kalman = []
bin_centers = []
bin_counts = []
for i in range(1, len(bin_edges)):
    bin_mask = (bin_indices == i)
    if np.any(bin_mask):
        mean_abs_variance_kmb2_kalman = np.mean(abs_variance_kmb2_kalman[bin_mask])
        mean_abs_variance_per_bin_KMB_2_kalman.append(mean_abs_variance_kmb2_kalman)
        bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)
        bin_counts.append(np.sum(bin_mask))
# Plot the mean absolute error for each bin
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(bin_centers,mean_abs_variance_per_bin_KMB_2_kalman,marker='o', linestyle='-', label='KMB2 kalman variance')
ax1.set_xlabel('Error between Correvit vehicle slip angle and KMB-2 (degrees)')
ax1.set_ylabel('Variance from KMB-2')
ax1.set_title('Varaince significance analysis KMB-2')
ax1.grid(True)
ax1.legend(loc='upper left')
ax2 = ax1.twinx()

# Plot the histogram data as a bar plot on the twin axis
ax2.bar(bin_centers, (bin_counts/sum(bin_counts))*100, width=0.25, color='gray', alpha=0.5, label='Data count per bin')
ax2.set_ylabel('Data Percentage in bin')
ax2.set_ylim(0, max((bin_counts/sum(bin_counts))*100)+5)  # Adjust the ylim for better visualization

# Add legend for the histogram
ax2.legend(loc='upper right')
plt.show()
plt.close()










#Computing fusion block with uncertainty from the model
for i in range(len(dataset)):
    variance = np.sqrt(dataset["variance_informer"].iloc[i])
    if variance > 0.25:
        if variance > 1:
            #dataset["variance_informer"].iloc[i] = 1 # just for making plotting easy
            variance = 1
        # dataset["fused_slip"].iloc[i] = (dataset['Vehicle_slip_obd'].iloc[i] * variance + dataset["fused_slip"].iloc[i] * (1 - variance))

    if variance > 0.25:
        if dataset["speedo_obd"].iloc[i] < 20 and dataset["LatAcc_obd"].iloc[i] < 3:
            dataset["fused_slip"].iloc[i] = (dataset['Vehicle_slip_obd'].iloc[i] * variance + dataset["fused_slip"].iloc[i] * (1 - variance))
        elif dataset["speedo_obd"].iloc[i] < 20 and dataset["LatAcc_obd"].iloc[i] >= 3 and np.abs(dataset["pred_informer"].iloc[i]) < 5:
            dataset["fused_slip"].iloc[i] = (dataset['KMB_slip2'].iloc[i] * variance + dataset["fused_slip"].iloc[i] * (1 - variance))
        elif dataset["speedo_obd"].iloc[i] < 20 and dataset["LatAcc_obd"].iloc[i] >= 3 and np.abs(dataset["pred_informer"].iloc[i]) >= 5:
            dataset["fused_slip"].iloc[i] = (dataset['Vehicle_slip_obd'].iloc[i] * variance + dataset["fused_slip"].iloc[i] * (1 - variance))
        elif dataset["speedo_obd"].iloc[i] > 20 and np.abs(dataset["pred_informer"].iloc[i]) < 5:
            dataset["fused_slip"].iloc[i] = (dataset['KMB_slip2'].iloc[i] * variance + dataset["fused_slip"].iloc[i] * (1 - variance))
        else:
            dataset["fused_slip"].iloc[i] = dataset["pred_informer"].iloc[i]

    #Reducing the velocity to zero
    if dataset["speedo_obd"].iloc[i] < 2:
        dataset["fused_slip"].iloc[i] = 0
        dataset["Vehicle_slip_obd"].iloc[i] = 0

variance_kmb2_uncertainty = []
#Fusion 2 architecture
for i in range(len(dataset)):
    variance = np.sqrt(dataset["variance_informer"].iloc[i])
    if variance > 0.25:
        if variance > 1.5:
            #dataset["variance_informer"].iloc[i] = 1 # just for making plotting easy
            variance = 1.5
        # dataset["fused_slip"].iloc[i] = (dataset['Vehicle_slip_obd'].iloc[i] * variance + dataset["fused_slip"].iloc[i] * (1 - variance))

    if variance > 0.25:
        x = (np.abs(dataset['Yaw_rate_kmb1'].iloc[i] - dataset['yaw_rate'].iloc[i]))/100
        y = dataset["kalman_slip_kmb2_variance"].iloc[i]/1e6
        z = variance
        dataset["fused_slip_2"].iloc[i] = ((x*y*z)/(y*x+x*z+y*z))*((1/x)*dataset["Vehicle_slip_obd"].iloc[i]+(1/y)*dataset['KMB_slip2'].iloc[i]+(1/z)*dataset["fused_slip_2"].iloc[i])

    #Reducing the velocity to zero
    if dataset["speedo_obd"].iloc[i] < 2:
        dataset["fused_slip_2"].iloc[i] = 0
        dataset["Vehicle_slip_obd"].iloc[i] = 0

plt.figure(figsize=(10, 6))
plt.plot(dataset["fused_slip_2"], label='fused slip2')
plt.plot(dataset["fused_slip"], label='fused slip1')
plt.plot(dataset["Correvit_slip_angle_COG_corrvittiltcorrected"], label='GT')
plt.legend()
plt.show()
#calculating slip angle errors (numerical for entire dataset)
condition =  ((dataset['speedo_obd'] <= 200) & ( dataset["LatAcc_obd"].abs() <= 30))
print('condition percentage', (sum(condition)/len(condition))*100)
slip_values = ((dataset.loc[condition, 'Vehicle_slip_obd'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean()
informer_values = ((dataset.loc[condition, 'pred_informer'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean()
fused_values = ((dataset.loc[condition, 'fused_slip'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean()
slip_values_kmb2 = ((dataset.loc[condition, 'KMB_slip2'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean()
slip_values_kmb2_kalman = ((dataset.loc[condition, 'kalman_slip_kmb2'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean()
slip_values_fused_2 = ((dataset.loc[condition, 'fused_slip_2'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean()
print('Informer - mean', informer_values)
print('fused - mean', fused_values)
print('fused 2 - mean', slip_values_fused_2)
print('KMB - mean ', slip_values)
print('KMB 2 - mean ', slip_values_kmb2)
print('KMB 2- mean kalman', slip_values_kmb2_kalman)
#Plots
#condition = ((dataset['speedo_obd'] >= 20 ) & (dataset["LatAcc_obd"].abs() < 3))
#condition = dataset['speedo_obd'] < 1000 #If you dont want conditions

#Comparing with Correvit Slip angle
abs_diff_informer = (dataset.loc[condition,"pred_informer"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
abs_diff_kmb = (dataset.loc[condition,"Vehicle_slip_obd"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
abs_diff_kmb_2 = (dataset.loc[condition,"KMB_slip2"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
abs_diff_kmb_2_kalman = (dataset.loc[condition,'kalman_slip_kmb2'] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
abs_diff_kmb_2_kalman = (dataset.loc[condition,'KMB_3'] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
abs_diff_fused_slip = (dataset.loc[condition,"fused_slip"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
bin_edges = np.arange(dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].min(), dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].max() + 0.25, 0.25)
bin_indices = np.digitize(dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"], bins=bin_edges)

#Comparing with ADMA slip COG
#abs_diff_informer = (dataset.loc[condition,"pred_informer"] - dataset.loc[condition,"side_slip_angle_COG"]).abs()
#abs_diff_kmb = (dataset.loc[condition,"Vehicle_slip_obd"] - dataset.loc[condition,"side_slip_angle_COG"]).abs()
#abs_diff_kmb_2 = (dataset.loc[condition,"KMB_slip2"] - dataset.loc[condition,"side_slip_angle_COG"]).abs()
#abs_diff_kmb_2_kalman = (dataset.loc[condition,'kalman_slip_kmb2'] - dataset.loc[condition,"side_slip_angle_COG"]).abs()
#abs_diff_fused_slip = (dataset.loc[condition,"fused_slip"] - dataset.loc[condition,"side_slip_angle_COG"]).abs()
#bin_edges = np.arange(dataset["side_slip_angle_COG"].min(), dataset["side_slip_angle_COG"].max() + 0.25, 0.25)
#bin_indices = np.digitize(dataset.loc[condition,"side_slip_angle_COG"], bins=bin_edges)
#abs_diff_regression_df = (dataset.loc[condition,"regression"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()

# Calculate mean absolute error for each bin
mean_abs_error_per_bin_informer = []
mean_abs_error_per_bin_kmb = []
mean_abs_error_per_bin_kmb_2 = []
mean_abs_error_per_bin_kmb_2_kalman = []
#mean_abs_error_per_bin_regression = []
mean_abs_error_per_bin_fused_slip = []
bin_centers = []

for i in range(1, len(bin_edges)):
    bin_mask = (bin_indices == i)
    if np.any(bin_mask):
        mean_abs_error_informer = np.mean(abs_diff_informer[bin_mask])
        mean_abs_error_kmb = np.mean(abs_diff_kmb[bin_mask])
        mean_abs_error_kmb_2 = np.mean(abs_diff_kmb_2[bin_mask])
        mean_abs_error_kmb_2_kalman = np.mean(abs_diff_kmb_2_kalman[bin_mask])
        #mean_abs_error_regression = np.mean(abs_diff_regression_df[bin_mask])
        mean_abs_error_fused_slip = np.mean(abs_diff_fused_slip[bin_mask])

        mean_abs_error_per_bin_informer.append(mean_abs_error_informer)
        mean_abs_error_per_bin_kmb.append(mean_abs_error_kmb)
        mean_abs_error_per_bin_kmb_2.append(mean_abs_error_kmb_2)
        mean_abs_error_per_bin_kmb_2_kalman.append(mean_abs_error_kmb_2_kalman)
        #mean_abs_error_per_bin_regression.append(mean_abs_error_regression)
        mean_abs_error_per_bin_fused_slip.append(mean_abs_error_fused_slip)
        bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)

# Plot the mean absolute error for each bin
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, mean_abs_error_per_bin_informer, marker='o', linestyle='-', label='Informer MAE')
plt.plot(bin_centers, mean_abs_error_per_bin_kmb, marker='o', linestyle='-',  label='KMB MAE')
plt.plot(bin_centers, mean_abs_error_per_bin_kmb_2, marker='o', linestyle='-',  label='KMB2 Tire MAE')
plt.plot(bin_centers, mean_abs_error_per_bin_kmb_2_kalman, marker='o', linestyle='-',  label='KMB2 Tire MAE kalman')
plt.plot(bin_centers, mean_abs_error_per_bin_fused_slip, marker='o', linestyle='-',  label='Fused slip MAE')
plt.xlabel('Correvit vehicle slip angle (degrees)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Mean Absolute Error between prediction vs correvit slip for 0.25 bins')
plt.grid(True)
plt.legend()
plt.show()
plt.close()

########Percentage########
mean_abs_error_per_bin_informer = []
mean_abs_error_per_bin_kmb = []
mean_abs_error_per_bin_kmb_2 = []
mean_abs_error_per_bin_kmb_2_kalman = []
mean_abs_error_per_bin_regression = []
mean_abs_error_per_bin_fused_slip = []
bin_centers = []
bin_counts = []

for i in range(1, len(bin_edges)):
    bin_mask = (bin_indices == i)
    if np.any(bin_mask):
        mean_abs_error_informer = np.mean(abs_diff_informer[bin_mask])
        mean_abs_error_kmb = np.mean(abs_diff_kmb[bin_mask])
        mean_abs_error_kmb_2 = np.mean(abs_diff_kmb_2[bin_mask])
        mean_abs_error_kmb_2_kalman = np.mean(abs_diff_kmb_2_kalman[bin_mask])
        #mean_abs_error_regression = np.mean(abs_diff_regression_df[bin_mask])
        mean_abs_error_fused_slip = np.mean(abs_diff_fused_slip[bin_mask])

        mean_abs_error_per_bin_informer.append(mean_abs_error_informer)
        mean_abs_error_per_bin_kmb.append(mean_abs_error_kmb)
        mean_abs_error_per_bin_kmb_2.append(mean_abs_error_kmb_2)
        #mean_abs_error_per_bin_regression.append(mean_abs_error_regression)
        mean_abs_error_per_bin_fused_slip.append(mean_abs_error_fused_slip)
        mean_abs_error_per_bin_kmb_2_kalman.append(mean_abs_error_kmb_2_kalman)
        bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)
        bin_counts.append(np.sum(bin_mask))  # Number of data points in each bin

# Plot the mean absolute error for each bin
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(bin_centers, mean_abs_error_per_bin_informer, marker='o', linestyle='-', label='Informer MAE')
ax1.plot(bin_centers, mean_abs_error_per_bin_kmb, marker='o', linestyle='-',  label='KMB MAE')
ax1.plot(bin_centers, mean_abs_error_per_bin_kmb_2, marker='o', linestyle='-',  label='KMB2 MAE')
#ax1.plot(bin_centers, mean_abs_error_per_bin_kmb_2_kalman, marker='o', linestyle='-',  label='KMB2 Kalman Filter')
ax1.plot(bin_centers, mean_abs_error_per_bin_fused_slip, marker='o', linestyle='-',  label='Fused slip MAE')

ax1.set_xlabel('ADMA Vehicle slip angle COG vehicle slip angle (degrees)')
ax1.set_ylabel('Mean Absolute Error (MAE)')
ax1.set_title('Mean Absolute Error between prediction vs adma slip for 0.25 bins')
ax1.grid(True)
ax1.legend(loc='upper left')

# Create a twin axis for the histogram
ax2 = ax1.twinx()

# Plot the histogram data as a bar plot on the twin axis
ax2.bar(bin_centers, (bin_counts/sum(bin_counts))*100, width=0.25, color='gray', alpha=0.5, label='Data count per bin')
ax2.set_ylabel('Data Percentage in bin')
ax2.set_ylim(0, max((bin_counts/sum(bin_counts))*100)+5)  # Adjust the ylim for better visualization

# Add legend for the histogram
ax2.legend(loc='upper right')
plt.show()

######################################################################################################################
#Plot histogram with bins of 0.5 degrees
bins = np.arange(0, np.max(abs_diff_informer) + 0.1, 0.1)
hist, bin_edges = np.histogram(abs_diff_informer, bins=bins)
hist_percent = (hist / hist.sum()) * 100

plt.bar(bin_edges[:-1], hist_percent, width=0.1, edgecolor='black', align='edge')
plt.xlabel('Absolute Difference vehicle slip angle (degrees)')
plt.ylabel('Frequency (%)')
plt.title('Histogram of Absolute Difference between correvit slip angle vs informer predictions')
plt.grid(True)
for i in range(len(hist)):
    plt.text(bin_edges[i]+0.05, hist_percent[i] + 0.5, f'{hist_percent[i]:.1f}%', ha='center')
plt.show()
plt.close()


# Plot true vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(dataset["true_informer"], label='Ground truth',color = "g")
plt.plot(dataset["pred_informer"], label='Informer prediction',color = "b")
plt.plot(dataset["fused_slip"], label='Fused slip')
#plt.fill_between(dataset.index, dataset["pred_informer"] - 2 * np.sqrt(dataset["variance_informer"]), dataset["pred_informer"] + 2 * np.sqrt(dataset["variance_informer"]), color='r', alpha=0.3, label='Uncertainty (2 std)')
#plt.plot(dataset["true_informer"], label='Ground truth',color = "g")
#plt.plot(dataset["regression"], label='MLP prediction',color = "r")
plt.plot(dataset['Vehicle_slip_obd'], label='KMB slip angle - 1',color = "orange")
#plt.plot(dataset['KMB_slip2'], label='KMB slip angle - 2',color = "orange")
plt.plot(dataset["kalman_slip_kmb2"], label='KMB slip angle KF - 2',color = "red")
plt.xlabel('Time stamps')
plt.ylabel('Slip angle in degrees')
plt.title("True vs Informer vehicle slip angle prediction")
plt.legend()
plt.show()
