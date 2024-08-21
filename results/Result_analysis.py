#result _ file: informer_OBD_ADMA_ftMS_sl100_ll25_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0
#Dataset _ path:
""""The goal of this code is to visualize the slip angle predicted vs ground truth slip angle (Correvit vs ADMA)
Then we compare this OBD based slip angles"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

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

data_pred_regression = np.load(f'results/Regression/pred_regression_fivehours.npy')
data_true_regression = np.load(f'results/Regression/true_regression_fivehours.npy')
pred_regression_first_elements = data_pred_regression[:-25, 0, :] #-25 here seems right and matches with dataloaders
true_regression_first_elements = data_true_regression[:-25, 0, :]

pred_first_elements = data_pred[:, 0, :]
true_first_elements = data_true[:, 0, :]
variance_first_elements = data_variance[:, 0, :]
"""Check this code block whether things are matching"""
#Now we extract the test dataset from the dataset to compare things like obd based slip angle
dataset = dataset[-len(pred_first_elements)-30:-30] #-30 here seems right and matches with dataloaders

#Still there might be some mismatch between the two due to some padding in the time series (check it here)
#index = np.where(true_first_elements[:] == dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].iloc[0])[0].item()
index = 0
dataset["true_informer"] = np.nan
true_first_elements_1d = true_first_elements[:, 0].flatten()
dataset["true_informer"]=true_first_elements_1d
#Check whether this is correct and true_informer matches with correvit slip angle in dataset
dataset["pred_informer"] = np.nan
pred_first_elements_1d = pred_first_elements[:, 0].flatten()
dataset["pred_informer"]=pred_first_elements_1d

#regression block
dataset["regression"] = np.nan
regression_first_elements_1d = pred_regression_first_elements[:, 0].flatten()
dataset["regression"]= regression_first_elements_1d

#variance block
dataset["variance_informer"] = np.nan
variance_first_elements_1d = variance_first_elements[:, 0].flatten()
dataset["variance_informer"]=variance_first_elements_1d

plt.figure(figsize=(10, 6))
plt.plot(np.sqrt(dataset['variance_informer']))
plt.show()

#Calculation of OBD based slip value for comparision (KMB-1)
dataset["SW_pos_obd"] = dataset["SW_pos_obd"] - 6
play = 20
mask_steering = ((dataset["SW_pos_obd"] <= 0) & (dataset["SW_pos_obd"] >= -play))
dataset.loc[mask_steering,'SW_pos_obd'] = 0
mask_steering = ((dataset["SW_pos_obd"] <= play) & (dataset["SW_pos_obd"] >= 0))
dataset.loc[mask_steering,'SW_pos_obd'] = 0
dataset.loc[dataset["SW_pos_obd"] > 0, "SW_pos_obd"] -= play
dataset.loc[dataset["SW_pos_obd"] < 0, "SW_pos_obd"] += play
dataset['Vehicle_slip_obd'] = (np.arcsin(np.tan((dataset["SW_pos_obd"]*2*np.pi)/(22*360))*0.427))*(180/np.pi) #lh=799,8,l=1873 for IssaK
#Computing vehicle slip angle based on OBD data
#mask_speedo = dataset['speedo_obd'] > 20
#dataset.loc[mask_speedo, 'Vehicle_slip_obd'] = np.nan
#mask_middle_nan = (dataset['Vehicle_slip_obd'].shift(-1).isna() & dataset['Vehicle_slip_obd'].shift(1).isna())
#dataset.loc[mask_middle_nan, 'Vehicle_slip_obd'] = np.nan

#calculating slip angle based on yaw rate and lateral acceleration (KMB-2)
mask_speedyaw = dataset['speedo_obd'].notna() & dataset['yaw_rate'].notna() & dataset['LatAcc_obd'].notna() & (dataset['yaw_rate'].abs() > 2)
dataset.loc[mask_speedyaw, 'speed_y'] = -1*dataset.loc[mask_speedyaw, 'LatAcc_obd'] / ((dataset.loc[mask_speedyaw, 'yaw_rate'])*(np.pi/180))
dataset.loc[mask_speedyaw, 'speed_x'] = np.sqrt(dataset.loc[mask_speedyaw, 'speedo_obd']**2 - dataset.loc[mask_speedyaw, 'speed_y']**2)
dataset.loc[mask_speedyaw,'KMB_slip2'] = ((np.arctan(dataset.loc[mask_speedyaw,'speed_y']/dataset.loc[mask_speedyaw,'speed_x'])*(180/np.pi)))
dataset.loc[~mask_speedyaw,'KMB_slip2'] = 0
#dataset['KMB_slip2'] = np.arctan((dataset['LatAcc_obd']/ ((dataset['speedo_obd']/3.6) * ((dataset['yaw_rate']*np.pi)/180))))*(180/np.pi)


dataset["fused_slip"] = pred_first_elements_1d
#Computing fusion block with uncertainty from the model
for i in range(len(dataset)):
    if np.sqrt(dataset["variance_informer"].iloc[i]) > 0.3:
        x = np.sqrt(dataset["variance_informer"].iloc[i])
        dataset["fused_slip"].iloc[i] = (dataset['Vehicle_slip_obd'].iloc[i] * 0.5 + dataset["fused_slip"].iloc[i] * 0.5)

    if dataset["speedo_obd"].iloc[i] < 0.1:
        dataset["fused_slip"].iloc[i] = 0
        dataset["Vehicle_slip_obd"].iloc[i] = 0


#calculating slip angle errors (numerical for entire dataset)
condition =  ((dataset['speedo_obd'] <= 20) & ( dataset["LatAcc_obd"].abs() <= 3))
print('condition percentage', (sum(condition)/len(condition))*100)
slip_values = ((dataset.loc[condition, 'Vehicle_slip_obd'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean()
informer_values = ((dataset.loc[condition, 'pred_informer'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean()
fused_values = ((dataset.loc[condition, 'fused_slip'] - dataset.loc[condition,'Correvit_slip_angle_COG_corrvittiltcorrected']).abs()).mean()
#Plots
#condition = ((dataset['speedo_obd'] >= 20 ) & (dataset["LatAcc_obd"].abs() < 3))
#condition = dataset['speedo_obd'] < 1000 #If you dont want conditions
abs_diff_informer = (dataset.loc[condition,"pred_informer"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
abs_diff_kmb = (dataset.loc[condition,"Vehicle_slip_obd"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
abs_diff_fused_slip = (dataset.loc[condition,"fused_slip"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()
bin_edges = np.arange(dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].min(), dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].max() + 0.25, 0.25)
bin_indices = np.digitize(dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"], bins=bin_edges)

abs_diff_regression_df = (dataset.loc[condition,"regression"] - dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]).abs()

# Calculate mean absolute error for each bin
mean_abs_error_per_bin_informer = []
mean_abs_error_per_bin_kmb = []
mean_abs_error_per_bin_regression = []
mean_abs_error_per_bin_fused_slip = []
bin_centers = []

for i in range(1, len(bin_edges)):
    bin_mask = (bin_indices == i)
    if np.any(bin_mask):
        mean_abs_error_informer = np.mean(abs_diff_informer[bin_mask])
        mean_abs_error_kmb = np.mean(abs_diff_kmb[bin_mask])
        mean_abs_error_regression = np.mean(abs_diff_regression_df[bin_mask])
        mean_abs_error_fused_slip = np.mean(abs_diff_fused_slip[bin_mask])

        mean_abs_error_per_bin_informer.append(mean_abs_error_informer)
        mean_abs_error_per_bin_kmb.append(mean_abs_error_kmb)
        mean_abs_error_per_bin_regression.append(mean_abs_error_regression)
        mean_abs_error_per_bin_fused_slip.append(mean_abs_error_fused_slip)
        bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)

# Plot the mean absolute error for each bin
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, mean_abs_error_per_bin_informer, marker='o', linestyle='-', label='Informer MAE')
plt.plot(bin_centers, mean_abs_error_per_bin_kmb, marker='o', linestyle='-',  label='KMB MAE')
#plt.plot(bin_centers, mean_abs_error_per_bin_regression, marker='o', linestyle='-',  label='MLP MAE')
plt.plot(bin_centers, mean_abs_error_per_bin_fused_slip, marker='o', linestyle='-',  label='Fused slip MAE')
plt.xlabel('Correvit vehicle slip angle (degrees)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Mean Absolute Error between prediction vs correvit slip for 0.25 bins')
plt.grid(True)
plt.legend()
plt.show()
plt.close()


# Plot histogram with bins of 0.5 degrees
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
#plt.fill_between(dataset.index, dataset["pred_informer"] - 2 * np.sqrt(dataset["variance_informer"]), dataset["pred_informer"] + 2 * np.sqrt(dataset["variance_informer"]), color='r', alpha=0.3, label='Uncertainty (2 std)')
plt.plot(dataset["regression"], label='MLP prediction',color = "r")
plt.plot(dataset['Vehicle_slip_obd'], label='KMB slip angle - 1',color = "orange")
#plt.plot(dataset["KMB_slip2"], label='KMB slip angle - 2',color = "red")
plt.xlabel('Time stamps')
plt.ylabel('Slip angle in degrees')
plt.title('True vs Informer vehicle slip angle prediction')
plt.legend()
plt.show()
