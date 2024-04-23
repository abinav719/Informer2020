import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import signal

"""The goal of this code is to estimate vehicle slip angle from OBD measurements data through simpler vehicle dynamics models
1) At low speed cornering below 20Kmphr, single track motion model ignoring the slip angle of the tire is used
2) At high speed corning above 20Kmphr, 
3) Using the velocity of each tire and yaw rate the vehicle slip angle can be estimated"""
"""Problem of yaw rate between ADMA and OBD sensor yaw rate. The yaw rates are calculated at different axes positions making a huge difference"""
"""Solution for cog point calculation: The velocities calculated at the ADMA can be translated to cog considering the rotation of the three axis"""

folder_name = 'Data_set_unzipped'
file_names = os.listdir(folder_name)
frames = []
for i in file_names:
    frames.append(pd.read_csv(f"{folder_name}/{i}/{i}ADMA+OBD_synced.csv"))
dataset = pd.concat(frames)
#i = '2023-03-10-06-59-30'
#dataset = pd.read_csv(f"{folder_name}/{i}/{i}ADMA+OBD_synced.csv")
dataset = dataset[(dataset["status_gnss_mode"]==8) & (dataset["status_speed"]==2)] #Filtering settled values after Kalman settled
dataset.reset_index(drop=True, inplace=True)

#Speedo error correction - Sometimes standstill car has speedo values above 4094 Kmphr and velocity of tires also has an error
#This error is corrected back to 0 velocity
condition = (dataset['speedo_obd'] >= 4094) & (dataset['status_standstill'] == 1)
columns_to_update = ['VelFR_obd', 'VelFL_obd', 'VelRR_obd', 'VelRL_obd', 'speedo_obd']
dataset.loc[condition, columns_to_update] = 0
dataset["ins_vel"] = np.sqrt(dataset["ins_vel_hor_z"]**2 +  dataset["ins_vel_hor_x"]**2 + dataset["ins_vel_hor_y"]**2)
dataset["ins_vel_kmph"] =  dataset["ins_vel"]*(3.6)

"""Problem of yaw rate between ADMA and OBD sensor yaw rate. The yaw rates should theoretically be same. A detailed analysis on yaw rates
Max yaw rate from ADMA = 77 and negative direction is -71 degrees
OBD yaw rate has a max value of 327 for some observations which is a sensor error and this was treated as nan"""
dataset.loc[dataset["yaw_rate"]<-100,"yaw_rate"]  = np.nan
mask_yawrate = dataset["yaw_rate"].notna()
dataset.loc[mask_yawrate ,"yaw_rate"] = dataset.loc[mask_yawrate ,"yaw_rate"]*-1 #OBD and ADMA data are at different axis
ground_truth = dataset.loc[mask_yawrate,"rate_hor_z"].values
predictions = dataset.loc[mask_yawrate,"yaw_rate"].values
residuals = predictions - ground_truth
mae = mean_absolute_error(ground_truth, predictions)
mse = mean_squared_error(ground_truth, predictions)
rmse = mean_squared_error(ground_truth, predictions, squared=False)
r2 = r2_score(ground_truth, predictions)
# Plot ground truth vs. predictions
# plt.figure(figsize=(8, 6))
# plt.scatter(ground_truth, predictions, color='blue', label='Predictions')
# plt.plot(ground_truth, ground_truth, color='red', linestyle='--', label='Ground Truth')
# plt.xlabel('Ground Truth')
# plt.ylabel('Predictions')
# plt.title('Ground Truth vs. Predictions')
# plt.legend()
# plt.grid(True)
# plt.show()

# sns.kdeplot(residuals, color='blue', fill=True, bw_adjust=0.5)
# plt.xlabel('Residuals')
# plt.ylabel('Density')
# plt.title('Density Plot of Residuals')
# plt.grid(True)
# plt.xticks([i for i in range(-50,50,5)])
# plt.show()

#Signal correlation helps to get the lag between ADMA and OBD measurements
correlation = signal.correlate(ground_truth, predictions, mode="full")
lags = signal.correlation_lags(ground_truth.size, predictions.size, mode="full")
lag = lags[np.argmax(correlation)]

# Plot cross-correlation
plt.figure(figsize=(8, 6))
plt.plot(lags[(lags >= -2000) & (lags <= 2000)], correlation[(lags >= -2000) & (lags <= 2000)])
plt.axvline(x=lag, color='r', linestyle='--', label='Peak lag')
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between Ground Truth and Predictions')
plt.legend()
plt.grid(True)
plt.show()

#Experiment to check the lag between OBD and ADMA signal
mask_speedo = dataset["speedo_obd"].notna() & (dataset["acc_hor_x"] > 0.1)
ground_truth = dataset.loc[mask_speedo,"ins_vel_kmph"].values
predictions = dataset.loc[mask_speedo,"speedo_obd"].values
correlation = signal.correlate(ground_truth, predictions, mode="same")
lags = signal.correlation_lags(ground_truth.size, predictions.size, mode="same")
lag = lags[np.argmax(correlation)]
# def lag_finder(y1, y2, sr):
#     n = len(y1)
#
#     corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])
#
#     delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
#     delay = delay_arr[np.argmax(corr)]
#     print('y2 is ' + str(delay) + ' behind y1')
#
#     plt.figure()
#     plt.plot(delay_arr, corr)
#     plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
#     plt.xlabel('Lag')
#     plt.ylabel('Correlation coeff')
#     plt.show()
# y1 = ground_truth
# y2 = predictions
# sr = len(ground_truth)
# lag_finder(y1, y2, sr)
# Plot cross-correlation
plt.figure(figsize=(8, 6))
plt.plot(lags[(lags >= -100) & (lags <= 100)], correlation[(lags >= -100) & (lags <= 100)])
plt.axvline(x=lag, color='r', linestyle='--', label='Peak lag')
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between Ground Truth and Predictions')
plt.legend()
plt.grid(True)
plt.show()



#Calculation of vehicle side slip angle from steering angle and centre of mass positions
mask_sw = dataset['SW_pos_obd'].notna()
dataset.loc[mask_sw, 'Vehicle_slip_obd'] = (np.arcsin(np.tan((dataset["SW_pos_obd"]*2*np.pi)/(22*360))*0.427))*(180/np.pi) #lh=799,8,l=1873 for IssaK

#Filtering and removing the vehicle slip angle for speedo velocity greater than 20Kmphr
mask_speedo = dataset['speedo_obd'] > 20
dataset.loc[mask_speedo, 'Vehicle_slip_obd'] = np.nan

mask_middle_nan = (dataset['Vehicle_slip_obd'].shift(-1).isna() & dataset['Vehicle_slip_obd'].shift(1).isna())
dataset.loc[mask_middle_nan, 'Vehicle_slip_obd'] = np.nan
print('hi')


#Estimation of velocity at the COG of the vehicle using coordinate frame transfomrations refer Prof Michael slides and books.
#Basically v_cog = V_adma + omega X pivot (adma--cog) The position vector of pivot is from ADMA to COG

arm = np.array([0.923, 0.099, 0.381])
#Rotation rates are in degrees, So need to convert to radians/Sec
angular_conversion = (np.pi/180)
def calculate_cross_product(row):
    skew_symmetric_matrix = np.array([
        [0, -row['rate_hor_z']*angular_conversion, row['rate_hor_y']*angular_conversion],
        [row['rate_hor_z']*angular_conversion, 0, -row['rate_hor_x']*angular_conversion],
        [-row['rate_hor_y']*angular_conversion, row['rate_hor_x']*angular_conversion, 0]
    ])
    return np.dot(skew_symmetric_matrix, arm)
#This is actual conversion formula, this formulation is slow so direct implementation is done below with direct formulas
#cross_products = dataset.apply(calculate_cross_product, axis=1)

dataset["INS_velCOG_hor_x"] =  dataset["ins_vel_hor_x"] + (-1*dataset['rate_hor_z']*angular_conversion*arm[1] + dataset['rate_hor_y']*angular_conversion*arm[2])
dataset["INS_velCOG_hor_y"] =  dataset["ins_vel_hor_y"] + (dataset['rate_hor_z']*angular_conversion*arm[0]-dataset['rate_hor_x']*angular_conversion*arm[2])
dataset["INS_velCOG_hor_z"] =  dataset["ins_vel_hor_z"] + (-1*dataset['rate_hor_y']*angular_conversion*arm[0]+ dataset['rate_hor_x']*angular_conversion*arm[1])
#Purposefully left z as car wont have velocity in the z direction
dataset["INS_totalvelCOG_hor"] = np.sqrt(dataset["INS_velCOG_hor_x"]**2 + dataset["INS_velCOG_hor_y"]**2)
dataset["INS_totalvelCOG_hor_kmph"] = dataset["INS_totalvelCOG_hor"]*3.6
mask_insvel =  dataset["INS_totalvelCOG_hor_kmph"] > 1
dataset.loc[mask_insvel, "INS_slip_angle_COG"] = np.arctan(dataset["INS_velCOG_hor_y"]/dataset["INS_velCOG_hor_x"])*(180/np.pi)


#Correvit velocity space correction and slip angle calculation
#Correct the direction of velocity vector in the slip angle for correvit sensor
#The correvit sensor is giving velocity vector erratically which could be corrected with values from INS_velocity vector.
#For our driving dataset, mostly reverse is not used and with the coordinate axis the velocity should always be positive. Except fluctuations at 0 to -0.1 due to sensor fluctioantions
dataset["ext_vel_x_corrected"] = dataset["ext_vel_x_corrected"].abs()
mask_vel_neg = dataset["ins_vel_hor_x"] < 0
dataset.loc[mask_vel_neg,"ext_vel_x_corrected"] = dataset.loc[mask_vel_neg,"ext_vel_x_corrected"]*-1
dataset.loc[mask_insvel, "Correvit_slip_angle"] = np.arctan(dataset["ext_vel_y_corrected"]/dataset["ext_vel_x_corrected"])*(180/np.pi)
"""The slip angle from correvit direction and magnitude might be wrong due to lever arm position, need to mount and take coordinates"""
correvit_arm = np.array([1.273, 0.099+.15, -0.169]) #Roughly 35,15 and 55cm from adma to correvit
dataset["Correvit_COG_x"] =  dataset["ext_vel_x_corrected"] + (-1*dataset['rate_hor_z']*correvit_arm[1]*angular_conversion + dataset['rate_hor_y']*correvit_arm[2])*angular_conversion
dataset["Correvit_COG_y"] =  dataset["ext_vel_y_corrected"] + (dataset['rate_hor_z']*correvit_arm[0]*angular_conversion -dataset['rate_hor_x']*angular_conversion*correvit_arm[2])
dataset["Correvit_cog_velocity"] = np.sqrt(dataset["Correvit_COG_x"]**2 + dataset["Correvit_COG_y"]**2)*3.6
dataset["speedo_obd_kmph"] = dataset["speedo_obd"]
dataset["Diff_vel"] = (dataset["INS_totalvelCOG_hor_kmph"] - dataset["Correvit_cog_velocity"]).abs() #ataset["speedo_obd"][speedo_index] - dataset["ins_vel_kmph"]
difference_extreme = dataset["Diff_vel"].agg(['min', 'max'])
plt.figure(figsize=(8, 5))  # Set the figure size
plt.hist(dataset["Diff_vel"], bins=20, color='skyblue', edgecolor='black')  # Adjust bins and colors as needed
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of INS vs Correvit speed Diff')
plt.show()


dataset.loc[mask_insvel, "Correvit_slip_angle_COG"] = np.arctan(dataset["Correvit_COG_y"]/dataset["Correvit_COG_x"])*(180/np.pi)
dataset.loc[mask_insvel, "Diff_slipangle_INS_correvit"] = (dataset["Correvit_slip_angle_COG"] - dataset["INS_slip_angle_COG"]).abs()
correvit_mask = (dataset["Correvit_slip_angle_COG"] >= -0.5) & (dataset["Correvit_slip_angle_COG"] <= 0.5) & pd.notna(dataset["Correvit_slip_angle_COG"])
filtered_dataset = dataset[correvit_mask]
difference_extreme = filtered_dataset["Diff_slipangle_INS_correvit"].agg(['min', 'max'])
bin_edges = np.arange(0.5, int(difference_extreme['max'])+1, 0.5)
hist, _ = np.histogram(filtered_dataset["Diff_slipangle_INS_correvit"], bins=bin_edges, density=True)

# Plot the histogram
plt.figure(figsize=(8, 5))
plt.hist(filtered_dataset["Diff_slipangle_INS_correvit"], bins=bin_edges, color='skyblue', edgecolor='black', density=True)
plt.xlabel('Absolute Difference in Slip Angle (degrees)')
plt.ylabel('Density')
plt.title('Histogram of Absolute Difference between Correvit and INS Slip Angles')

# Calculate and plot the percentage density
total_density = sum(hist * np.diff(bin_edges))
percentage_density = hist / total_density * 100
plt.xticks(np.arange(0, int(difference_extreme['max'])+1, 0.5))
plt.grid(axis='y', alpha=0.75)
plt.show()
# difference_extreme = filterd_dataset["Diff_slipangle_INS_correvit"].agg(['min', 'max'])
# plt.hist(filtered_dataset["Diff_slipangle_INS_correvit"], bins=range(1, int(difference_extreme['max'])+2), color='skyblue', edgecolor='black')
# plt.xlabel('Absolute Difference in Slip Angle (degrees)')
# plt.ylabel('Frequency')
# plt.title('Histogram of Absolute Difference between Correvit and INS Slip Angles')
# plt.xticks(range(1, int(difference_extreme['max'])+2))  # Set x-ticks to desired intervals
# plt.grid(axis='y', alpha=0.75)
# plt.show()

#Slip angle calculation from velocity of tires and velocity of Cog(speedometer value)
#The max and min slip angles from INS_slip_angle_cog is -17 to 13 degrees. So small angle approximations hold.
#Refer to page 311 at Automotive control systems for engine, driveline and Vehicle and datasheet for values of ISSAK car
#Problem is lower yaw rate causes errors in calculations
mask_tirevel = (dataset["VelFL_obd"].notna()) & (dataset["yaw_rate"].abs() > 0.2)
dataset.loc[mask_tirevel,"veh_slip_fl"] = ((((dataset.loc[mask_tirevel,"VelFL_obd"] - dataset.loc[mask_tirevel,"speedo_obd"])/(1*dataset.loc[mask_tirevel,"yaw_rate"]*3.6)) + (1.283/2)) / 1.0732)
dataset.loc[mask_tirevel,"veh_slip_fr"] = ((((dataset.loc[mask_tirevel,"VelFR_obd"] - dataset.loc[mask_tirevel,"speedo_obd"])/(1*dataset.loc[mask_tirevel,"yaw_rate"]*3.6)) - (1.283/2)) / 1.0732)
dataset.loc[mask_tirevel,"veh_slip_rl"] = ((((dataset.loc[mask_tirevel,"VelRL_obd"] - dataset.loc[mask_tirevel,"speedo_obd"])/(1*dataset.loc[mask_tirevel,"yaw_rate"]*3.6)) + (1.385/2)) / (-1*.7998))
dataset.loc[mask_tirevel,"veh_slip_rr"] = ((((dataset.loc[mask_tirevel,"VelRR_obd"] - dataset.loc[mask_tirevel,"speedo_obd"])/(1*dataset.loc[mask_tirevel,"yaw_rate"]*3.6)) - (1.385/2)) / (-1*.7998))
print("hi")


#Plots for low velocity slip angles.
mask_slipangles = ((dataset["Vehicle_slip_obd"].notna()) & (dataset["INS_slip_angle_COG"].notna()))
ground_truth = dataset.loc[mask_slipangles,"side_slip_angle"].values
#ground_truth = dataset.loc[mask_slipangles,"INS_slip_angle_COG"].values
predictions = dataset.loc[mask_slipangles,"Vehicle_slip_obd"].values
residuals = predictions - ground_truth
mae = mean_absolute_error(ground_truth, predictions)
mse = mean_squared_error(ground_truth, predictions)
rmse = mean_squared_error(ground_truth, predictions, squared=False)
r2 = r2_score(ground_truth, predictions)

#Plot ground truth vs. predictions
plt.figure(figsize=(8, 6))
plt.scatter(ground_truth, predictions, color='blue', label='Predictions')
plt.plot(ground_truth, ground_truth, color='red', linestyle='--', label='Ground Truth')
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.title('Ground Truth vs. Predictions')
plt.legend()
plt.grid(True)
plt.show()

sns.kdeplot(residuals, color='blue', fill=True, bw_adjust=0.5)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Density Plot of Residuals')
plt.grid(True)
plt.show()
print('hi')

