import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import sys
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#Tasks first figure out speedo is in m/sec or km/hr - Verdict its in Km/Hr

folder_name = 'Data_set_unzipped'
#dataset = pd.read_csv("Data_set_unzipped/1bagfromeduardo/1bagfromeduardoADMA+OBD_synced.csv")
file_names = os.listdir(folder_name)
frames = []
for i in file_names:
    frames.append(pd.read_csv(f"{folder_name}/{i}/{i}ADMA+OBD_synced.csv"))
dataset = pd.concat(frames)
#i = '2023-03-10-06-59-30'
#dataset = pd.read_csv(f"{folder_name}/{i}/{i}ADMA+OBD_synced.csv")
dataset = dataset[(dataset["status_gnss_mode"]==8) & (dataset["status_speed"]==2)] #Filtering settled values
dataset.reset_index(drop=True, inplace=True)

#Code to experiment for a mapper value between INS velocity and Speedometer
vel_z_extreme = dataset["ins_vel_hor_z"].agg(['min', 'max']) #During turn and acceleration there is velocity across z axis as well due to roll pitch
speedo_extreme = dataset["speedo_obd"].agg(['min', 'max'])
dataset["ins_vel"] = np.sqrt(dataset["ins_vel_hor_z"]**2 +  dataset["ins_vel_hor_x"]**2 + dataset["ins_vel_hor_y"]**2)
#dataset["ins_vel"] = np.sqrt(dataset["ins_vel_hor_x"]**2 + dataset["ins_vel_hor_y"]**2) #Need to know which is correct
ins_vel_extreme = dataset["ins_vel"].agg(['min', 'max'])
print(vel_z_extreme,speedo_extreme,ins_vel_extreme)

#Speedo error correction - Sometimes standstill car has speedo values above 4094 Kmphr and velocity of tires also has an error
#This error is corrected back to 0 velocity
condition = (dataset['speedo_obd'] >= 4094) & (dataset['status_standstill'] == 1)
columns_to_update = ['VelFR_obd', 'VelFL_obd', 'VelRR_obd', 'VelRL_obd', 'speedo_obd']
dataset.loc[condition, columns_to_update] = 0

#Plots to visualize the velocity from speedo and INS in Kmphr
#Speedo frequency from OBD is only half of ADMA. Considering only timestamps with both values
speedo_index = dataset["speedo_obd"].notna()
plt.figure(figsize=(12, 5))  # Set the figure size
plt.subplot(1, 2, 1)
plt.hist(dataset["speedo_obd"], bins=20, color='skyblue', edgecolor='black')  # Adjust bins and colors as needed
plt.xlabel('Speed_OBD_Kmph')
plt.ylabel('Frequency')
plt.title('Histogram of Speedo Values')

plt.subplot(1, 2, 2)
dataset["ins_vel_kmph"] =  dataset["ins_vel"]*(3.6)
plt.hist(dataset["ins_vel_kmph"][speedo_index], bins=20, color='skyblue', edgecolor='black')  # Adjust bins and colors as needed
plt.xlabel('Speed_INSADMA_Kmph')
plt.ylabel('Frequency')
plt.title('Histogram of INS Velocity Values')
plt.tight_layout()
plt.show()

# #Roughly seeing the error between ins and speedo values
dataset["Diff_vel"] = (dataset["speedo_obd"][speedo_index] - dataset["ins_vel_kmph"]).abs() #ataset["speedo_obd"][speedo_index] - dataset["ins_vel_kmph"]
difference_extreme = dataset["Diff_vel"][speedo_index].agg(['min', 'max'])
plt.figure(figsize=(8, 5))  # Set the figure size
plt.hist(dataset["Diff_vel"][speedo_index], bins=20, color='skyblue', edgecolor='black')  # Adjust bins and colors as needed
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of INS vs Speedo Diff Values')
plt.show()


#Defining an error metric to compute velocity
Total_velocity_error = sum((dataset["Diff_vel"][speedo_index]).abs())
dataset['Speedo_bins'] = pd.cut(dataset['speedo_obd'][speedo_index], bins = [i for i in range(0,101,5)], include_lowest=True, right= False) #In order to include 0 it was formulated as -5,101,5
#The correlation_coefficient denotes a strong linear relationship between both variables
correlation_coefficients = dataset.groupby('Speedo_bins')['speedo_obd'].corr(dataset["ins_vel_kmph"])

#Visualizing the average error between INS and speedo in each bin is more important and matches with eduardo paper.
#Linear Regression from Speedo to INS velocity based on speedo bins (Need one LR model for each speedo bin
models = [LinearRegression() for i in range(0,101,5)]
trees = [DecisionTreeRegressor(max_depth=1000) for i in range(0,101,5)]
dataset["predict_ins_vel(LR)"] = np.nan
for i in range(0,101,5):
    j = int((i/5))
    if len(dataset[dataset["Speedo_bins"]==pd.Interval(i, i+5, closed='left')]["speedo_obd"].values.reshape(-1, 1))>1:
        condition = (dataset["Speedo_bins"]==pd.Interval(i, i+5, closed='left'))
        models[j].fit(dataset.loc[condition]["speedo_obd"].values.reshape(-1, 1),dataset.loc[condition]["ins_vel_kmph"])
        trees[j].fit(dataset.loc[condition]["speedo_obd"].values.reshape(-1, 1),dataset.loc[condition]["ins_vel_kmph"])
        dataset.loc[condition, "predict_ins_vel(LR)"] = models[j].predict(dataset[condition]["speedo_obd"].values.reshape(-1, 1))
        dataset.loc[condition, "predict_ins_vel(TreesNLR)"] = models[j].predict(dataset[condition]["speedo_obd"].values.reshape(-1, 1))
    else:
        continue

#Plot script for error(Diff_vel) in each speedo bin.
dataset["Diff_vel_LM"] = (dataset["predict_ins_vel(LR)"][speedo_index] - dataset["ins_vel_kmph"]).abs()
dataset["Diff_vel_NLM"] = (dataset["predict_ins_vel(TreesNLR)"][speedo_index] - dataset["ins_vel_kmph"]).abs() #Decision trees

avg_velerror_groups = dataset.groupby('Speedo_bins')["Diff_vel"].mean()
avg_velerror_predicted_groups = (dataset.groupby('Speedo_bins')["Diff_vel_LM"]).mean()
avg_velerror_predicted_groups_trees = (dataset.groupby('Speedo_bins')["Diff_vel_NLM"]).mean() #Non Linear Models are performing at same rate as linear models.

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
bar_positions = range(len(avg_velerror_groups))
ax.bar([pos - bar_width/2 for pos in bar_positions], avg_velerror_groups, bar_width, label='Total Vel Error')
ax.bar([pos + bar_width/2 for pos in bar_positions], avg_velerror_predicted_groups, bar_width, label='Total Vel Error (Predicted)', color='orange')
ax.set_xticks(bar_positions)
ax.set_xticklabels([i for i in range(5,106,5)], fontsize=7)
plt.xlabel('Speedo Bins')
plt.ylabel('Average Error')
plt.legend()
plt.show()

#Code to compute vehicle velocity from tires and use the data to compute graphs
dataset['speed_tires'] = (dataset["VelFR_obd"]+dataset["VelFL_obd"]+dataset["VelRR_obd"]+dataset["VelRL_obd"])/4
tirevel_obd_rostime_index = dataset["speed_tires"].notna()
dataset['Diff_tires_adma'] = (dataset["speed_tires"][tirevel_obd_rostime_index] - dataset["ins_vel_kmph"]).abs()
dataset['speed_tires_bins'] = pd.cut(dataset['speed_tires'][tirevel_obd_rostime_index], bins = [i for i in range(0,101,5)], include_lowest=True, right= False)
avg_velerror_groups_tire_Speedo_bins_AG_OBDc = dataset.groupby('speed_tires_bins')["Diff_tires_adma"].mean()
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
bar_positions = range(len(avg_velerror_groups_tire_Speedo_bins_AG_OBDc ))
ax.bar([pos - bar_width/2 for pos in bar_positions], avg_velerror_groups, bar_width, label='Vel Error speedo')
ax.bar([pos + bar_width/2 for pos in bar_positions], avg_velerror_groups_tire_Speedo_bins_AG_OBDc, bar_width, label='Vel error tires', color='orange')
ax.set_xticks(bar_positions)
ax.set_xticklabels([i for i in range(5,106,5)], fontsize=7)
plt.xlabel('Speed Bins')
plt.ylabel('Average Error')
plt.legend()
plt.show()

#Steering, acceleration_y , pedal acceleration , brake , yaw rate, speedo , velocity of tires.
#Steering data
steering_extreme = dataset["SW_pos_obd"].agg(['min', 'max'])
steering_obd_rostime_index = dataset['SW_pos_obd'].notna()
dataset['Steering_bins'] = pd.cut(dataset['SW_pos_obd'][steering_obd_rostime_index], bins = [i for i in range(-640,640,20)], include_lowest=True, right= False)
relative_freq = dataset['Steering_bins'].value_counts(normalize=True)
relative_freq.sort_index().plot(kind='bar', width=1.0, edgecolor='black')
plt.title('Relative Frequency Histogram of Steering Bins')
plt.xlabel('Steering Bins')
plt.ylabel('Relative Frequency')
plt.show()

# acceleration_y
latacceleration_extreme = dataset["LatAcc_obd"].agg(['min', 'max'])
latacc_obd_rostime_index = dataset["LatAcc_obd"].notna()
dataset['Latacc_bins'] = pd.cut(dataset["LatAcc_obd"][latacc_obd_rostime_index], bins = [i for i in range(-12,12,1)], include_lowest=True, right= False)
relative_freq = dataset['Latacc_bins'].value_counts(normalize=True)
relative_freq.sort_index().plot(kind='bar', width=1.0, edgecolor='black')
plt.title('Relative Frequency Histogram of Lateral acceleration')
plt.xlabel('Lateral Acceleration')
plt.ylabel('Relative Frequency')
plt.show()

#Acceleration_pedal
accelerationpedal_extreme = dataset["acc_input%_obd"].agg(['min', 'max'])
accpedal_obd_rostime_index = dataset["acc_input%_obd"].notna()
dataset['accpedal_bins'] = pd.cut(dataset["acc_input%_obd"][accpedal_obd_rostime_index ], bins = [i for i in range(0,100,5)], include_lowest=True, right= False)
relative_freq = dataset['accpedal_bins'].value_counts(normalize=True)
relative_freq.sort_index().plot(kind='bar', width=1.0, edgecolor='black')
plt.title('Relative Frequency Acceleration Pedal')
plt.xlabel('Acceleration pedal %')
plt.ylabel('Relative Frequency')
plt.show()

#Brake Pressure
brakepedal_extreme = dataset["brake_pressure_obd"].agg(['min', 'max'])
brakepressure_obd_rostime_index = dataset["brake_pressure_obd"].notna()
dataset['brakepressure_bins'] = pd.cut(dataset["brake_pressure_obd"][brakepressure_obd_rostime_index], bins = [i for i in range(0,200,10)], include_lowest=True, right= False)
relative_freq = dataset['brakepressure_bins'].value_counts(normalize=True)
relative_freq.sort_index().plot(kind='bar', width=1.0, edgecolor='black')
plt.title('Relative Frequency Brake Pressure')
plt.xlabel('Brake Pressure')
plt.ylabel('Relative Frequency')
plt.show()



# #Speedo_OBD_created vs ADMA generated time plot requiestes are derived here
# speedo_data_rostime = dataset[["speedo_obd", "rostimestamp_speedo"]].copy(deep=True)
# speedo_data_rostime["rostimestamp_speedo"] = (speedo_data_rostime["rostimestamp_speedo"] / (10**9))
# speedo_data_rostime.dropna(subset=['rostimestamp_speedo'], inplace=True)
# tolerance_time = .01
# speedo_data_rostime=speedo_data_rostime.rename(columns={"speedo_obd": "speedo_obd_AG_OBDc", 'rostimestamp_speedo':'rostimestamp_speedo_v2'})#v1 is alrady in dataset
# speedo_data_rostime["sync_timer_sec"] = speedo_data_rostime['rostimestamp_speedo_v2']
#
# #pd.merge requires sorted columns for efficient implementation.
# dataset.sort_values('rosbagTimestamp_ADMA', inplace=True)
# speedo_data_rostime.sort_values("sync_timer_sec", inplace=True)
# dataset = pd.merge_asof(dataset,speedo_data_rostime, on="sync_timer_sec", direction='nearest', allow_exact_matches=True, tolerance=tolerance_time)
# duplicates_mask = dataset.duplicated(subset='rostimestamp_speedo_v2', keep='first')
# # duplicates from df2 with NaN in the merged DataFrame
# dataset.loc[duplicates_mask, 'rostimestamp_speedo_v2'] = np.nan
# dataset.loc[duplicates_mask, "speedo_obd_AG_OBDc"] = np.nan
# #condition = (dataset["speedo_obd_AG_OBDc"] >= 4094) & (dataset['status_standstill'] == 1)
# #columns_to_update = ['speedo_obd_rostime']
# #dataset.loc[condition, columns_to_update] = 0
# speedo_obd_rostime_index = dataset["speedo_obd_AG_OBDc"].notna()
# dataset["Diff_vel_AG_OBDc"] = (dataset["speedo_obd_AG_OBDc"][speedo_obd_rostime_index] - dataset["ins_vel_kmph"]).abs()
# dataset['Speedo_bins_AG_OBDc'] = pd.cut(dataset['speedo_obd_AG_OBDc'][speedo_obd_rostime_index], bins = [i for i in range(0,101,5)], include_lowest=True, right= False)
# avg_velerror_groups_Speedo_bins_AG_OBDc = dataset.groupby('Speedo_bins_AG_OBDc')["Diff_vel_AG_OBDc"].mean()
#
#
#
# #Speedo_OBD_created vs ADMA scaled time plot requiestes are derived here
# speedo_data_rostime = dataset[["speedo_obd", "rostimestamp_speedo"]].copy(deep=True)
# speedo_data_rostime.dropna(subset=['rostimestamp_speedo'], inplace=True)
# tolerance_time = 10**7
# speedo_data_rostime=speedo_data_rostime.rename(columns={"speedo_obd": "speedo_obd_AS_OBDc", 'rostimestamp_speedo':'rostimestamp_speedo_v3'})#v1 is alrady in dataset
# speedo_data_rostime["rosbagTimestamp_ADMA"] = speedo_data_rostime["rostimestamp_speedo_v3"].astype(int)
# #pd.merge requires sorted columns for efficient implementation.
# dataset.sort_values('rosbagTimestamp_ADMA', inplace=True)
# speedo_data_rostime.sort_values("rosbagTimestamp_ADMA", inplace=True)
# dataset = pd.merge_asof(dataset,speedo_data_rostime, on="rosbagTimestamp_ADMA", direction='nearest', allow_exact_matches=True, tolerance=tolerance_time)
# duplicates_mask = dataset.duplicated(subset='rostimestamp_speedo_v3', keep='first')
# # duplicates from df2 with NaN in the merged DataFrame
# dataset.loc[duplicates_mask, 'rostimestamp_speedo_v3'] = np.nan
# dataset.loc[duplicates_mask, "speedo_obd_AS_OBDc"] = np.nan
# #condition = (dataset["speedo_obd_AG_OBDc"] >= 4094) & (dataset['status_standstill'] == 1)
# #columns_to_update = ['speedo_obd_rostime']
# #dataset.loc[condition, columns_to_update] = 0
# speedo_obd_rostime_index = dataset["speedo_obd_AS_OBDc"].notna()
# dataset["Diff_vel_AS_OBDc"] = (dataset["speedo_obd_AS_OBDc"][speedo_obd_rostime_index] - dataset["ins_vel_kmph"]).abs()
# dataset['Speedo_bins_AS_OBDc'] = pd.cut(dataset['speedo_obd_AS_OBDc'][speedo_obd_rostime_index], bins = [i for i in range(0,101,5)], include_lowest=True, right= False)
# avg_velerror_groups_Speedo_bins_AS_OBDc = dataset.groupby('Speedo_bins_AS_OBDc')["Diff_vel_AS_OBDc"].mean()
#
#
#
#
#
#
# #Speedo OBD_received with ADMA robag time stamp. The purspose is to check the synchronization issues with Alberto paper.
# speedo_data_rostime = dataset[["speedo_obd", "speedo_OBDtime_sec"]].copy(deep=True)
# speedo_data_rostime["speedo_OBDtime_sec"] = (speedo_data_rostime["speedo_OBDtime_sec"] * (10**9))
# speedo_data_rostime.dropna(subset=['speedo_OBDtime_sec'], inplace=True)
# tolerance_time = 10**7
# speedo_data_rostime=speedo_data_rostime.rename(columns={"speedo_obd": "speedo_obd_rostime", 'speedo_OBDtime_sec':"speedo_OBDtime_sec_rostime"})
# speedo_data_rostime["rosbagTimestamp_ADMA"] = speedo_data_rostime["speedo_OBDtime_sec_rostime"]
# dataset["rosbagTimestamp_ADMA"] = dataset["rosbagTimestamp_ADMA"].astype(float)
# #pd.merge requires sorted columns for efficient implementation. Need to
# dataset.sort_values('rosbagTimestamp_ADMA', inplace=True)
# speedo_data_rostime.sort_values('rosbagTimestamp_ADMA', inplace=True)
# dataset = pd.merge_asof(dataset,speedo_data_rostime, on='rosbagTimestamp_ADMA', direction='nearest', allow_exact_matches=True, tolerance=tolerance_time)
# duplicates_mask =dataset.duplicated(subset="speedo_OBDtime_sec_rostime", keep='first')
# # duplicates from df2 with NaN in the merged DataFrame
# dataset.loc[duplicates_mask, 'speedo_OBDtime_sec_rostime'] = np.nan
# dataset.loc[duplicates_mask, 'speedo_obd_rostime'] = np.nan
# condition = (dataset['speedo_OBDtime_sec_rostime'] >= 4094) & (dataset['status_standstill'] == 1)
# columns_to_update = ['speedo_obd_rostime']
# dataset.loc[condition, columns_to_update] = 0
# speedo_obd_rostime_index = dataset["speedo_obd_rostime"].notna()
# dataset["Diff_vel_rostime"] = (dataset["speedo_obd_rostime"][speedo_obd_rostime_index] - dataset["ins_vel_kmph"]).abs() #ataset["speedo_obd"][speedo_index] - dataset["ins_vel_kmph"]
# dataset['Speedo_bins_rostime'] = pd.cut(dataset['speedo_obd_rostime'][speedo_obd_rostime_index], bins = [i for i in range(0,101,5)], include_lowest=True, right= False)
# avg_velerror_groups_rostime = dataset.groupby('Speedo_bins_rostime')["Diff_vel_rostime"].mean()
# fig, ax = plt.subplots(figsize=(10, 6))
# bar_width = 0.2
# bar_positions = range(len(avg_velerror_groups))
# plt.bar([pos - 1.5*bar_width for pos in bar_positions], avg_velerror_groups, bar_width, label='AG_OR')
# plt.bar([pos - 0.5*bar_width for pos in bar_positions], avg_velerror_groups_rostime, bar_width, label='AS_OR', color='orange')
# plt.bar([pos + 0.5*bar_width for pos in bar_positions], avg_velerror_groups_Speedo_bins_AS_OBDc, bar_width, label='AS_OC', color='green')
# plt.bar([pos + 1.5*bar_width for pos in bar_positions], avg_velerror_groups_Speedo_bins_AG_OBDc, bar_width, label='AG_OC', color='red')
# ax.set_xticks(bar_positions)
# ax.set_xticklabels([i for i in range(5,106,5)], fontsize=7)
# plt.xlabel('Speedo Bins')
# plt.ylabel('Average Error')
# plt.legend()
# plt.show()



