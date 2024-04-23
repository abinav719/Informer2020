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

"""The goal of this code is to analyse the data lag between OBD data channels, ADMA and car PC
1) ADMA generated time stamp is the best one which is UTC and coming with synchronization from the satellites
2) OBD data received is time stamp at which OBD data is received

The challenges to analyse and document is that
1) Time sync between carpc and unix time
2) Data latency of OBD and ADMA"""

folder_name = 'Data_set_unzipped'
file_names = os.listdir(folder_name)
frames = []
# for i in file_names:
#     frames.append(pd.read_csv(f"{folder_name}/{i}/{i}ADMA+OBD_synced.csv"))
# dataset = pd.concat(frames)
#i = '2023-03-10-06-59-30'
i = 'Time_corrected_2024-04-18-17-29-57'
dataset = pd.read_csv(f"{folder_name}/{i}/{i}ADMA+OBD_synced.csv")
dataset = dataset[(dataset["status_gnss_mode"]==8) & (dataset["status_speed"]==2)] #Filtering settled values after Kalman settled
dataset.reset_index(drop=True, inplace=True)
dataset["yaw_rate"] = dataset["yaw_rate"]*-1
dataset["Yawrate_rostimesync"] = dataset["Yawrate_rostimesync"]*-1


#Plot script to analyse the data lag. Just analyse and input the start and end points
start = 1713454360149486340  #1678428170598774403
end   = 1713454400149486340   #1678428185598774403
columns_extract = ["rosbagTimestamp_ADMA","INS_time_sec","rate_hor_z", "yaw_rate","yaw_OBDtime_sec"]
plot_dataset = dataset[(dataset['rosbagTimestamp_ADMA'] >= start) & (dataset['rosbagTimestamp_ADMA'] <= end)][columns_extract]
plot_dataset = plot_dataset.dropna(subset=["yaw_rate"])
plt.figure(figsize=(10, 5))
df = plot_dataset
cross_corr = np.correlate(plot_dataset['rate_hor_z'], plot_dataset['yaw_rate'], mode='full')
lag = np.argmax(cross_corr) - (len(plot_dataset['rate_hor_z']) - 1)
plt.plot((df["rosbagTimestamp_ADMA"]-start)/1e9, df["yaw_rate"], label='OBD')
plt.plot((df["rosbagTimestamp_ADMA"]-start)/1e9, df['rate_hor_z'], label="ADMA")
plt.text(0.5, 0.95, f'Data Lag: {lag} samples', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
plt.text(0.5, 0.90, f'Total samples: {len(plot_dataset)}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
plt.xlabel('Time (sec)')
plt.ylabel('Yaw (deg/sec)')
plt.title('Lag between ADMA(INS) vs OBD(ROS)')
plt.legend()
plt.grid(True)
plt.show()
print('Hi')
plt.close()

#Plot script to analyse the external correvit velocity
start = 1713454320149486340  #1678428170598774403
end   = 1713454400149486340  #1678428185598774403
columns_extract = ["rosbagTimestamp_ADMA","INS_time_sec","ext_vel_x_corrected", "ins_vel_hor_x"]
plot_dataset = dataset[(dataset['rosbagTimestamp_ADMA'] >= start) & (dataset['rosbagTimestamp_ADMA'] <= end)][columns_extract]
plt.figure(figsize=(10, 5))
df = plot_dataset
cross_corr = np.correlate(plot_dataset["ext_vel_x_corrected"], plot_dataset["ins_vel_hor_x"], mode='full')
lag = np.argmax(cross_corr) - (len(plot_dataset['ins_vel_hor_x']) - 1)
plt.plot((df["rosbagTimestamp_ADMA"]-start)/1e9, df["ext_vel_x_corrected"], label='Correvit')
plt.plot((df["rosbagTimestamp_ADMA"]-start)/1e9, df['ins_vel_hor_x'], label="ADMA")
plt.text(0.5, 0.95, f'Data Lag: {lag} samples', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
plt.text(0.5, 0.90, f'Total samples: {len(plot_dataset)}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
plt.xlabel('Time (s)')
plt.ylabel('Vel_x (m/s)')
plt.title('Velocity_X Comparision  ')
plt.legend()
plt.grid(True)
plt.show()
print('Hi')
plt.close()

#Plot script to analyse time lag
dataset["rosbagTimestamp_ADMA"] = (dataset["rosbagTimestamp_ADMA"] - dataset["rosbagTimestamp_ADMA"][0])/(1e9)
dataset["INS_time_sec"] = (dataset["INS_time_sec"] - dataset["INS_time_sec"][0])
dataset["time_lag"] = (dataset["rosbagTimestamp_ADMA"] - dataset["INS_time_sec"])*1e3
plt.figure(figsize=(10, 6))
plt.plot(dataset.index, dataset["time_lag"], color='blue', marker='o', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Time Lag (milli seconds)')
plt.title('Time Lag between rosbagTimestamp_ADMA and INS_time_sec')
plt.grid(True)
plt.show()
print("hi")


#Code for rostimestamp based sync in data (ADMA rosbagtimestamp vs OBD rosbagtimestamp
columns_extract = ["rosbagTimestamp_ADMA","rosbagTimestamp_yaw","rate_hor_z", "yaw_rate","Yawrate_rostimesync"]
plot_dataset = dataset[(dataset['rosbagTimestamp_ADMA'] >= start) & (dataset['rosbagTimestamp_ADMA'] <= end)][columns_extract]
plot_dataset = plot_dataset.dropna(subset=["Yawrate_rostimesync"])
plt.figure(figsize=(10, 5))
df = plot_dataset
cross_corr = np.correlate(plot_dataset['rate_hor_z'], plot_dataset['Yawrate_rostimesync'], mode='full')
lag = np.argmax(cross_corr) - (len(plot_dataset['rate_hor_z']) - 1)
plt.plot((df["rosbagTimestamp_ADMA"]-start)/1e9, df["Yawrate_rostimesync"], label='OBD')
plt.plot((df["rosbagTimestamp_ADMA"]-start)/1e9, df['rate_hor_z'], label="ADMA")
plt.text(0.5, 0.95, f'Data Lag: {lag} samples', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
plt.text(0.5, 0.90, f'Total samples: {len(plot_dataset)}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
plt.xlabel('Time (Sec)')
plt.ylabel('Yaw (Deg/Sec)')
plt.title('Data Lag between ADMA vs OBD in rostime')
plt.legend()
plt.grid(True)
plt.show()
print('Hi')
plt.close()



#Plot code for meeting
start = 1678431689370731870
end   = 1678431705370731870
columns_extract = ["rosbagTimestamp_ADMA","INS_time_sec","rate_hor_z", "yaw_rate","yaw_OBDtime_sec"]
plot_dataset = dataset[(dataset['rosbagTimestamp_ADMA'] >= start) & (dataset['rosbagTimestamp_ADMA'] <= end)][columns_extract]
plot_dataset = plot_dataset.dropna(subset=["yaw_rate"])
plt.figure(figsize=(10, 5))
plot_dataset['lagged_rate_hor_z'] = plot_dataset['rate_hor_z'].shift(-32)#-570731870
df = plot_dataset
cross_corr = np.correlate(plot_dataset['lagged_rate_hor_z'], plot_dataset['yaw_rate'], mode='full')
lag = np.argmax(cross_corr) - (len(plot_dataset['lagged_rate_hor_z']) - 1)
plt.plot(df.index, df["yaw_rate"], label='OBD')
plt.plot(df.index, df['lagged_rate_hor_z'], label="ADMA")
plt.text(0.5, 0.95, f'Data Lag: {lag} samples', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
plt.xlabel('Time')
plt.ylabel('Yaw')
plt.title('Data Lag between ADMA vs OBD')
plt.legend()
plt.grid(True)
plt.show()
print('Hi')