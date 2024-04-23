import numpy as np
import pandas as pd
import argparse
import sys
import os

"""The goal of this code is to generate ground truth vehicle slip and velocity from correvit 
sensor at the COG of the vehicle. The velocity measured by the correvit is at the mounting position
It needs to be transformed to the COG of the car to get the velocity and slip at the COG.
The correvit measured quantities will also be compared with ADMA INS values at COG
Since Correvit directly measures velocity while the ADMA estimates the velocity, the vehicle slip angle from correvit is better 
and should be preferred"""

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

#Converting the velovity of INS ADMA to the COG of the vehicle for Issak car using coordinate frame transfomrations refer Prof Michael slides and books.
#Basically v_cog = V_adma + omega X pivot (adma--cog)
arm = np.array([0.923, 0.099, 0.381]) #adma pivot arm - SAE coordinate system as measurement was taken in it
#The coordinate system can be found by direction of the acceleration in the z axis.
#Rotation rates are in degrees, So need to convert to radians/Sec
angular_conversion = (np.pi/180)
#Direct conversion formula
dataset["INS_velCOG_hor_x"] =  dataset["ins_vel_hor_x"] + (-1*dataset['rate_hor_z']*angular_conversion*arm[1] + dataset['rate_hor_y']*angular_conversion*arm[2])
dataset["INS_velCOG_hor_y"] =  dataset["ins_vel_hor_y"] + (dataset['rate_hor_z']*angular_conversion*arm[0]-dataset['rate_hor_x']*angular_conversion*arm[2])
dataset["INS_velCOG_hor_z"] =  dataset["ins_vel_hor_z"] + (-1*dataset['rate_hor_y']*angular_conversion*arm[0]+ dataset['rate_hor_x']*angular_conversion*arm[1])
dataset["INS_totalvelCOG_hor"] = np.sqrt(dataset["INS_velCOG_hor_x"]**2 + dataset["INS_velCOG_hor_y"]**2) #Purposefully left z as need not calculate for car
mask_insvel =  dataset["INS_totalvelCOG_hor"] > 1
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
dataset["INS_totalvelCOG_hor_kmph"] = dataset["INS_totalvelCOG_hor"]*3.6
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
dataset.loc[mask_insvel, "Diff_slipangle"] = (dataset["Correvit_slip_angle_COG"] - dataset["INS_slip_angle_COG"]).abs()
difference_extreme = dataset["Diff_vel"].agg(['min', 'max'])
plt.figure(figsize=(8, 5))  # Set the figure size
plt.hist(dataset["Diff_vel"], bins=20, color='skyblue', edgecolor='black')  # Adjust bins and colors as needed
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of INS slip vs Correvit slip Diff')
plt.show()