"""The goal of this code file is to learn the vehicle dynamics parameters
and steering wheel play from the configuration data.

We need to implement the single track based prediction without for loop."""
import numpy as np
import pandas as pd
import sys
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.signal import correlate
from pandas.tseries.frequencies import to_offset


dataset = pd.read_csv(f"data/OBD_ADMA/SingleTrack_configureADMA+OBD_synced_new.csv")
print(dataset.shape)
#dataset = dataset.iloc[75000:-5000]
print(dataset.shape)
dataset = dataset[(dataset["status_gnss_mode"]==8) & (dataset["status_speed"]==2)] #Filtering settled values after Kalman settled
dataset.reset_index(drop=True, inplace=True)
#Speedo error correction - Sometimes standstill car has speedo values above 4094 Kmphr and velocity of tires also has an error
#This error is corrected back to 0 velocity.
condition = (dataset['speedo_obd'] >= 4094) & (dataset['status_standstill'] == 1)
columns_to_update = ['VelFR_obd', 'VelFL_obd', 'VelRR_obd', 'VelRL_obd', 'speedo_obd']
dataset.loc[condition, columns_to_update] = 0

#ADMA slip angle at COG
angular_conversion = (np.pi/180)
arm = np.array([0.923, 0.099, 0.381])
dataset["INS_velCOG_hor_x"] =  dataset["ins_vel_hor_x"] + (-1*dataset['rate_hor_z']*angular_conversion*arm[1] + dataset['rate_hor_y']*angular_conversion*arm[2])
dataset["INS_velCOG_hor_y"] =  dataset["ins_vel_hor_y"] + (dataset['rate_hor_z']*angular_conversion*arm[0]-dataset['rate_hor_x']*angular_conversion*arm[2])
dataset["INS_velCOG_hor_z"] =  dataset["ins_vel_hor_z"] + (-1*dataset['rate_hor_y']*angular_conversion*arm[0]+ dataset['rate_hor_x']*angular_conversion*arm[1])
dataset["INS_totalvelCOG_hor"] = np.sqrt(dataset["INS_velCOG_hor_x"]**2 + dataset["INS_velCOG_hor_y"]**2) #Purposefully left z as need not calculate for car
mask_insvel =  dataset["INS_totalvelCOG_hor"] > 1
dataset.loc[mask_insvel, "INS_slip_angle_COG"] = np.arctan(dataset["INS_velCOG_hor_y"]/dataset["INS_velCOG_hor_x"])*(180/np.pi)
dataset.loc[~mask_insvel, "INS_slip_angle_COG"] = 0


#Correvit slip angle
correvit_arm = np.array([49+92.3,-15.5,7])/100
dataset["Correvit_COG_x"] =  dataset["ext_vel_an_x"] + (-1*dataset['rate_hor_z']*correvit_arm[1]*angular_conversion + dataset['rate_hor_y']*correvit_arm[2])*angular_conversion
dataset["Correvit_COG_y"] =  dataset["ext_vel_an_y"] + (dataset['rate_hor_z']*correvit_arm[0]*angular_conversion -dataset['rate_hor_x']*angular_conversion*correvit_arm[2])
dataset["Correvit_cog_velocity"] = np.sqrt(dataset["Correvit_COG_x"]**2 + dataset["Correvit_COG_y"]**2)
mask_insvel =  dataset["Correvit_cog_velocity"] > 1
dataset.loc[mask_insvel, "Correvit_slip_angle_COG"] = np.arctan(dataset["Correvit_COG_y"] /dataset["Correvit_COG_x"])*(180/np.pi)
x=0.85 #Tilt correction 0.85
error=[]
#for i in np.linspace(0.5,2,150):
#x=i
dataset["Correvit_COG_x_tiltcorrected"] =  dataset["Correvit_COG_y"]*np.sin(np.deg2rad(x)) + dataset["Correvit_COG_x"]*np.cos(np.deg2rad(x))
dataset["Correvit_COG_y_tiltcorrected"] = dataset["Correvit_COG_y"]*np.cos(np.deg2rad(x)) - dataset["Correvit_COG_x"]*np.sin(np.deg2rad(x))
dataset.loc[mask_insvel, "Correvit_slip_angle_COG_corrvittiltcorrected"] = np.arctan(dataset["Correvit_COG_y_tiltcorrected"] /dataset["Correvit_COG_x_tiltcorrected"])*(180/np.pi)
#During low velocities of less than 1 m/sec the slip angle is filled as 0
dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].fillna(0, inplace=True)
#error.append(np.mean(np.abs(dataset["Correvit_slip_angle_COG_corrvittiltcorrected"]-dataset["INS_slip_angle_COG"])))
#plt.figure(figsize=(8, 5))
#plt.plot(np.linspace(0.5,2,150),error)
#plt.show()


##KMB-1 Check
dataset["SW_pos_obd"] = dataset["SW_pos_obd"] - 10
negative_play = 5
positive_play = 15  #For new datasets check whether 5 or 10 holds good for positive play.
mask_steering = ((dataset["SW_pos_obd"] <= 0) & (dataset["SW_pos_obd"] >= -negative_play))
dataset.loc[mask_steering,'SW_pos_obd'] = 0
mask_steering = ((dataset["SW_pos_obd"] <= positive_play) & (dataset["SW_pos_obd"] >= 0))
dataset.loc[mask_steering,'SW_pos_obd'] = 0
dataset.loc[dataset["SW_pos_obd"] > 0, "SW_pos_obd"] -= positive_play
dataset.loc[dataset["SW_pos_obd"] < 0, "SW_pos_obd"] += negative_play
dataset.loc[mask_insvel,'Vehicle_slip_obd'] = (np.arctan(np.tan((dataset.loc[mask_insvel,"SW_pos_obd"]*np.pi)/(22*180))*0.427))*(180/np.pi) #lh=799,8,l=1873 for IssaK
#dataset.loc[mask_insvel,'Vehicle_slip_obd'] = (0.427*((dataset.loc[mask_insvel,"SW_pos_obd"]*2*np.pi)/(22*360)))*(180/np.pi)
dataset.loc[~mask_insvel,'Vehicle_slip_obd'] = 0
optimizer = np.mean(np.abs(dataset["side_slip_angle_COG"] - dataset["Vehicle_slip_obd"]))

plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(dataset["INS_slip_angle_COG"], color='blue',label="INS_slip_COG")
plt.plot(dataset["Correvit_slip_angle_COG_corrvittiltcorrected"],color="red",label="corrvit_slip_COG") # Adjust bins and colors as needed
plt.plot(dataset["Vehicle_slip_obd"],label="KMB-1")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Vehicle slip angle in degress at COG')
plt.title('Vehicle slip angle analysis in degrees')
plt.show()

condition =  ((dataset['speedo_obd'] <= 20) & ( dataset["LatAcc_obd"].abs() <= 3))
abs_diff_kmb = (dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"]-dataset.loc[condition,"Vehicle_slip_obd"]).abs()
bin_edges = np.arange(dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].min(), dataset["Correvit_slip_angle_COG_corrvittiltcorrected"].max() + 0.25, 0.25)
bin_indices = np.digitize(dataset.loc[condition,"Correvit_slip_angle_COG_corrvittiltcorrected"], bins=bin_edges)


mean_abs_error_per_bin_kmb = []
bin_centers = []
bin_counts = []

for i in range(1, len(bin_edges)):
    bin_mask = (bin_indices == i)
    if np.any(bin_mask):
        mean_abs_error_kmb = np.mean(abs_diff_kmb[bin_mask])
        mean_abs_error_per_bin_kmb.append(mean_abs_error_kmb)
        bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)
        bin_counts.append(np.sum(bin_mask))  # Number of data points in each bin

# Plot the mean absolute error for each bin
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(bin_centers, mean_abs_error_per_bin_kmb, marker='o', linestyle='-',  label='KMB MAE')
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

#Vehicle dynamic parameters optimize
dataset["SW_pos_obd"] = dataset['SW_pos_obd'].ffill()
dataset["yaw_rate"] = dataset['yaw_rate'].ffill()
dataset["speedo_obd"] = dataset["speedo_obd"].ffill()
dataset["VelFR_obd"] = dataset["VelFR_obd"].ffill()
dataset["VelFL_obd"] = dataset["VelFL_obd"].ffill()
dataset["VelRR_obd"] = dataset["VelRR_obd"].ffill()
dataset["VelRL_obd"] = dataset["VelRL_obd"].ffill()
dataset["LatAcc_obd"] = dataset["LatAcc_obd"].ffill()


#Check whether you have the right obd decoder
plt.figure(figsize=(8, 5))  # Set the figure size
# Adjust bins and colors as needed
plt.plot(dataset["acc_hor_y"]*-9.81,color="blue",label="adma_lat_acc")
plt.plot(dataset["LatAcc_obd"],color="green",label="lat_acc")
plt.legend()
plt.xlabel('Time')
plt.ylabel('lat acceleration comparision')
plt.title('checker plot')
plt.show()
plt.close()

plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(dataset["yaw_rate"],color="green",label="yaw_rate") # Adjust bins and colors as needed
plt.plot(dataset["rate_hor_z"],color="blue",label="Adma_yaw_rate")
plt.legend()
plt.xlabel('Time')
plt.ylabel('yaw rate comparision')
plt.title('checker plot')
plt.show()
plt.close()

plt.figure(figsize=(8, 5))  # Set the figure size
 # Adjust bins and colors as needed
plt.plot(dataset["VelFR_obd"],color="red",label="FR_tire speed")
plt.plot(dataset["VelFL_obd"],color="orange",label="FL_tire speed")
plt.plot(dataset["VelRR_obd"],color="blue",label="RR_tire speed")
plt.plot(dataset["VelRL_obd"],color="black",label="RL_tire speed")
plt.plot(dataset["INS_totalvelCOG_hor"]*3.6,color="green",label="speedo speed")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Tire speeds')
plt.title('checker plot')
plt.show()
plt.close()



"""Trouble shoot formula and signs
"""
def bicycle_model(params, data):

    # Vehicle parameters
    m = 1050    # Mass of vehicle
    lf = 1.0732 # Distance from CoG to front axle
    lr = 0.7998 # Distance from CoG to rear axle
    T = 0.02    # The prediction is offset by 10 values
    i_s = 22    # Steering wheel ratio

    #Params to be optimized
    c_alpha_f, c_alpha_r, I_z = params

    vel = data["INS_totalvelCOG_hor"]#data["ins_vel_frame_COG_x"]
    correvit_slip = data["Correvit_slip_angle_COG_corrvittiltcorrected"] * np.pi/180
    psi_dot_meas = data["rate_hor_z"]* np.pi/180
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

    # Iterate over the data and calculate the model predictions
    # for i in range(len(velocity)-1):
    #     vel = velocity.iloc[i]
    #     if np.abs(vel) < 1:
    #         slip_pred[i + 1], psi_dot_pred[i + 1] = 0,0
    #     else:
    #         matrix_a = np.array([[1 - ((c_alpha_f + c_alpha_r) / (m * vel)) * T,
    #                           (((c_alpha_r * lr - c_alpha_f * lf) / (m * vel * vel)) - 1) * T],
    #                          [((c_alpha_r * lr - c_alpha_f * lf) / I_z) * T,
    #                           1 - (((c_alpha_f * lf * lf + c_alpha_r * lr * lr) / (I_z * vel)) * T)]])
    #         matrix_b = np.array([[(c_alpha_f / (m * vel)) * T, (c_alpha_r / (m * vel)) * T],
    #                          [(c_alpha_f * lf / I_z) * T, -(c_alpha_r * lr / I_z) * T]])
    #
    #         x_n_1 = np.array([[np.float(correvit_slip.iloc[i]) * np.pi / 180],
    #                       [np.float(psi_dot_meas.iloc[i]) * np.pi / 180]])
    #         u_n_1 = np.array([[(np.float(data["SW_pos_obd"].iloc[i]) * np.pi) / (i_s * 180)], [0]])
    #
    #
    #         slip_pred[i+1], psi_dot_pred[i+1]  = matrix_a @ x_n_1 +  matrix_b @ u_n_1

    return slip_pred*(180/np.pi), psi_dot_pred*(180/np.pi)


# Define the cost function (sum of squared errors)
def cost_function(params, data):
    slip_pred, psi_dot_pred = bicycle_model(params, data)
    slip_meas, psi_dot_meas = data["Correvit_slip_angle_COG_corrvittiltcorrected"], data["rate_hor_z"]
    error = np.sum((slip_meas - np.roll(slip_pred,-2)) ** 2 + (psi_dot_meas - np.roll(psi_dot_pred,-2)) ** 2)
    return error

# Example data (replace with actual measured data)
data = dataset

# Initial guess for the parameters [C_f, C_r, I_z]
initial_guess = [80000, 100000, 1200]

#Options
options = {
    'disp': True,  # Display convergence messages
    'gtol': 1e-4   # Set gradient tolerance for convergence
}

# Bounds for the parameters: [(min, max), (min, max), (min, max)]
bounds = [(40000, 100000), (40000, 100000), (100, 1500)]

# Optimize parameters
result = minimize(cost_function, initial_guess, args=(data), method='L-BFGS-B', bounds=bounds, tol=1e-5)

# Optimized parameters
C_f_opt, C_r_opt, I_z_opt = result.x
print("Optimized Parameters:", C_f_opt, C_r_opt, I_z_opt)

#Testing
slip_pred, psi_dot_pred = bicycle_model([40000,63321,1500], data)
plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(np.roll(slip_pred,-2), color='red',label="Pred_slip_VD")#
plt.plot(data["Correvit_slip_angle_COG_corrvittiltcorrected"],color="green",label="corrvit_slip_COG") # Adjust bins and colors as needed
plt.plot(data["INS_slip_angle_COG"],color="blue",label="ADMA_slip_COG")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Vehicle slip angle in degress at COG')
plt.title('After VD optimization in params')
plt.show()



