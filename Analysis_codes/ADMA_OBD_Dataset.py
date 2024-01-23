"""
1) This code is to Extract the interested Data from ADMA_scaled_csv and OBD_CSV files
2) After extraction based on OBD data receipt's rosbag timestamp and ADMA data generation timestamp data synchronization is carried out
3) Saving the data as a CSV file which could be later used for data analysis or ML model training
"""
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import sys
import os

# def create_arg_parser():
#     # Creates and returns the ArgumentParser object
#
#     parser = argparse.ArgumentParser(description='pass in dataset directory and rosbagfile nae')
#     parser.add_argument('DatasetDirectory',
#                     help='Path to the dataset unzipped directory.') #Dataset directory given as position argument
#     parser.add_argument('--rosbag',
#                     help='Name of rosbag.')
#     return parser
#
#
# "Data_set_unzipped" --rosbag "1bagfromeduardo"
# arg_parser = create_arg_parser()
# parsed_args = arg_parser.parse_args()
# dataset_location = parsed_args.DatasetDirectory
# rosbag_name = parsed_args.rosbag

dataset_location = 'Data_set_unzipped'
rosbag_names = os.listdir(dataset_location)

#For loop implemented to process multiple files at once.
for rosbag_name in rosbag_names:
    print(rosbag_name)
    adma_scaled = pd.read_csv(f"{dataset_location}/{rosbag_name}/_slash_adma_slash_data_scaled.csv")

    # Code to find the simulation duration from Rosbag Time stamps
    #We are converting the time from unix (Nano secs) format to date time format
    time_stamp_array = adma_scaled["rosbagTimestamp"]
    unix_epoch_timestamp_t0 = time_stamp_array[0]
    unix_epoch_timestamp_tf = time_stamp_array.iloc[-1]
    T_initial_datetime_obj = datetime.fromtimestamp(unix_epoch_timestamp_t0/1e9)
    T_final_datetime_obj = datetime.fromtimestamp(unix_epoch_timestamp_tf/1e9)
    simulation_time = T_final_datetime_obj - T_initial_datetime_obj
    print("Total simulation time=",simulation_time)

    # Extracting the Data's of Interest from ADMA
    # Quantities are in the frame and position of ADMA (Initial Measuring point of Interest)

    #The time given in ADMA header is wrong (1980's). In order to give correction this below two lines are formulated.
    #This is the data generated time denoted as INS_time_sec
    jan1_12am_1980 = 315532800 + 5*60*60*24
    adma_scaled["INS_time_sec"] = (adma_scaled["ins_time_msec"]* 0.001 + adma_scaled["ins_time_week"]* 7 * 24 * 60 * 60 + jan1_12am_1980)

    columns = adma_scaled.columns
    adma_extracted_doi = adma_scaled[["rosbagTimestamp","INS_time_sec","inv_path_radius", "side_slip_angle","ins_roll",
    "ins_pitch", "ins_yaw", "ins_pos_rel_x", "ins_pos_rel_y", "status_gnss_mode", "status_standstill", "status_skidding", "status_dead_reckoning", "status_tilt", "status_pos",
    "status_kalmanfilter_settled" , "status_speed"]]

    #Extracting acc_body_hr
    acc_body_hr = adma_scaled.columns.get_loc("acc_body_hr")
    acc_body_hr_df = adma_scaled.iloc[:,[acc_body_hr+1,acc_body_hr+2,acc_body_hr+3]]
    adma_extracted_doi = pd.concat([adma_extracted_doi , acc_body_hr_df.rename(columns={"x": "acc_body_hr_x", "y": "acc_body_hr_y", "z": "acc_body_hr_z"})],axis=1, join="inner")
    #Extracting rate_body_hr
    rate_body_hr = adma_scaled.columns.get_loc("rate_body_hr")
    rate_body_hr_df = adma_scaled.iloc[:,[rate_body_hr+1,rate_body_hr+2,rate_body_hr+3]]
    adma_extracted_doi = pd.concat([adma_extracted_doi , rate_body_hr_df.rename(columns={"x.1": "rate_body_hr_x", "y.1": "rate_body_hr_y", "z.1": "rate_body_hr_z"})],axis=1, join="inner")
    #Extracting acc_hor
    acc_hor = adma_scaled.columns.get_loc("acc_hor")
    acc_hor_df = adma_scaled.iloc[:,[acc_hor+1,acc_hor+2,acc_hor+3]]
    adma_extracted_doi = pd.concat([adma_extracted_doi , acc_hor_df.rename(columns={"x.5": "acc_hor_x", "y.5": "acc_hor_y", "z.5": "acc_hor_z"})],axis=1, join="inner")
    #Extracting rate_hor
    rate_hor = adma_scaled.columns.get_loc("rate_hor")
    rate_hor_df = adma_scaled.iloc[:,[rate_hor+1,rate_hor+2,rate_hor+3]]
    adma_extracted_doi = pd.concat([adma_extracted_doi , rate_hor_df.rename(columns={"x.3": "rate_hor_x", "y.3": "rate_hor_y", "z.3": "rate_hor_z"})],axis=1, join="inner")
    #Extracting INS_velocity in body frame
    inv_vel_frame = adma_scaled.columns.get_loc("ins_vel_frame")
    ins_vel_frame_df = adma_scaled.iloc[:,[inv_vel_frame+1,inv_vel_frame+2,inv_vel_frame+3]]
    adma_extracted_doi = pd.concat([adma_extracted_doi , ins_vel_frame_df.rename(columns={"x.9": "ins_vel_frame_x", "y.9": "ins_vel_frame_y", "z.9": "ins_vel_frame_z"})],axis=1, join="inner")
    #Extracting INS Vel horizontal
    inv_vel_hor = adma_scaled.columns.get_loc("ins_vel_hor")
    ins_vel_hor_df = adma_scaled.iloc[:,[117,118,119]]
    adma_extracted_doi = pd.concat([adma_extracted_doi , ins_vel_hor_df.rename(columns={"x.8": "ins_vel_hor_x", "y.8": "ins_vel_hor_y", "z.8": "ins_vel_hor_z"})],axis=1, join="inner")


    """What should be the logic to associate data from ADMA and OBD??
    The raw obd data is received at one time stamp and then it takes some time for the decoder to decipher it. The typical lag is 2 milliseconds.
    So ROSbagtimestamp of decoded OBD data vs acutal real data is not much different but problem occurs with ROS time stamp of ADMA scaled data
    
    In ADMA scaled data 4-5 values have very close rosbagtime stamp in .05msec and then there is some 40 msec gap. (Major Problem of synchronization)
    This is analysed with the commented out code to compare syncronization with matching rosbagtime stamps vs the method described below
    
    Since we are creating an dataset to correlate ground truth data of ADMA as close as possible in time stamps to OBD. This logic is used
    1) Time at which ADMA data is recorded ( time computed from INS time in ADMA_scaled with a formula )
    2) Time at which OBD data is received (Raw OBD data) which is passed through header
    3) During matching additionally forward direction is used which is better to reduce some discrepancies in delay of OBD data.
    This method is okay and did not lead to loss of data which happened with matching rosbag time stamps"""


    ##Data extraction from CSV files of OBD data with closest associated time stamps (10 Milliseconds)
    #adma_extracted_doi['timestamp'] = pd.to_datetime(adma_extracted_doi['rosbagTimestamp'], unit='ns')#For time synchronization
    adma_extracted_doi = adma_extracted_doi.rename(columns={"rosbagTimestamp": "rosbagTimestamp_ADMA"})
    adma_extracted_doi["sync_timer_sec"] = adma_extracted_doi["INS_time_sec"]
    tolerance_time = pd.Timedelta(milliseconds=10)

    #Lateral acceleration synchronization
    obd_data_latacc = pd.read_csv(f"{dataset_location}/{rosbag_name}/_slash_LatAccelInfo.csv")
    obd_data_latacc['latacc_OBDtime_sec'] = obd_data_latacc["secs"]+ ((obd_data_latacc['nsecs'])/10**9)
    obd_data_latacc['sync_timer_sec'] = obd_data_latacc['latacc_OBDtime_sec']
    obd_data_latacc=obd_data_latacc.drop(columns=['timestampLatAccel', 'secs',"nsecs","rosbagTimestamp"])

    #Forward is chosen for the logic that 10 milliseconds delay could be compensated for OBD data.
    #As we use generated time of ADMA and not rosbagtime. But for OBD we have only rawobd receipt rosbag time
    merged_df = pd.merge_asof(adma_extracted_doi, obd_data_latacc, on='sync_timer_sec', direction='forward', allow_exact_matches=False, tolerance=.010)
    duplicates_mask = merged_df.duplicated(subset='latacc_OBDtime_sec', keep='first')

    # Replace duplicates from df2 with NaN in the merged DataFrame
    merged_df.loc[duplicates_mask, 'latacc_OBDtime_sec'] = np.nan
    merged_df.loc[duplicates_mask, 'LateralAcceleration'] = np.nan

    #Lateral Acceleration time synchronization with ROS bagtimestamp
    #obd_data_lataccinput = pd.read_csv("Data_set_unzipped/1bagfromeduardo/_slash_LatAccelInfo.csv")
    #obd_data_lataccinput['timestamp'] = pd.to_datetime(obd_data_lataccinput['rosbagTimestamp'], unit='ns')
    #obd_data_lataccinput=obd_data_lataccinput.drop(columns=['timestampLatAccel','secs',"nsecs" ])
    #obd_data_lataccinput=obd_data_lataccinput.rename(columns={"rosbagTimestamp": "rosbagTimestamp_lataccinput", 'LateralAcceleration':"checkerlatacceleration"})
    #merged_df = pd.merge_asof(merged_df, obd_data_lataccinput, on='timestamp', direction='nearest', allow_exact_matches=False, tolerance=tolerance_time)
    #duplicates_mask = merged_df.duplicated(subset='rosbagTimestamp_lataccinput', keep='first')
    #Replace duplicates from df2 with NaN in the merged DataFrame
    #merged_df.loc[duplicates_mask, 'rosbagTimestamp_lataccinput'] = np.nan
    #merged_df.loc[duplicates_mask, 'checkerlatacceleration'] = np.nan


    #input acceleration synchronization
    obd_data_accinput = pd.read_csv(f"{dataset_location}/{rosbag_name}/_slash_ams_accel_input.csv")
    obd_data_accinput['accinput_OBDtime_sec'] = obd_data_accinput["secs"]+ ((obd_data_accinput['nsecs'])/10**9)
    obd_data_accinput['sync_timer_sec'] = obd_data_accinput['accinput_OBDtime_sec']
    obd_data_accinput = obd_data_accinput.drop(columns=['timestampAmsAccelInput', 'secs',"nsecs","rosbagTimestamp"])
    merged_df = pd.merge_asof(merged_df, obd_data_accinput, on='sync_timer_sec', direction='forward', allow_exact_matches=False, tolerance=.010)
    duplicates_mask = merged_df.duplicated(subset='accinput_OBDtime_sec', keep='first')
    merged_df.loc[duplicates_mask, 'accinput_OBDtime_sec'] = np.nan
    merged_df.loc[duplicates_mask, 'ams_accel_input_Percent'] = np.nan


    #Brake Pressure
    obd_data_brakepressure = pd.read_csv(f"{dataset_location}/{rosbag_name}/_slash_BrakePressInfo.csv")
    obd_data_brakepressure ['brakepressure_OBDtime_sec'] = obd_data_brakepressure ["secs"]+ ((obd_data_brakepressure ['nsecs'])/10**9)
    obd_data_brakepressure['sync_timer_sec'] = obd_data_brakepressure['brakepressure_OBDtime_sec']
    obd_data_brakepressure  = obd_data_brakepressure.drop(columns=['timestampBrakePressInDec', 'secs',"nsecs","rosbagTimestamp" ,'secs.1',"nsecs.1","timestampBrakeLight","BrakeLight"])
    merged_df = pd.merge_asof(merged_df, obd_data_brakepressure, on='sync_timer_sec', direction='forward', allow_exact_matches=False, tolerance=.010)
    duplicates_mask = merged_df.duplicated(subset='brakepressure_OBDtime_sec', keep='first')
    merged_df.loc[duplicates_mask, 'brakepressure_OBDtime_sec'] = np.nan
    merged_df.loc[duplicates_mask, 'BrakePressInDec'] = np.nan

    #Speedometer Value
    obd_data_speedo = pd.read_csv(f"{dataset_location}/{rosbag_name}/_slash_SpeedoInfo.csv")
    obd_data_speedo ['speedo_OBDtime_sec'] = obd_data_speedo["secs"]+ ((obd_data_speedo['nsecs'])/10**9)
    obd_data_speedo ['sync_timer_sec'] = obd_data_speedo['speedo_OBDtime_sec']
    obd_data_speedo = obd_data_speedo.rename(columns={"rosbagTimestamp":"rostimestamp_speedo"})
    obd_data_speedo   = obd_data_speedo.drop(columns=['timestampSpeedoInDec', 'secs',"nsecs"])
    merged_df = pd.merge_asof(merged_df, obd_data_speedo , on='sync_timer_sec', direction='forward', allow_exact_matches=False, tolerance=.010)
    duplicates_mask = merged_df.duplicated(subset='speedo_OBDtime_sec', keep='first')
    merged_df.loc[duplicates_mask, 'speedo_OBDtime_sec'] = np.nan
    merged_df.loc[duplicates_mask, 'SpeedoInDec'] = np.nan
    merged_df.loc[duplicates_mask, 'timestampSpeedoInDec'] = np.nan

    #Steering wheel position
    obd_data_swpos = pd.read_csv(f"{dataset_location}/{rosbag_name}/_slash_SteerWheelInfo.csv")
    obd_data_swpos['sw_OBDtime_sec'] = obd_data_swpos["secs"]+ ((obd_data_swpos['nsecs'])/10**9)
    obd_data_swpos['sync_timer_sec'] = obd_data_swpos['sw_OBDtime_sec']
    obd_data_swpos = obd_data_swpos.drop(columns=['timestampSWPosinDec', 'secs',"nsecs","rosbagTimestamp"])
    merged_df = pd.merge_asof(merged_df, obd_data_swpos , on='sync_timer_sec', direction='forward', allow_exact_matches=False, tolerance=.010)
    duplicates_mask = merged_df.duplicated(subset='sw_OBDtime_sec', keep='first')
    merged_df.loc[duplicates_mask, 'sw_OBDtime_sec'] = np.nan
    merged_df.loc[duplicates_mask, 'SWPosinDec'] = np.nan


    #Velocity Front
    obd_data_velfront = pd.read_csv(f"{dataset_location}/{rosbag_name}/_slash_VelFrontInfo.csv")
    obd_data_velfront ['velfront_OBDtime_sec'] = obd_data_velfront ["secs"]+ ((obd_data_velfront ['nsecs'])/10**9)
    obd_data_velfront ['sync_timer_sec'] = obd_data_velfront ['velfront_OBDtime_sec']
    obd_data_velfront  = obd_data_velfront.drop(columns=['timestampVelFrontInDec', 'secs',"nsecs","rosbagTimestamp"])
    merged_df = pd.merge_asof(merged_df, obd_data_velfront , on='sync_timer_sec', direction='forward', allow_exact_matches=False, tolerance=.010)
    duplicates_mask = merged_df.duplicated(subset='velfront_OBDtime_sec', keep='first')
    merged_df.loc[duplicates_mask, 'velfront_OBDtime_sec'] = np.nan
    merged_df.loc[duplicates_mask, 'VelFRInDec'] = np.nan
    merged_df.loc[duplicates_mask, 'VelFLInDec'] = np.nan


    #Velocity Rear
    obd_data_velrear = pd.read_csv(f"{dataset_location}/{rosbag_name}/_slash_VelRearInfo.csv")
    obd_data_velrear ['velrear_OBDtime_sec'] = obd_data_velrear ["secs"]+ ((obd_data_velrear ['nsecs'])/10**9)
    obd_data_velrear ['sync_timer_sec'] = obd_data_velrear ['velrear_OBDtime_sec']
    obd_data_velrear  = obd_data_velrear.drop(columns=['timestampVelRearInDec', 'secs',"nsecs","rosbagTimestamp"])
    merged_df = pd.merge_asof(merged_df, obd_data_velrear , on='sync_timer_sec', direction='forward', allow_exact_matches=False, tolerance=.010)
    duplicates_mask = merged_df.duplicated(subset='velrear_OBDtime_sec', keep='first')
    merged_df.loc[duplicates_mask, 'velrear_OBDtime_sec'] = np.nan
    merged_df.loc[duplicates_mask, 'VelRRInDec'] = np.nan
    merged_df.loc[duplicates_mask, 'VelRLInDec'] = np.nan

    #yaw rate
    obd_data_yawrate = pd.read_csv(f"{dataset_location}/{rosbag_name}/_slash_YawRateInfo.csv")
    obd_data_yawrate ['yaw_OBDtime_sec'] = obd_data_yawrate ["secs"]+ ((obd_data_yawrate ['nsecs'])/10**9)
    obd_data_yawrate ['sync_timer_sec'] = obd_data_yawrate ['yaw_OBDtime_sec']
    obd_data_yawrate  = obd_data_yawrate.drop(columns=['timestampGierrateRohsignal', 'secs',"nsecs","rosbagTimestamp"])
    merged_df = pd.merge_asof(merged_df, obd_data_yawrate , on='sync_timer_sec', direction='forward', allow_exact_matches=False, tolerance=.010)
    duplicates_mask = merged_df.duplicated(subset='yaw_OBDtime_sec', keep='first')
    merged_df.loc[duplicates_mask, 'yaw_OBDtime_sec'] = np.nan
    merged_df.loc[duplicates_mask, 'GierrateDegXSec'] = np.nan

    merged_df = merged_df.rename(columns={"LateralAcceleration": "LatAcc_obd", "ams_accel_input_Percent": "acc_input%_obd", "BrakePressInDec": "brake_pressure_obd","SpeedoInDec": "speedo_obd","SWPosinDec": "SW_pos_obd","VelFRInDec": "VelFR_obd",
                              "VelFLInDec": "VelFL_obd","VelRRInDec": "VelRR_obd","VelRLInDec": "VelRL_obd",
                              "GierrateDegXSec": "yaw_rate"}, errors="raise")

    """The following quantities from OBD are added to the ADMA data
    1) Lateral acceleration, 2) Acceleration Input percentage 3) Brake Pressure 4) Speedometer speed
    5) Steering Wheel position 6) Velocity of each tire 7) Yaw rate"""

    save_filename = rosbag_name+"ADMA+OBD_synced.csv"
    merged_df.to_csv(f"{dataset_location}/{rosbag_name}/{save_filename}", encoding='utf-8', index=False)




