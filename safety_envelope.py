"""This code analyses and reports the implementation of safety envelope
Required components are the same as fusion data_fusion_2_test.csv with additional best fusion values
exploration of speed and steering to switch safety envelopes between one and two
safety envelope does not use the machine learning model output"""


"""The evaluation of safety envelope could be done with two ideas
1) area of the safety envelope
2) distance during top5 close hits of the safety envelope
A priority is for the test dataset the safety envelope should not be violated"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('data_fusion_2_test.csv')
fusion_best_pred = pd.read_csv('y_pred_original_trained_on_train_best.csv')
fusion_best_pred.columns = ['best_fusion']
fusion_best_pred = fusion_best_pred.iloc[:-2] #Check this to make both same size
dataset = pd.concat([dataset,fusion_best_pred],axis=1)
print('Hi')

#Safety envelope-1 (Safety with only KMB models)
dataset['se1_uplimit'] =  dataset[['kmb_1_val', 'kmb_2_val']].max(axis=1) + 4
dataset['se1_downlimit'] =  dataset[['kmb_1_val', 'kmb_2_val']].min(axis=1)  - 4
condition_1 = dataset['true_val'].between(dataset['se1_downlimit'],dataset['se1_uplimit'])
area_1 = sum((dataset["se1_uplimit"] - dataset['se1_downlimit']).abs())*.02
print('total violations in test dataset is',len(dataset)-sum(condition_1))
print('length of the test dataset',len(dataset))
print('area',area_1)

#Safety envelope-2 (Safety with only KMB models and uncertainty)
#Sucessful combinations(x2.5,x4), (/4,x4)
dataset['se1_envelope'] = dataset['kmb_1_var']*.25 #/4
dataset['se2_envelope'] = dataset['kmb_2_var']*1.5 #*4
dataset['se2_uplimit'] =  dataset[['kmb_1_val', 'kmb_2_val']].max(axis=1)+dataset[['se1_envelope', 'se2_envelope']].max(axis=1)+1
dataset['se2_downlimit'] = dataset[['kmb_1_val', 'kmb_2_val']].min(axis=1)-dataset[['se1_envelope', 'se2_envelope']].max(axis=1)-1
difference = abs(dataset['se2_uplimit'] - dataset['se2_downlimit'])
# mask = difference < 2
# dataset.loc[mask, 'se2_uplimit'] += 1
# dataset.loc[mask, 'se2_downlimit'] -= 1
dataset.loc[dataset['se2_uplimit']>10, 'se2_uplimit'] = 13
dataset.loc[dataset['se2_downlimit']<-10, 'se2_downlimit'] = -13
condition_2 = dataset['true_val'].between(dataset['se2_downlimit'],dataset['se2_uplimit'])
area_2 = sum((dataset["se2_uplimit"] - dataset['se2_downlimit']).abs())*0.02
print('total violations in test dataset is',len(dataset)-sum(condition_2))
print('length of the test dataset',len(dataset))
print('area',area_2)

plt.figure(figsize=(10, 6))
plt.plot(dataset.loc[condition_2,"true_val"], label='Ground truth',color = "g")
plt.plot(dataset.loc[condition_2,"se2_uplimit"], label='se2 uplimit')
plt.plot(dataset.loc[condition_2,"se2_downlimit"], label='se2 downlimit')
# plt.plot(dataset.loc[condition_2,"kmb_1_var"]*.25, label='kmb-1')
# plt.plot(dataset.loc[condition_2,"kmb_2_var"]*1.5, label='kmb-2')
plt.xlabel('Time stamps')
plt.ylabel('Slip angle in degrees')
plt.legend()
plt.show()