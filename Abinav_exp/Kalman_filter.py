import numpy as np
import pandas as pd
from numpy.linalg import inv


#Values for linear single model with constant velocity
i_s = 22 #based on internet and used as the same for KMB-1
c_alpha_f = 80000 #80000  #Botsch values
c_alpha_r = 100000#100000 #Botsch values
m = 1050 #Internet based weight 975. Plus our weight and some extra allowance for sensors
lf = 1.0732
lr = 0.7998
I_z = 1500#1900 #Botsch values

vel = 0
T = 0.02
# State matrix a and b
def state_matrix(vel):
    matrix_a = np.array([[1 - ((c_alpha_f + c_alpha_r)/(m*vel))*T, (((c_alpha_r*lr-c_alpha_f*lf)/(m*vel*vel)) - 1)*T],
                      [((c_alpha_r*lr - c_alpha_f*lf)/I_z) * T, 1 - (((c_alpha_f*lf*lf + c_alpha_r*lr*lr)/(I_z*vel))*T)]])
    matrix_b = np.array([[(c_alpha_f/(m*vel))*T, (c_alpha_r/(m*vel))*T], [(c_alpha_f*lf/I_z)*T, -(c_alpha_r*lr/I_z)*T]])
    return matrix_a,matrix_b
# Measurement matrix
matrix_c = np.matrix([[1,0],[0,1]])
matrix_g = np.array([[T*T/2,0],[0,T*T/2]])
# Noise matrix can be experimented for improvement
#2000 was max double derivative for slip angle beta dot after using convolve for 5 steps averaging.
#3500 was the max double derivative for yaw rate after using convolve for 5 steps averaging.
Cns = np.array([[(np.pi*2000/3*180)**2, 0] ,[0, (np.pi*3500/3*180)**2]]) #Actual system noise
#Cns = np.array([[0.5, 0] ,[0, 0.5]])
#3 degrees could be the variation from tire based slip angle and 5 degrees for yaw rate.
Cnm = np.array([[(np.pi*3/3*180)**2, 0] ,[0, (np.pi*10/3*180)**2]]) #Actual measurement noise
#Cnm = np.array([[(np.pi*0/3*180)**2, 0] ,[0, (np.pi*0/3*180)**2]])
def pred_a_prior(matrix_a,matrix_b,matrix_g,x_n_1,u_n_1,p_n_1):
    x_a_prior = matrix_a @ x_n_1 + matrix_b @ u_n_1
    p_a_prior = matrix_a @ p_n_1 @ np.transpose(matrix_a) + matrix_g @ Cns @ np.transpose(matrix_g)
    return x_a_prior,p_a_prior
def inno_gain_covariance(y_n,matrix_c,x_a_prior,p_a_prior,Cnm):
    innovation = y_n - matrix_c @ x_a_prior
    inno_residual = matrix_c @ p_a_prior @ np.transpose(matrix_c) + Cnm
    gain = p_a_prior @ np.transpose(matrix_c) @ inv(inno_residual)
    return innovation,inno_residual,gain
def update_posterior (x_a_prior,gain,innovation,matrix_c,p_a_prior):
    x_post = x_a_prior + gain @ innovation
    p_post = ( np.identity(2) - gain @ matrix_c ) @ p_a_prior
    return x_post,p_post





