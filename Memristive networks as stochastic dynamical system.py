###############################################################################
# Copyright (C) 2024 by
# Gianluca Milano
# 
# This work is licensed under a  
# 
# Creative Commons Attribution Non-Commercial 4.0 License
# ( https://creativecommons.org/licenses/by-nc/4.0/ )
# 
# Please contact g.milano@inrim.it for information.
#
###############################################################################

import matplotlib.pyplot as plt
import random
import bpl

import numpy as np
from numpy import exp, sqrt, zeros

random.seed(15)

#%%Ornstein-Uhlenbeck parameters 

# Deterministic parameters 
kp0 = 0.0005217037178270165
kd0 = 37.93422536389374
eta_p = 1.2922283233062284
eta_d = 1.785933208836627
G_min = 1.4558148272052695e-07
G_max = 6.857961268862245e-05

# Parameters for noise
sigma = 0.00032

#Parameters for jumps

# Parameters for jump amplitude  P(A) = A_coeff*A^(-A_exp)
A_exp = 2.78
A_coeff = 1

A_min = 0.00000006925924850166/(G_max-G_min)*0.6295  #lower bound
A_max = 0.00000110821/(G_max-G_min)*0.6295           #higher bound

# Parameters for jumps probability    P(dt)= E*dt^-gamma (with gamma = 1, homogeneous Poisson process)
E = 0.08243

#%% ELECTRICAL STIMULATION
    
#Voltage list
x = 3.6      ##pulse amplitude
n = 637      ##pulse length
dt= 0.6295

Vin_list=[x]*n

##Time list
delta_t=[dt]*n
timesteps= len(delta_t)
time_list=[]
time_list.append(0)

for t in range(1,len(delta_t)):
    s = time_list[t-1] + delta_t[t]
    time_list.append(s)
    
#%% DETERMINISTIC MODEL

#iterative solution

def deterministic_model(time_list, kp0, kd0, eta_p, eta_d, G_0, G_min, G_max):
    g = zeros(len(time_list),)
    G = zeros(len(time_list))
    
    for i in range(0, len(time_list)):
        signal = Vin_list[i]
        kp = kp0*exp(eta_p*signal)
        kd = kd0*exp(-eta_d * signal)
        
        g_tilde = kp/(kp+kd)
        theta   = kp+kd
        
        if i == 0:
            G[i] = G_0
            g[i] = (G_0-G_min)/(G_max-G_min)
        else:
            delta_t = time_list[i] - time_list[i-1]
           
            #equation 1: ionics
            g[i] = g_tilde*(1-exp(-theta*delta_t))+g[i-1]*exp(-theta*delta_t)
            
            #equation 2: electronics
            G[i] = G_min*(1-g[i])+G_max*g[i]  
            
    return g, G

#%% STOCHASTIC MODEL (DETERMINISTIC + NOISE)

#iterative solution

def stochastic_model_noise(time_list, kp0, kd0, eta_p, eta_d, G_0, G_min, G_max, sigma):
    g = zeros(len(time_list),)
    G = zeros(len(time_list))
    
    for i in range(0, len(time_list)):
        signal = Vin_list[i]
        kp = kp0*exp(eta_p*signal)
        kd = kd0*exp(-eta_d * signal)
        
        g_tilde = kp/(kp+kd)
        theta   = kp+kd
        
        
        if i == 0:
            G[i] = G_0
            g[i] = (G_0-G_min)/(G_max-G_min)
        else:
            delta_t = time_list[i] - time_list[i-1]
            
            #equation 1: ionics
            sigma_t = sqrt(((1-exp(-2*theta*delta_t))*(sigma**2))/(2*theta)) 
            g[i] = g_tilde*(1-exp(-theta*delta_t))+g[i-1]*exp(-theta*delta_t) + sigma_t*np.random.normal(0, 1)
            
            #equation 2: electronics
            G[i] = G_min*(1-g[i])+G_max*g[i]  
            
    return g, G

#%% STOCHASTIC MODEL (DETERMINISTIC + NOISE + JUMPS)

#Euler Mayurama solution

def stochastic_model_noise_jumps(time_list, kp0, kd0, eta_p, eta_d, G_0, G_min, G_max, sigma, A_exp, A_coeff, A_min, A_max):
    g = zeros(len(time_list),)
    G = zeros(len(time_list))
    
    for i in range(0, len(time_list)):
        signal = Vin_list[i]
        kp = kp0*exp(eta_p*signal)
        kd = kd0*exp(-eta_d * signal)
        
        g_tilde = kp/(kp+kd)
        theta   = kp+kd
        
        if i == 0:
            G[i] = G_0
            g[i] = (G_0-G_min)/(G_max-G_min)
        else:
            delta_t = time_list[i] - time_list[i-1]
            #equation 1: ionics
            
            #jump probability
            jump_probability = np.random.choice([0,1], p=[1-E*delta_t, E*delta_t])
                        
            #jump amplitude
            jump_amplitude   = bpl.sample(alpha=A_exp, size=1, xmin=A_min, xmax=A_max)
            
            #jump direction
            direction_prob = (g_tilde-g[i-1])/(2*(1+2*A_max))
            jump_direction   = np.random.choice([-1,1],p=[0.5-direction_prob,0.5+direction_prob])
        
    
            delta_g = theta*(g_tilde-g[i-1])*delta_t + sigma*np.sqrt(delta_t)*np.random.normal(0, 1) + jump_probability*jump_direction*jump_amplitude
            g[i] = g[i-1]+delta_g
            
            #equation 2: electronics
            G[i] = G_min*(1-g[i])+G_max*g[i] 
            
    return g, G


#%%% MODELING CONDUCTANCE TIME TRACE

#set initial condition
G_0 = 1.23364E-5


#model_deterministic_2
g_list_deterministic_model, G_list_deterministic_model = deterministic_model(time_list, kp0, kd0, eta_p, eta_d, G_0, G_min, G_max)
#model_deterministic_stochastic_iterative
g_list_stochastic_noise_model, G_list_stochastic_noise_model = stochastic_model_noise(time_list, kp0, kd0, eta_p, eta_d, G_0, G_min, G_max, sigma)
#model_deterministic_stochastic_jumps_Euler_2
g_list_stochastic_noise_jumps_model, G_list_stochastic_noise_jumps_model = stochastic_model_noise_jumps(time_list, kp0, kd0, eta_p, eta_d, G_0, G_min, G_max, sigma, A_exp, A_coeff, A_min, A_max)

#%% PLOTS

#Plot G
plt.figure(10,figsize=(10,7))
#plt.title(title)
plt.plot(time_list, G_list_deterministic_model, 'g', label='Deterministic model', linewidth=2.2)
plt.plot(time_list, G_list_stochastic_noise_model, 'k', label='Stochastic model (deterministic + noise)', linewidth=2.2)
plt.plot(time_list, G_list_stochastic_noise_jumps_model, 'r', label='Stochastic model (deterministic + noise + jumps)', linewidth=2.2)
plt.legend(fontsize=20)
plt.grid()
plt.xlabel('time [s]', fontsize=20)
plt.ylabel('G [S]', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

