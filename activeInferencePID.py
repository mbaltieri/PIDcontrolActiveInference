#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:52:21 2018

Active inference version of a PID controller. Example built on cruise control 
problem from Astrom and Murray (2010), pp 65-69.
In this specific example, only Proportional and Integral terms are used, 
since standard cruise control problems do not usually adopt the D-term.

@author: manuelbaltieri
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
large_value = np.exp(50)

### define font size for plots ###
#
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)            # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)       # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)     # fontsize of the figure title
#

dt = .05                                                    # integration step

### generative model constants ###
#
alpha = 100000.                                             # drift in Generative Model
gamma = 1                                                   # drift in OU process (if you want to simulate coloured noise)
obs_states = 1
hidden_states = 1                                           # x, in Friston's work
hidden_causes = 1                                           # v, in Friston's work
##states = obs_states + hidden_states
temp_orders_states = 3                                      # generalised coordinates for hidden states x
temp_orders_causes = 3                                      # generalised coordinates for hidden causes v
#

### cruise control problem from Astrom and Murray (2010), pp 65-69    
# environment parameters
ga = 9.81                                                   # gravitational acceleration
theta = 4.                                                  # hill angle
C_r = .01                                                   # rolling friction coefficient
C_d = .32                                                   # drag coefficient
rho = 1.3                                                   # air density
A = 2.4                                                     # frontal area agent

# car's parameters
#m = 1000                                                    # car mass (book example)
m = 100                                                     # car mass
T_m = 190                                                   # maximum torque
omega_m = 420                                               # engine speed to reach T_m
r_g = 12                                                    # = gear ration/wheel radius,
                                                            # a1 = 40, a2 = 25, a3 = 16, a4 = 12, a5 = 10
beta = .4                                                   # motor constant
#


### FUNCTIONS ###

## motor action ##
def sigmoid(x):                                             # limit the motor control (not used in the paper)
    return np.tanh(x)
    return 1 / (1+np.exp(-x))

def dsigmoid(x):                                            # derivative of the above function to improve active inference (not used in the paper)
    return sigmoid(x) * (1 - sigmoid(x))

## cruise control problem ##
def force_gravitation(theta):
    return m * ga * np.sin(theta)

def force_friction(v):
    return m * ga * C_r * np.sign(v)

def force_drag(v):
    return .5 * rho * C_d * A * v**2

def force_disturbance(v, theta):
    return force_gravitation(theta) + force_friction(v) + force_drag(v)

def force_drive(v, u):
    return r_g * u * torque(v)

def torque(v):
    return T_m * (1 - beta * (omega(v) / omega_m)**2)

def omega(v):
    return r_g * v

## free energy functions ##
# generative process
    
# x: hidden states
# v: hidden causes
# a: action
# w: fluctuations in process dynamics
def g(x, v):
    return x

def f(x, v, a):
    return (force_drive(x, v + a) - force_disturbance(x, theta)) / m

# generative model
def g_gm(x, v):
    return g(x, v)

def f_gm(x, v):
    # a = 0.0, no action in generative model
    return f(x, v, 0.0)

def getObservation(x, v, a, w):
    # w = 0.0, no added fluctuations in the car dynamics
    x[:, 1:] = f(x[:, :-1], v, a)# + w
    x[:, 0] += dt * x[:, 1]
    return g(x[:, :-1], v)

# main function, PID control
def pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, mu_gamma_z_input, mu_gamma_w_input):
    iterations = int(T / dt)
    
    ### car variables ###
    x = np.zeros((hidden_states, temp_orders_states))                       # hidden states
    v = np.zeros((hidden_causes, temp_orders_states - 1))                   # hidden causes
    y = np.zeros((obs_states, temp_orders_states - 1))                      # observations (without noise)
    
    # noise on sensory input (world - generative process)
#    gamma_z = 0. * np.ones((obs_states, temp_orders_states))                # sensory log-precisions, correlated noise
#    gamma_z[:,1] = gamma_z[:,0] - np.log(2 * gamma)
#    gamma_z[:,2] = gamma_z[:,1] - np.log(2 * gamma)
    
    gamma_z = 5. * np.ones((obs_states, temp_orders_states))                # sensory log-precisions, uncorrelated noise
    pi_z = np.exp(gamma_z) * np.ones((obs_states, temp_orders_states))
    sigma_z = 1 / (np.sqrt(pi_z))
    z = np.zeros((iterations, obs_states, temp_orders_states))
    for i in range(obs_states):
        for j in range(temp_orders_states):
            z[:, i, j] = sigma_z[i, j] * np.random.randn(1, iterations)
    
    # noise on motion of hidden states (world - generative process)
    gamma_w = 32 * np.ones((hidden_states, temp_orders_states))             # process log-precisions, uncorrelated noise (not used in the paper)
    gamma_w[:,1] = gamma_w[:,0] - np.log(2 * gamma)
    gamma_w[:,2] = gamma_w[:,1] - np.log(2 * gamma)
    pi_w = np.exp(gamma_w) * np.ones((hidden_states, temp_orders_states))
    sigma_w = 1 / (np.sqrt(pi_w))
    w = np.zeros((iterations, hidden_states, temp_orders_states))
    for i in range(hidden_states):
        for j in range(temp_orders_states - 1):
            w[:, i, j] = sigma_w[i, j] * np.random.randn(1, iterations)    
    
    
    
    ### free energy variables ###    
    mu_x = 0.0001*np.random.randn(hidden_states, temp_orders_states)        # expected hidden states
    mu_v = np.zeros((hidden_causes, temp_orders_states))                    # expected hidden causes (not used in the paper, see equations 20-21)
    a = np.zeros((obs_states, temp_orders_states-1))
    
    # priors
    eta_x = np.zeros((hidden_causes, temp_orders_states - 1))               # priors on expected hidden states, entering as expected hidden causes
    eta_gamma_z = np.zeros((obs_states, temp_orders_states - 1))            # priors on expected sensory precisions (not used in the paper), hyperpriors
    eta_gamma_w = np.zeros((hidden_states, temp_orders_states - 1))         # priors on expected process precisions (not used in the paper), hyperpriors
    
    # minimisation of variables and (hyper)parameters
    dFdmu_x = np.zeros((hidden_states, temp_orders_states))
    dFdmu_v = np.zeros((hidden_causes, temp_orders_states))
    Dmu_x = np.zeros((hidden_states, temp_orders_states))
    Dmu_v = np.zeros((hidden_causes, temp_orders_states))
    
    dFdmu_gamma_z = np.zeros((hidden_causes, temp_orders_states))
    phi_z = np.zeros((obs_states, temp_orders_states-1))
    phi_w = np.zeros((hidden_states, temp_orders_states-1))
    
    # learning rates (not used in the paper)
    k_mu_x = 1                                                              # learning rate perception
    k_a = 1                                                                 # learning rate action
    k_mu_gamma_z = 1                                                        # learning rate attention (sensory precisions)
    k_mu_gamma_w = 1                                                        # learning rate attention (process precisions)
    
    # damping terms for hyperparameters optimisation
    kappa_z = 5                                                             # damping on sensory precisions minimisation
    kappa_w = 10                                                            # damping on process precisions minimisation
    
    # agent's estimates of the noise (agent - generative model)
    mu_gamma_z = mu_gamma_z_input * np.ones((obs_states, temp_orders_states - 1))    # sensory log-precisions, correlated noise
    mu_gamma_z[0, 1] = mu_gamma_z[0, 0] - np.log(2 * gamma)
    mu_pi_z = np.exp(mu_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
    
    mu_gamma_w = mu_gamma_w_input * np.ones((hidden_states, temp_orders_states - 1)) # process log-precisions, correlated noise
    mu_gamma_w[0, 1] = mu_gamma_w[0, 0] - np.log(2)
    mu_pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
    
    # hyperpriors' precisions (not used in the paper, left for future work)
    mu_gamma_gamma_z = - large_value * np.ones((obs_states, temp_orders_states - 1))
    mu_p_gamma_z = np.exp(mu_gamma_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
    
    mu_gamma_gamma_w = - large_value * np.ones((hidden_states, temp_orders_states - 1))
    mu_p_gamma_w = np.exp(mu_gamma_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
    
    
    # history, here for reference, only some variable are used
    x_history = np.zeros((iterations, hidden_states, temp_orders_states))
    y_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
    v_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))
    psi_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
    mu_x_history = np.zeros((iterations, hidden_states, temp_orders_states))
    eta_x_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))
    eta_gamma_z_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
    eta_gamma_w_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))
    a_history = np.zeros((iterations, temp_orders_states - 1))
    mu_gamma_z_history = np.zeros((iterations, temp_orders_states-1))
    mu_gamma_w_history = np.zeros((iterations, temp_orders_states-1))
    mu_pi_z_history = np.zeros((iterations, temp_orders_states-1))
    mu_pi_w_history = np.zeros((iterations, temp_orders_states-1))
    dFdmu_gamma_z_history = np.zeros((iterations, temp_orders_states-1))
    dFdmu_gamma_w_history = np.zeros((iterations, temp_orders_states-1))
    
    xi_z_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
    xi_w_history = np.zeros((iterations, hidden_states, temp_orders_states - 1))
    
    kappa_z_history = np.zeros((iterations,1))
    kappa_w_history = np.zeros((iterations,1))
    
    gamma_z_history = np.zeros((iterations, temp_orders_states))
    gamma_w_history = np.zeros((iterations, temp_orders_states))
    
    
    ### initialisation ###
    # hyperpriors (not used in the paper, left for reference)
    eta_gamma_z[0, 0] = 3.
    eta_gamma_z[0, 1] = 1.
    
    eta_gamma_w[0, 0] = - 18.
    eta_gamma_w[0, 1] = - 18.
    
    desired_velocity = 10.
    eta_x[0, 0] = desired_velocity
    
    
    ### main loop ###
    for i in range(iterations - 1):
        print(i)
        
        # re-encode precisions after hyperparameters update
        mu_pi_z = np.exp(mu_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
        mu_pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
        mu_p_gamma_z = np.exp(mu_gamma_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
        mu_p_gamma_w = np.exp(mu_gamma_gamma_w) * np.ones((obs_states, temp_orders_states - 1))
        
        # include an external disturbance to test integral term
        if (simulation == 0) or (simulation == 1):
            if (i > iterations/2):
                v[0,0] = 3.0
        
        # test 2DOF
        if (simulation == 2):
            if (i > iterations/2):
                eta_x[0, 0] = desired_velocity - 3

        
        
        # Analytical noise, for one extra level of generalised cooordinates, this is equivalent to an ornstein-uhlenbeck process
    #    dw2 = - gamma * w[i, 0, 1] + w[i, 0, 2] / np.sqrt(dt)
    #    dz2 = - gamma * z[i, 0, 1] + z[i, 0, 2] / np.sqrt(dt)
    #    
    #    w[i+1, 0, 1] = w[i, 0, 1] + dt * dw2
    #    z[i+1, 0, 1] = z[i, 0, 1] + dt * dz2
    #    
    #    dw = - gamma * w[i, 0, 0] + w[i, 0, 1]
    #    dz = - gamma * z[i, 0, 0] + z[i, 0, 1]
    #    
    #    w[i+1, 0, 0] = w[i, 0, 0] + dt * dw
    #    z[i+1, 0, 0] = z[i, 0, 0] + dt * dz
        
        y = getObservation(x, v, a, w[i, 0, :-1])
        psi = y + z[i, 0, :-1]
        
        ### minimise free energy ###
        # perception
        Dmu_x[0, :-1] = mu_x[0, 1:]
    #    dFdmu_x[0, :-1] = np.array([mu_pi_z * - (psi - mu_x[0, :-1]) + mu_pi_w * alpha * (mu_x[0, 1:] + alpha * (mu_x[0, :-1] - eta_x))]) # if the noise is not white
        if limit_case != 1:
            dFdmu_x[0, :-1] = np.array([mu_pi_z * - (y + z[i, 0, :-1]/np.sqrt(dt) - mu_x[0, :-1]) + mu_pi_w * alpha * (mu_x[0, 1:] + alpha * (mu_x[0, :-1] - eta_x))])
        else:
            # or use the line below to simulate indipendence of set-point adaptation
            # with respect to precisions of measurement noise
            dFdmu_x[0, :-1] = np.array([mu_pi_w * alpha * (mu_x[0, 1:] + alpha * (mu_x[0, :-1] - eta_x))])
        
        # action
    #    dFdy = mu_pi_z * (psi - mu_x[0, :-1])                                  # if the noise is not white
        dFdy = mu_pi_z * (y + z[i, 0, :-1]/np.sqrt(dt) - mu_x[0, :-1])
        dyda = np.ones((obs_states, temp_orders_states-1))
        dFda = np.zeros((obs_states, temp_orders_states-1))
        dFda[0, 0] = np.sum(dFdy * dyda)
        
        # attention
    #    dFdmu_gamma_z = .5 * (mu_pi_z * (psi - mu_x[0, :-1])**2 - 1) + mu_p_gamma_z * (mu_gamma_z - eta_gamma_z)       # if the noise is not white
        dFdmu_gamma_z = .5 * (mu_pi_z * (y**2 + z[i, 0, :-1]**2 + mu_x[0, :-1]**2 + 2*y*z[i, 0, :-1]/np.sqrt(dt) - 2*mu_x[0, :-1]*z[i, 0, :-1]/np.sqrt(dt) - 2*y*mu_x[0, :-1]) - 1) + mu_p_gamma_z * (mu_gamma_z - eta_gamma_z)
        dFdmu_gamma_w = .5 * (mu_pi_w * (mu_x[0, 1:] + alpha * (mu_x[0, :-1] - eta_x))**2 - 1) + mu_p_gamma_w * (mu_gamma_w - eta_gamma_w)
        
        
        ## update equations ##
        mu_x += dt * (Dmu_x - k_mu_x * dFdmu_x)
        a += dt * - k_a * dFda
        
        # only used for hyperparameters
        phi_z += dt * (- dFdmu_gamma_z - kappa_z * phi_z)
        phi_w += dt * (- dFdmu_gamma_w - kappa_w * phi_w)
        
        # test conditions for hyperparameters optimisation
        if simulation == 1 and (i > switch_condition_time/dt):
            if (i < iterations/2):
                mu_gamma_z += dt * k_mu_gamma_z * phi_z
        
        if simulation == 3 and (i > switch_condition_time/dt):
            if learning == 1:
                mu_gamma_z += dt * k_mu_gamma_z * phi_z
            else:
                if i < iterations/2:
                    mu_gamma_z += dt * k_mu_gamma_z * phi_z
                
            if i == iterations/2:
                mu_gamma_gamma_z = -111 * np.ones((obs_states, temp_orders_states - 1))         # uncomment or comment to get a prior or just follow the changing measurement noise
                
                gamma_z = 2. * np.ones((obs_states, temp_orders_states))                        # one of the sensors "breaks"
                pi_z = np.exp(gamma_z) * np.ones((obs_states, temp_orders_states))
                sigma_z = 1 / (np.sqrt(pi_z))
                z = np.zeros((iterations, obs_states, temp_orders_states))
                for j in range(obs_states):
                    for k in range(temp_orders_states):
                        z[:, j, k] = sigma_z[j, k] * np.random.randn(1, iterations)

        # save history
        y_history[i, :] = y
        psi_history[i, :] = psi
        mu_x_history[i, :, :] = mu_x
        eta_x_history[i] = eta_x
            
        eta_gamma_z_history[i] = eta_gamma_z
        eta_gamma_w_history[i] = eta_gamma_w
        a_history[i] = a
        v_history[i] = v
        mu_gamma_z_history[i] = mu_gamma_z
        mu_gamma_w_history[i] = mu_gamma_w
        
        gamma_z_history[i] = gamma_z
        gamma_w_history[i] = gamma_w    
    return psi_history, y_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history



### Simulations ###
# 0: PID control as active inference (Fig 2)
# 1: PID parameters tuning (Fig 4)
# 2: Active inference PID control with 2DOF, load disturbance response affected by pi_z + set point response affected by pi_w (Fig 3)
# 3: Summary statistics of IAE with/without parameters tuning (Fig 5)
# 4: Summary statistics of variance with continual/interrupted adaptation (Fig 6)


simulation = 4
learning = 0                                                        # learning precisions, 0 off, 1 on
limit_case = 0                                                      # simulation 3 requires to explicitly implement equation 24, 
                                                                    # since numerical approximation prevent the assumptions 
                                                                    # to be in place for varying precisions in simulation 3, 0 implicit, 1 explicit

if simulation == 0:
    T = 300
    switch_condition_time = 0                                       # time to start optimising precisions (not used in simulation 0)
    psi_history, y_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history = pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, -3.,-20)
elif simulation == 1:
    T = 300
    switch_condition_time = T/10                                    # time to start optimising precisions
    psi_history, y_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history =  pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, -3.,-20)
elif simulation == 2:
    T = 100
    switch_condition_time = 0                                       # time to start optimising precisions (not used in simulation 0)
    limit_case = 1
    
    simulation = 0
    psi_history, y_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history =  pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, 1.,-24)
    psi_history2, y_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history2, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history2 =  pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, 1.,-22)
    psi_history3, y_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history3, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history3 =  pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, 1.,-20)   
    
    simulation = 2
    
    psi_history4, y_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history4, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history4 =  pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, 1.,-24)
    psi_history5, y_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history5, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history5 =  pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, 1.,-22)
    psi_history6, y_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history6, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history6 =  pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, 1.,-20)
elif simulation == 3:
    T = 300
    iterations = int(T / dt)    
    simulations_number = 20
    switch_condition_time = T/10                                    # time to start optimising precisions
    
    variance_before = np.zeros((simulations_number, 2))
    variance_after = np.zeros((simulations_number, 2))
    
    for i in range(simulations_number):
        learning = 0
        psi_history, y_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history =  pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, -2.,-20)
        variance_before[i, 0] = np.var(y_history[int(T/dt/4):int(T/dt/2)-1, 0, 0])
        variance_after[i, 0] = np.var(y_history[int(T/dt/4*3):int(T/dt)-1, 0, 0])
        
        learning = 1
        psi_history2, y_history2, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history2, v_history, gamma_z_history, mu_gamma_z_history2, gamma_w_history, mu_gamma_w_history =  pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, -2.,-20)
        variance_before[i, 1] = np.var(y_history2[int(T/dt/4):int(T/dt/2)-1, 0, 0])
        variance_after[i, 1] = np.var(y_history2[int(T/dt/4*3):int(T/dt)-1, 0, 0])    
elif simulation == 4:
    T = 300
    iterations = int(T / dt)
    
    simulations_number = 20
    desired_velocity = 10
    epsilon = 0.5
    
    simulation = 0
    
    y_history_stats = np.zeros((simulations_number, iterations, obs_states, temp_orders_states - 1))
    y_history_stats2 = np.zeros((simulations_number, iterations, obs_states, temp_orders_states - 1))
    
    tauNoAdaptation = np.zeros(simulations_number,)
    tauAdaptation = np.zeros(simulations_number,)
    
    iaeNoAdaptation = np.zeros(simulations_number,)
    iaeAdaptation = np.zeros(simulations_number,)
    
    for i in range(simulations_number):
        random_sensory_precision = np.random.rand()*2 - 3.
        random_process_precision = np.random.rand()*2 - 22.
        
        simulation = 0
        switch_condition_time = 0
        psi_history, y_history_stats[i,:,:,:], mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history = pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, random_sensory_precision, random_process_precision)
        foo = np.where((y_history_stats[i,int(T/2/dt)+3:,0,0] <= desired_velocity + epsilon) & (y_history_stats[i,int(T/2/dt)+3:,0,0] >= desired_velocity - epsilon))[0]
                                                                # consider 3*dt of distance to avoid t = tau
        tauNoAdaptation[i] = foo[0]                             # first zero crossing after load disturbance
        
        simulation = 1
        switch_condition_time = T/10
        psi_history, y_history_stats2[i,:,:,:], mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, a_history, v_history, gamma_z_history, mu_gamma_z_history, gamma_w_history, mu_gamma_w_history = pidControl(simulation, T, dt, switch_condition_time, learning, limit_case, random_sensory_precision, random_process_precision)
        foo2 = np.where((y_history_stats2[i,int(T/2/dt)+3:,0,0] <= desired_velocity + epsilon) & (y_history_stats2[i,int(T/2/dt)+3:,0,0] >= desired_velocity - epsilon))[0]
                                                                # consider 3*dt of distance to avoid t = tau
        tauAdaptation[i] = foo2[0]                              # first zero crossing after load disturbance
        
        iaeNoAdaptation[i] = np.sum(np.absolute(y_history_stats[i,int(T/2/dt):int(T/2/dt+3+tauNoAdaptation[i]), 0, 0]))
        iaeAdaptation[i] = np.sum(np.absolute(y_history_stats2[i, int(T/2/dt):int(T/2/dt+3+tauAdaptation[i]), 0, 0]))

    simulation = 4                                              # to plot the right figure






### FIGURES ###

if simulation < 3:
    plt.figure(figsize=(9, 6))
    plt.xlim((0., T))
    if simulation == 0 or simulation == 1:
        plt.plot(np.arange(0, T-dt, dt), psi_history[:-1,0,0], 'b', linewidth=1, label='Sensed velocity, $\psi$')
        plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,0], 'r', linewidth=1, label='Expec. of velocity, $\mu_x$')    
        #plt.plot(np.arange(0, T-dt, dt), eta_x_history[:-1,0,0], 'k--', linewidth=1, label='Desired velocity, $\eta_x$')
    elif simulation == 2:
        x_min = 0
        x_max = 20
        plt.xlim((x_min, x_max))
        plt.plot(np.arange(x_min, x_max, dt), psi_history[int((x_min+40)/dt):int((x_max+40)/dt),0,0], 'b', linewidth=1, label='Sensed velocity, $\psi_1$; $\pi_w = exp($'+str(mu_gamma_w_history[1,0])+')')
        plt.plot(np.arange(x_min, x_max, dt), psi_history2[int((x_min+40)/dt):int((x_max+40)/dt),0,0], 'r', linewidth=1, label='Sensed velocity, $\psi_1$; $\pi_w = exp($'+str(mu_gamma_w_history2[1,0])+')')
        plt.plot(np.arange(x_min, x_max, dt), psi_history3[int((x_min+40)/dt):int((x_max+40)/dt),0,0], 'g', linewidth=1, label='Sensed velocity, $\psi_1$; $\pi_w = exp($'+str(mu_gamma_w_history[31,0])+')')
    
    plt.title('Car velocity')
    plt.xlabel('Time ($s$)')
    plt.ylabel('Velocity ($km/h$)')
    plt.legend(loc=1)
    if simulation == 0:
        plt.text(T+20, eta_x_history[-2,0,0], "$\eta_x$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
        plt.ylim((-10., 50.))
        plt.savefig("figures/activeInferencePID_a.pdf")
    elif simulation == 1:
        plt.text(T+20, eta_x_history[-2,0,0], "$\eta_x$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
        plt.axvline(x=switch_condition_time, linewidth=3, color='k', linestyle='-.')
        plt.ylim((-10., 50.))
        plt.savefig("figures/activeInferencePIDTuning_a.pdf")
    elif simulation == 2:
        plt.text(x_max+1.5, eta_x_history[-2,0,0], "$\eta_x$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
        plt.ylim((0., 15.))
        plt.title('Car velocity - Load disturbance')
        plt.legend(loc=4)
        plt.annotate("new ext. input",
                xy=((x_min+x_max)/2, 9.), xycoords='data',
                xytext=((x_min+x_max)/2-2.85, 6.), textcoords='data',
                arrowprops=dict(arrowstyle="simple",
                                connectionstyle="arc3"))
        plt.savefig("figures/activeInferencePIDLoad.pdf")


if simulation == 2:
    plt.figure(figsize=(9, 6))
    x_min = 0
    x_max = 20
    plt.xlim((x_min, x_max))
    plt.plot(np.arange(x_min, x_max, dt), psi_history4[int((x_min+40)/dt):int((x_max+40)/dt),0,0], 'b', linewidth=1, label='Sensed velocity, $\psi_1$; $\pi_w = exp($'+str(mu_gamma_w_history4[1,0])+')')
    plt.plot(np.arange(x_min, x_max, dt), psi_history5[int((x_min+40)/dt):int((x_max+40)/dt),0,0], 'r', linewidth=1, label='Sensed velocity, $\psi_1$; $\pi_w = exp($'+str(mu_gamma_w_history5[1,0])+')')
    plt.plot(np.arange(x_min, x_max, dt), psi_history6[int((x_min+40)/dt):int((x_max+40)/dt),0,0], 'g', linewidth=1, label='Sensed velocity, $\psi_1$; $\pi_w = exp($'+str(mu_gamma_w_history6[1,0])+')')
    
    plt.title('Car velocity')
    plt.xlabel('Time ($s$)')
    plt.ylabel('Velocity ($km/h$)')
    plt.legend(loc=1)
    plt.text(x_max+1.5, eta_x_history[-2,0,0], "$\eta_x$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
    plt.ylim((0., 20.))
    plt.title('Car velocity - Set-point change')
    plt.legend(loc=4)
    plt.annotate("new target",
        xy=((x_min+x_max)/2, 9.), xycoords='data',
        xytext=((x_min+x_max)/2-2.15, 6.), textcoords='data',
        arrowprops=dict(arrowstyle="simple",
                        connectionstyle="arc3"))
    plt.savefig("figures/activeInferencePIDSetPoint.pdf")


if simulation == 0 or simulation == 1:
    plt.figure(figsize=(9, 6))
    plt.xlim((0., T))
    plt.plot(np.arange(0, T-dt, dt), psi_history[:-1,0,1], 'b', linewidth=1, label='Sensed acceleration, $\psi\'$')
    plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,1], 'r', linewidth=1, label='Expec. of acceleration, $\mu_x\'$')
    #plt.plot(np.arange(0, T-dt, dt), eta_x_history[:-1,0,1], 'g', label='Desired acceleration, $\eta_x\'$')
    plt.title('Car acceleration')
    plt.xlabel('Time ($s$)')
    plt.ylabel('Acceleration ($km/h^2$)')
    plt.legend(loc=1)
    if simulation == 0:
        plt.text(T+22, eta_x_history[-2,0,1], "$\eta'_x$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
        plt.ylim((-70., 100.))
        plt.savefig("figures/activeInferencePID_b.pdf")
    elif simulation == 1:
        plt.text(T+22, eta_x_history[-2,0,1], "$\eta'_x$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
        plt.axvline(x=switch_condition_time, linewidth=3, color='k', linestyle='-.')
        plt.ylim((-70., 100.))
        plt.savefig("figures/activeInferencePIDTuning_b.pdf")


if simulation == 0 or simulation == 1:
    plt.figure(figsize=(9, 6))
    plt.xlim((0., T))
    plt.title('Motor output')
    plt.plot(np.arange(0, T-dt, dt), a_history[:-1,0], linewidth=1, label='Action, a')
    
    plt.xlabel('Time ($s$)')
    plt.ylabel('Acceleration ($km/h^2$)')
    if simulation == 0:
        plt.plot(np.arange(0, T-dt, dt), v_history[:-1,0,0], 'k', linewidth=1, label='Ext. input, v')
        plt.legend(loc=1)
        plt.ylim((-5., 5.))
        plt.savefig("figures/activeInferencePID_c.pdf")
    elif simulation == 1:
        plt.axvline(x=switch_condition_time, linewidth=3, color='k', linestyle='-.')
        plt.plot(np.arange(0, T-dt, dt), v_history[:-1,0,0], 'k', label='Ext. input, v')
        plt.ylim((-5., 5.))
        plt.legend(loc=1)
        plt.savefig("figures/activeInferencePIDTuning_c.pdf")


if simulation == 1:
    plt.figure(figsize=(9, 6))
    plt.xlim((0., T))
    plt.ylim((-6., 6.))
    plt.title('Log-(sensory) precisions (= log-PI gains)')
    plt.plot(np.arange(0, T-dt, dt), mu_gamma_z_history[:-1, 0], 'r', label='Expec. of log-precision, $\mu_{\gamma_z}$')
    plt.plot(np.arange(0, T-dt, dt), gamma_z_history[:-1, 0], 'b')
    
    plt.plot(np.arange(0, T-dt, dt), mu_gamma_z_history[:-1, 1], color='orange', label='Expec. of log-precision, $\mu_{\gamma_{z\'}}$')
    plt.text(T+20, gamma_z_history[-2,0], "$\gamma_{z}$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
    
    plt.xlabel('Time ($s$)')
    plt.legend(loc=4)
    plt.axvline(x=switch_condition_time, linewidth=3, color='k', linestyle='-.')
    mid = int(T/2)
    plt.axvline(x=mid, linewidth=3, color='k', linestyle='-')
    plt.text(mid-60, gamma_z_history[-2,0]-1, "Adaptation", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    plt.text(mid+75, gamma_z_history[-2,0]-1, "Control", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    
    plt.savefig("figures/activeInferencePIDTuning_d.pdf")


if simulation == 3:
    plt.figure(figsize=(10, 6))
    plt.boxplot([variance_after[:, 0], variance_after[:, 1]])
    plt.xticks([1, 2], ['Adaptation interrupted', 'Continual adaptation'])
    plt.ylabel('Variance (a.u.)')
    plt.savefig("figures/activeInferencePIDVaryingNoise.pdf")


if simulation == 4:
    plt.figure(figsize=(10, 6))
    plt.boxplot([iaeNoAdaptation, iaeAdaptation])
    plt.xticks([1, 2], ['No adaptation', 'Adaptation'])
    plt.ylabel('IAE (a.u.)')
    plt.savefig("figures/activeInferencePIDiae.pdf")


