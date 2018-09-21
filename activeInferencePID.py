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

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dt = .05

obs_states = 1
hidden_states = 1                                           # x, in Friston's work
hidden_causes = 1                                           # v, in Friston's work
states = obs_states + hidden_states
temp_orders_states = 3                                      # generalised coordinates for hidden states x, but only using n-1
temp_orders_causes = 3                                      # generalised coordinates for hidden causes v (or \eta in biorxiv manuscript), but only using n-1

# 0: PID control as active inference
# 1: PID tuning based on means of observation errors
# 2: PID tuning based on means and variances of observation errors 
# 3: load disturbance response affected by pi_z + set point response affected by pi_w
# 4: set point response affected by pi_w
# 5: measurement error response affected by p_pi_z
# 6: model uncertainty response affected by p_pi_w
# 7: PID tuning based on means of dynamic errors
# 8: PID tuning based on means and variances of dynamic errors

### FUNCTIONS ###

def sigmoid(x):
    return np.tanh(x)
    return 1 / (1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

## cruise control ##

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
def g(x, v):
    return x

def f(x, v, a):
    return (force_drive(x, v + a) - force_disturbance(x, theta)) / m

# generative model
def g_gm(x, v):
    return g(x, v)

def f_gm(x, v):
    # no action in generative model, a = 0.0
    return f(x, v, 0.0)

def getObservation(x, v, a, w):
    x[:, 1:] = f(x[:, :-1], v, a)# + w
    x[:, 0] += dt * x[:, 1]
    return g(x[:, :-1], v)

def F(psi, mu_x, mu_gamma_z, mu_pi_w):
    return .5 * (np.sum(np.exp(mu_gamma_z) * (psi - mu_x[0, :-1])**2) +
                 np.sum(mu_pi_w * (mu_x[0, 1:] + alpha * (mu_x[0, :-1] - eta_x))**2) -
                 np.log(np.prod(np.exp(mu_gamma_z)) * np.prod(mu_pi_w)))


def pidControl(simulation, T, dt, mu_gamma_z_input, mu_gamma_w_input):
    iterations = int(T / dt)
    alpha = 100000.                                             # drift in Generative Model
    gamma = 1                                                   # drift in OU process    
    
    ### cruise control problem from Astrom and Murray (2010), pp 65-69
    
    # environment parameters
    ga = 9.81                                                   # gravitational acceleration
    theta = 4.                                                  # hill angle
    C_r = .01                                                   # rolling friction coefficient
    C_d = .32                                                   # drag coefficient
    rho = 1.3                                               # air density
    A = 2.4                                                     # frontal area agent
    
    # car's parameters
    #m = 1000                                                    # car mass (book example)
    m = 100                                                    # car mass
    T_m = 190                                                   # maximum torque
    omega_m = 420                                               # engine speed to reach T_m
    r_g = 12                                                # = gear ration/wheel radius,
                                                                # a1 = 40, a2 = 25, a3 = 16, a4 = 12, a5 = 10
    beta = .4
    
    x = np.zeros((hidden_states, temp_orders_states))           # position
    v = np.zeros((hidden_causes, temp_orders_states - 1))
    y = np.zeros((obs_states, temp_orders_states - 1))
    eta_x = np.zeros((hidden_causes, temp_orders_states - 1))
    eta_gamma_z = np.zeros((obs_states, temp_orders_states - 1))
    eta_gamma_w = np.zeros((hidden_states, temp_orders_states - 1))
    
    if simulation == 3:
        desired_velocity = 13.
    else:
        desired_velocity = 10.
    eta_x[0, 0] = desired_velocity
    # hyperpriors
    eta_gamma_z[0, 0] = 3.
    eta_gamma_z[0, 1] = 1.
    
    eta_gamma_w[0, 0] = - 18.
    eta_gamma_w[0, 1] = - 18.
    
    
    
    
    ### free energy variables
    a = np.zeros((obs_states, temp_orders_states-1))
    control = np.zeros((obs_states, temp_orders_states-1))
    phi_z = np.zeros((obs_states, temp_orders_states-1))
    phi_w = np.zeros((hidden_states, temp_orders_states-1))
    
    mu_x = 0.0001*np.random.randn(hidden_states, temp_orders_states)
    #mu_x = np.zeros((hidden_states, temp_orders_states))
    mu_v = np.random.randn(hidden_causes, temp_orders_states)
    mu_v = np.zeros((hidden_causes, temp_orders_states))
    
    # minimisation variables and parameters
    dFdmu_x = np.zeros((hidden_states, temp_orders_states))
    dFdmu_v = np.zeros((hidden_causes, temp_orders_states))
    dFdmu_gamma_z = np.zeros((hidden_causes, temp_orders_states))
    Dmu_x = np.zeros((hidden_states, temp_orders_states))
    Dmu_v = np.zeros((hidden_causes, temp_orders_states))
    k_mu_x = 1                                                  # learning rate perception
    k_a = 1                                                     # learning rate action
    k_mu_gamma_z = 1                                            # learning rate attention
    k_mu_gamma_w = 1                                            # learning rate attention
    kappa_z = 5                                                # damping on precisions minimisation
    kappa_w = 10                                                # damping on precisions minimisation
    
    # noise on sensory input (world - generative process)
    gamma_z = 0. * np.ones((obs_states, temp_orders_states))    # log-precisions
    gamma_z[:,1] = gamma_z[:,0] - np.log(2 * gamma)
    gamma_z[:,2] = gamma_z[:,1] - np.log(2 * gamma)
    
    gamma_z = 5. * np.ones((obs_states, temp_orders_states))    # log-precisions, uncorrelated noise
    pi_z = np.exp(gamma_z) * np.ones((obs_states, temp_orders_states))
    sigma_z = 1 / (np.sqrt(pi_z))
    z = np.zeros((iterations, obs_states, temp_orders_states))
    for i in range(obs_states):
        for j in range(temp_orders_states):
            z[:, i, j] = sigma_z[i, j] * np.random.randn(1, iterations)
    
    # noise on motion of hidden states (world - generative process)
    gamma_w = 32 * np.ones((hidden_states, temp_orders_states))    # log-precisions
    gamma_w[:,1] = gamma_w[:,0] - np.log(2 * gamma)
    gamma_w[:,2] = gamma_w[:,1] - np.log(2 * gamma)
    pi_w = np.exp(gamma_w) * np.ones((hidden_states, temp_orders_states))
    sigma_w = 1 / (np.sqrt(pi_w))
    w = np.zeros((iterations, hidden_states, temp_orders_states))
    for i in range(hidden_states):
        for j in range(temp_orders_states - 1):
            w[:, i, j] = sigma_w[i, j] * np.random.randn(1, iterations)
    
    
    # agent's estimates of the noise (agent - generative model)
    if simulation == 0 or simulation == 4:
        mu_gamma_z = - 3.0 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
    #    mu_gamma_z = 1.0 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions for sim 3 (sim 0 is run before sim 3 for plotting)
    elif simulation == 3:
        mu_gamma_z = 1. * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
        mu_gamma_z = - 3.0 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
    else:
        mu_gamma_z = - 3.0 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
    
    mu_gamma_z = mu_gamma_z_input * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
    mu_gamma_z[0, 1] = mu_gamma_z[0, 0] - np.log(2 * gamma)
    #mu_gamma_z[0, 0] = 3.708050201
    #mu_gamma_z[0, 1] = -.6931471806
    #mu_gamma_z[0, 1] = 1
    #mu_gamma_z[0, 1] = -0.7
    mu_pi_z = np.exp(mu_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
    if simulation == 3:
        mu_gamma_w = - 24 * np.ones((hidden_states, temp_orders_states - 1))   # log-precision
        mu_gamma_w = - 20 * np.ones((hidden_states, temp_orders_states - 1))   # log-precision
    else:
        mu_gamma_w = - 20 * np.ones((hidden_states, temp_orders_states - 1))   # log-precision
    
    mu_gamma_w = mu_gamma_w_input * np.ones((hidden_states, temp_orders_states - 1))   # log-precision for sim 3 (sim 0 is run before sim 3 for plotting)
    
    #if simulation == 4:
    #    mu_gamma_w = - 24 * np.ones((hidden_states, temp_orders_states - 1))   # log-precision
    #    mu_gamma_w = - 19 * np.ones((hidden_states, temp_orders_states - 1))   # log-precision
    #else:
    #    mu_gamma_w = - 22 * np.ones((hidden_states, temp_orders_states - 1))   # log-precision
    mu_gamma_w[0, 1] = mu_gamma_w[0, 0] - np.log(2)
    mu_pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
    
    # hyperpriors' precisions
    mu_gamma_gamma_z = - large_value * np.ones((obs_states, temp_orders_states - 1))
    mu_p_gamma_z = np.exp(mu_gamma_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
    
    mu_gamma_gamma_w = - large_value * np.ones((hidden_states, temp_orders_states - 1))
    mu_p_gamma_w = np.exp(mu_gamma_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
    
    
    # history
    x_history = np.zeros((iterations, hidden_states, temp_orders_states))
    y_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
    v_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))
    psi_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
    mu_x_history = np.zeros((iterations, hidden_states, temp_orders_states))
    eta_x_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))
    eta_gamma_z_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
    eta_gamma_w_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))
    a_history = np.zeros((iterations, temp_orders_states - 1))
    control_history = np.zeros((iterations, temp_orders_states - 1))
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
    
    FE_history = np.zeros((iterations,))
    
    gamma_z_history = np.zeros((iterations, temp_orders_states))
    gamma_w_history = np.zeros((iterations, temp_orders_states))
    
    
    if simulation == 5:
        eta_gamma_z = gamma_z
    
    u = 0
    for i in range(iterations - 1):
        print(i)
        
        # re-encode precisions after hyperparameters update
        mu_pi_z = np.exp(mu_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
        mu_pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
        mu_p_gamma_z = np.exp(mu_gamma_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
        mu_p_gamma_w = np.exp(mu_gamma_gamma_w) * np.ones((obs_states, temp_orders_states - 1))
        
        # include an external disturbance to test integral term
        if (simulation == 0):
            if (i > iterations/2):
                v[0,0] = 3.0
        
    #    if simulation == 1:
    #        if (i > switch_condition_time/2/dt):
    #            v[0,0] = 1.0
    #        else:
    #            v[0,0] = 0.0
        
        if (simulation == 3):
            if (i > iterations/2):
                eta_x[0, 0] = desired_velocity - 3
            else:
                eta_x[0, 0] = desired_velocity
        
        if (simulation == 4):
            if (i > iterations/3) and (i < 2*iterations/3):
                eta_x[0, 0] = 5.
            else:            
                eta_x[0, 0] = 10.
        
        
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
        
        y = getObservation(x, v, control, w[i, 0, :-1])
        psi = y + z[i, 0, :-1]
        
        # prediction errors (only used for plotting)
    #    eps_z = y - g_gm(mu_x[:, :-1], mu_v)
    #    xi_z = mu_pi_z * eps_z
    #    eps_w = mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1])
    #    xi_w = mu_pi_w * eps_w
        
        ### minimise free energy ###
        # perception
        Dmu_x[0, :-1] = mu_x[0, 1:]
    #    dFdmu_x[0, :-1] = np.array([mu_pi_z * - (psi - mu_x[0, :-1]) + mu_pi_w * alpha * (mu_x[0, 1:] + alpha * (mu_x[0, :-1] - eta_x))])
#        dFdmu_x[0, :-1] = np.array([mu_pi_z * - (y + z[i, 0, :-1]/np.sqrt(dt) - mu_x[0, :-1]) + mu_pi_w * alpha * (mu_x[0, 1:] + alpha * (mu_x[0, :-1] - eta_x))])
        dFdmu_x[0, :-1] = np.array([mu_pi_w * alpha * (mu_x[0, 1:] + alpha * (mu_x[0, :-1] - eta_x))])
        
        # action
    #    dFdy = mu_pi_z * (psi - mu_x[0, :-1])
        dFdy = mu_pi_z * (y + z[i, 0, :-1]/np.sqrt(dt) - mu_x[0, :-1])
        dyda = np.ones((obs_states, temp_orders_states-1))
        dFda = np.zeros((obs_states, temp_orders_states-1))
        dFda[0, 0] = np.sum(dFdy * dyda)
        
        # attention
    #    dFdmu_gamma_z = .5 * (mu_pi_z * (psi - mu_x[0, :-1])**2 - 1) + mu_p_gamma_z * (mu_gamma_z - eta_gamma_z)
    #    dFdmu_gamma_z = .5 * (mu_pi_z * (y + z[i, 0, :-1]/np.sqrt(dt) - mu_x[0, :-1])**2 - 1) + mu_p_gamma_z * (mu_gamma_z - eta_gamma_z)
        dFdmu_gamma_z = .5 * (mu_pi_z * (y**2 + z[i, 0, :-1]**2 + mu_x[0, :-1]**2 + 2*y*z[i, 0, :-1]/np.sqrt(dt) - 2*mu_x[0, :-1]*z[i, 0, :-1]/np.sqrt(dt) - 2*y*mu_x[0, :-1]) - 1) + mu_p_gamma_z * (mu_gamma_z - eta_gamma_z)
    #    dFdmu_gamma_z = .5 * (mu_pi_z * (y**2 + z[i, 0, :-1]**2 + eta_x**2 + 2*y*z[i, 0, :-1]/np.sqrt(dt) - 2*eta_x*z[i, 0, :-1]/np.sqrt(dt) - 2*y*eta_x) - 1) + mu_p_gamma_z * (mu_gamma_z - eta_gamma_z)
        dFdmu_gamma_w = .5 * (mu_pi_w * (mu_x[0, :-1] + alpha * (mu_x[0, :-1] - eta_x))**2 - 1) + mu_p_gamma_w * (mu_gamma_w - eta_gamma_w)
        
        
        # update equations
        mu_x += dt * (Dmu_x - k_mu_x * dFdmu_x)
        a += dt * - k_a * dFda
        
        control = np.array([sigmoid(a[0,0]), 0])
        control = sigmoid(a)
        control = a
        
        phi_z += dt * (- dFdmu_gamma_z - kappa_z * phi_z)
        phi_w += dt * (- dFdmu_gamma_w - kappa_w * phi_w)
            
        if simulation == 1 and (i > switch_condition_time/dt):
    #        mu_gamma_z += dt * k_mu_gamma_z * phi_z
            if (i > iterations/2):
                v[0,0] = 3.0
            else:
                mu_gamma_z += dt * k_mu_gamma_z * phi_z
                v[0,0] = 0.0
    #        mu_gamma_w += dt * k_mu_gamma_w * phi_w
    #        if i > 3*switch_condition_time/4/dt:
    #            kappa_z = 20
    ##            kappa_z = 50 * np.tanh((i - iterations/4 - iterations/8)/iterations*1)+1
    #            kappa_z_history[i] = kappa_z
        
        if simulation == 2 and (i > iterations/3):
            phi_z += dt * (- dFdmu_gamma_z - kappa_z * phi_z)
            phi_w += dt * (- dFdmu_gamma_w - kappa_w * phi_w)
            mu_gamma_z += dt * k_mu_gamma_z * phi_z
    #        if i > iterations/3 + iterations/16:
    #            kappa_z = 50 * np.tanh((i - iterations/3 - iterations/16)/(50*T))+10
    #            kappa_z_history[i] = kappa_z
            if i > 2*iterations/3:
                eta_gamma_z = gamma_z[0,:-1]
                eta_gamma_z = np.array([-4.0, -4.0 - np.log(2 * gamma)])
                mu_gamma_gamma_z = 2 * np.ones((obs_states, temp_orders_states - 1))
        
        if simulation == 5:
            mu_gamma_z += dt * k_mu_gamma_z * phi_z
            if i > iterations/8:
                kappa_z = 50 * np.tanh((i - iterations/8)/(50*T))+10
                kappa_z_history[i] = kappa_z
                
                if i == int(iterations/2):
                    mu_gamma_gamma_z = 6.5 * np.ones((obs_states, temp_orders_states - 1))         # uncomment or comment to get a prior or just follow the changing measurement noise
                    
                    gamma_z = -4 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
                    gamma_z[:,1] = gamma_z[:,0] - np.log(2 * gamma)
                    pi_z = np.exp(gamma_z) * np.ones((obs_states, temp_orders_states - 1))
                    sigma_z = 1 / (np.sqrt(pi_z))
                    z = np.zeros((iterations, obs_states, temp_orders_states - 1))
                    for j in range(obs_states):
                        for k in range(temp_orders_states - 1):
                            z[:, j, k] = sigma_z[j, k] * np.random.randn(1, iterations)
    
        if simulation == 6:
            mu_gamma_w += dt * k_mu_gamma_w * psi
            if i > iterations/8:
                kappa_w = 50 * np.tanh((i - iterations/8)/(50*T))+10
                kappa_w_history[i] = kappa_w
                
                if i == int(iterations/2):
                    mu_gamma_gamma_w = 4 * np.ones((obs_states, temp_orders_states - 1))         # uncomment or comment to get a prior or just follow the changing model uncertainty
                    
                    gamma_w = - 4 * np.ones((hidden_states, temp_orders_states - 1))    # log-precisions
                    gamma_w[:,1] = gamma_w[:,0] - np.log(2 * gamma)
                    pi_w = np.exp(gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
                    sigma_w = 1 / (np.sqrt(pi_w))
                    w = np.zeros((iterations, hidden_states, temp_orders_states - 1))
                    for j in range(hidden_states):
                        for k in range(temp_orders_states - 1):
                            w[:, j, k] = sigma_w[j, k] * np.random.randn(1, iterations)
                            
        if simulation == 7 and (i > iterations/2):
            mu_gamma_w += dt * k_mu_gamma_w * psi
            if i > iterations/2 + iterations/8:
                kappa_w = 50 * np.tanh((i - iterations/2 - iterations/8)/(50*T))+10
                kappa_w_history[i] = kappa_w
                
        if simulation == 8 and (i > iterations/3):
            mu_gamma_w += dt * k_mu_gamma_w * psi
            kappa_w = 50 * np.tanh((i - iterations/3)/(50*T))+10
            kappa_w_history[i] = kappa_w
            if i > 2*iterations/3:
                 mu_gamma_gamma_w = 5 * np.ones((obs_states, temp_orders_states - 1))
            
        # save history

        psi_history[i, :] = psi
        mu_x_history[i, :, :] = mu_x
        eta_x_history[i] = eta_x
            
        eta_gamma_z_history[i] = eta_gamma_z
        eta_gamma_w_history[i] = eta_gamma_w
        a_history[i] = a
        control_history[i] = control
        v_history[i] = v
        mu_gamma_z_history[i] = mu_gamma_z
        mu_gamma_w_history[i] = mu_gamma_w
        
        gamma_z_history[i] = gamma_z
        gamma_w_history[i] = gamma_w
        
    #    xi_z_history[i, :, :] = xi_z
    #    xi_w_history[i, :, :] = xi_w
        FE_history[i] = F(psi, mu_x, mu_gamma_z, mu_pi_w)
        
    #    dFdmu_gamma_z_history[i] = dFdmu_gamma_z
    #    dFdmu_gamma_w_history[i] = dFdmu_gamma_z
    #    mu_pi_z_history[i] = mu_pi_z
    #    mu_pi_w_history[i] = mu_pi_w
    
    return psi_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history

simulation = 0

if simulation == 0:
    T = 100
elif simulation == 1:
    T = 200
    switch_condition_time = T/10
elif simulation == 2:
    T = 1500
elif simulation == 3:
    T = 100
elif simulation == 4:
    T = 50
elif simulation == 5:
    T = 1000
elif simulation == 6:
    T = 1500
elif simulation == 7:
    T = 1000
elif simulation == 8:
    T = 1000

#psi_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, 1.,-25)
#psi_history2, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, 1.,-22)
#psi_history3, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, 1.,-20)
#
#
#simulation = 3
#
#psi_history5, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, 1.,-25)
#psi_history6, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, 1.,-22)
#psi_history7, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, 1.,-20)
    
psi_history, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, 1.,-20)
psi_history2, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, -1.,-20)
psi_history3, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, -3.,-20)


simulation = 3

psi_history5, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, 1.,-20)
psi_history6, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, -1.,-20)
psi_history7, mu_x_history, eta_x_history, eta_gamma_z_history, eta_gamma_w_history, control_history, mu_gamma_z_history, mu_gamma_w_history = pidControl(simulation, T, dt, -3.,-20)




fig1 = plt.figure(figsize=(9, 6))
plt.xlim((0., T))
#plt.plot(np.arange(0, T-dt, dt), psi_history[:-1,0,0], 'b', linewidth=1, label='Measured velocity, $\psi$')
#plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,0], 'r', linewidth=1, label='Expected velocity, $\mu_x$')
#plt.plot(np.arange(0, T-dt, dt), eta_x_history[:-1,0,0], 'k--', linewidth=1, label='Desired velocity, $\eta_x$')

plt.plot(np.arange(0, T-dt, dt), psi_history[:-1,0,0], linewidth=1, label='Measured velocity, $\psi$')
plt.plot(np.arange(0, T-dt, dt), psi_history2[:-1,0,0], linewidth=1, label='Measured velocity, $\psi$')
plt.plot(np.arange(0, T-dt, dt), psi_history3[:-1,0,0], linewidth=1, label='Measured velocity, $\psi$')
#plt.plot(np.arange(0, T-dt, dt), psi_history4[:-1,0,0], linewidth=1, label='Measured velocity, $\psi$')

plt.title('Hidden state, x (car velocity)')
plt.xlabel('Time ($s$)')
plt.ylabel('Velocity ($km/h$)')
plt.legend(loc=1)
if simulation == 0:
    plt.text(T+7, eta_x_history[-2,0,0], "$\eta_x$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
#    plt.ylim((-10., 50.))
    plt.savefig("figures/activeInferencePID_a.pdf")
elif simulation == 1:
    plt.axvline(x=switch_condition_time, linewidth=3, color='k', linestyle='-.')
    plt.ylim((-10., 50.))
    plt.savefig("figures/activeInferencePIDTuning_a.pdf")
elif simulation == 2:
    plt.savefig("figures/activeInferencePIDTuningHyperPriors_a.pdf")
elif simulation == 3:
#    plt.ylim((0., 20.))
#    plt.yticks(np.arange(0, 20, 5))
#    plt.text(0-7, eta_x_history[-2,0,0]+3, "$\eta_x^{[1]}$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="rarrow", fc="w", ec="0.5", alpha=0.9))
#    plt.text(T+7, eta_x_history[-2,0,0], "$\eta_x^{[1,2]}$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
#    plt.text(int(T/2), psi_history[int(iterations/2+1),0,0]+1, "$v$", size=20, rotation=90., ha="center", va="center", bbox=dict(boxstyle="rarrow", fc="w", ec="0.5", alpha=0.9))
#    plt.plot(np.arange(0, T-dt, dt), psi_history2[:-1,0,0], 'c', linewidth=1, label='Measured velocity, $\psi$')
#    plt.plot(np.arange(0, T-dt, dt), mu_x_history2[:-1,0,0], color='orange', linewidth=1, label='Expected velocity, $\mu_x$')
    plt.savefig("figures/activeInferencePIDLoad_a.pdf")
elif simulation == 4:
    plt.savefig("figures/activeInferencePIDSetPoint_a.pdf")
elif simulation == 5:
    plt.ylim((-100., 100.))
    plt.savefig("figures/activeInferencePIDMeasurementNoise_a.pdf")
elif simulation == 6:
    plt.savefig("figures/activeInferencePIDModelUncertainty_a.pdf")

plt.figure(figsize=(9, 6))
plt.xlim((0., T))
#plt.plot(np.arange(0, T-dt, dt), psi_history[:-1,0,0], 'b', linewidth=1, label='Measured velocity, $\psi$')
#plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,0], 'r', linewidth=1, label='Expected velocity, $\mu_x$')
#plt.plot(np.arange(0, T-dt, dt), eta_x_history[:-1,0,0], 'k--', linewidth=1, label='Desired velocity, $\eta_x$')

plt.plot(np.arange(0, T-dt, dt), psi_history5[:-1,0,0], linewidth=1, label='Measured velocity, $\psi$')
plt.plot(np.arange(0, T-dt, dt), psi_history6[:-1,0,0], linewidth=1, label='Measured velocity, $\psi$')
plt.plot(np.arange(0, T-dt, dt), psi_history7[:-1,0,0], linewidth=1, label='Measured velocity, $\psi$')
#plt.plot(np.arange(0, T-dt, dt), psi_history8[:-1,0,0], linewidth=1, label='Measured velocity, $\psi$')

plt.title('Hidden state, x (car velocity)')
plt.xlabel('Time ($s$)')
plt.ylabel('Velocity ($km/h$)')
plt.legend(loc=1)
if simulation == 0:
    plt.text(T+7, eta_x_history[-2,0,0], "$\eta_x$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
#    plt.ylim((-10., 50.))
    plt.savefig("figures/activeInferencePID_a.pdf")
elif simulation == 1:
    plt.axvline(x=switch_condition_time, linewidth=3, color='k', linestyle='-.')
    plt.ylim((-10., 50.))
    plt.savefig("figures/activeInferencePIDTuning_a.pdf")
elif simulation == 2:
    plt.savefig("figures/activeInferencePIDTuningHyperPriors_a.pdf")
elif simulation == 3:
#    plt.ylim((0., 20.))
#    plt.yticks(np.arange(0, 20, 5))
#    plt.text(0-7, eta_x_history[-2,0,0]+3, "$\eta_x^{[1]}$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="rarrow", fc="w", ec="0.5", alpha=0.9))
#    plt.text(T+7, eta_x_history[-2,0,0], "$\eta_x^{[1,2]}$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
#    plt.text(int(T/2), psi_history[int(iterations/2+1),0,0]+1, "$v$", size=20, rotation=90., ha="center", va="center", bbox=dict(boxstyle="rarrow", fc="w", ec="0.5", alpha=0.9))
#    plt.plot(np.arange(0, T-dt, dt), psi_history2[:-1,0,0], 'c', linewidth=1, label='Measured velocity, $\psi$')
#    plt.plot(np.arange(0, T-dt, dt), mu_x_history2[:-1,0,0], color='orange', linewidth=1, label='Expected velocity, $\mu_x$')
    plt.savefig("figures/activeInferencePIDLoad_a.pdf")
elif simulation == 4:
    plt.savefig("figures/activeInferencePIDSetPoint_a.pdf")
elif simulation == 5:
    plt.ylim((-100., 100.))
    plt.savefig("figures/activeInferencePIDMeasurementNoise_a.pdf")
elif simulation == 6:
    plt.savefig("figures/activeInferencePIDModelUncertainty_a.pdf")



























plt.figure(figsize=(9, 6))
plt.xlim((0., T))
plt.plot(np.arange(0, T-dt, dt), psi_history[:-1,0,1], 'b', linewidth=1, label='Measured acceleration, $\psi\'$')
plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,1], 'r', linewidth=1, label='Estimated acceleration, $\mu_x\'$')
#plt.plot(np.arange(0, T-dt, dt), eta_x_history[:-1,0,1], 'g', label='Desired acceleration, $\eta_x\'$')
plt.title('Hidden state, x\' (car acceleration)')
plt.xlabel('Time ($s$)')
plt.ylabel('Acceleration ($km/h^2$)')
plt.legend(loc=1)
if simulation == 0:
    plt.text(T+8, eta_x_history[-2,0,1], "$\eta'_x$", size=20, rotation=0., ha="center", va="center", bbox=dict(boxstyle="larrow", fc="w", ec="0.5", alpha=0.9))
    plt.ylim((-70., 70.))
    plt.savefig("figures/activeInferencePID_b.pdf")
elif simulation == 1:
    plt.axvline(x=switch_condition_time, linewidth=3, color='k', linestyle='-.')
    plt.ylim((-70., 70.))
    plt.savefig("figures/activeInferencePIDTuning_b.pdf")
elif simulation == 2:
    plt.savefig("figures/activeInferencePIDTuningHyperPriors_b.pdf")
elif simulation == 3:
    plt.plot(np.arange(0, T-dt, dt), psi_history2[:-1,0,1], 'c', linewidth=1, label='Measured velocity, $\psi$')
    plt.plot(np.arange(0, T-dt, dt), mu_x_history2[:-1,0,1], color='orange', linewidth=1, label='Expected velocity, $\mu_x$')
    plt.savefig("figures/activeInferencePIDLoad_b.pdf")
elif simulation == 4:
    plt.savefig("figures/activeInferencePIDSetPoint_b.pdf")
elif simulation == 5:
    plt.ylim((-150., 150.))
    plt.savefig("figures/activeInferencePIDMeasurementNoise_b.pdf")
elif simulation == 6:
    plt.savefig("figures/activeInferencePIDModelUncertainty_b.pdf")

#plt.figure(figsize=(9, 6))
#plt.xlim((0., T))
#plt.plot(np.arange(0, T-dt, dt), v_history[:-1,0,0], 'k')
#plt.title('Hidden input, v (external disturbance)')
#plt.xlabel('Time ($s$)')
#plt.ylabel('Acceleration ($km/h^2$)')
#if simulation == 0:
#    plt.savefig("figures/activeInferencePID_c.pdf")
#elif simulation == 1:
#    plt.savefig("figures/activeInferencePIDTuning_c.pdf")
#elif simulation == 2:
#    plt.savefig("figures/activeInferencePIDTuningHyperPriors_c.pdf")
#elif simulation == 3:
#    plt.savefig("figures/activeInferencePIDLoad_c.pdf")
#elif simulation == 4:
#    plt.savefig("figures/activeInferencePIDSetPoint_c.pdf")
#elif simulation == 5:
#    plt.savefig("figures/activeInferencePIDMeasurementNoise_c.pdf")
#elif simulation == 6:
#    plt.savefig("figures/activeInferencePIDModelUncertainty_c.pdf")

plt.figure(figsize=(9, 6))
plt.xlim((0., T))
plt.title('Motor output, a (control)')
plt.plot(np.arange(0, T-dt, dt), control_history[:-1,0], linewidth=1, label='Action, a')
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
elif simulation == 2:
    plt.savefig("figures/activeInferencePIDTuningHyperPriors_d.pdf")
elif simulation == 3:
    plt.savefig("figures/activeInferencePIDLoad_d.pdf")
elif simulation == 4:
    plt.ylim((-1., 0.5))
    plt.savefig("figures/activeInferencePIDSetPoint_d.pdf")
elif simulation == 5:
    plt.savefig("figures/activeInferencePIDMeasurementNoise_d.pdf")
elif simulation == 6:
    plt.savefig("figures/activeInferencePIDModelUncertainty_d.pdf")

if simulation == 0:
    print(np.var(psi_history[int(T/2/dt):-1,0,0]))
    print(np.var(psi_history[:,0,0]))
elif simulation == 1:
    print(np.var(psi_history[int(T/4/dt):int(T/2/dt),0,0]))
    print(np.var(psi_history[int(3*T/4/dt+1):int(T/dt),0,0]))
#    print(np.var(psi_history[:,0,0]))
elif simulation == 5 or simulation == 6 or simulation == 7:
    print(np.var(psi_history[int(T/4/dt):int(T/2/dt),0,0]))
    print(np.var(psi_history[int(T/2/dt+1):int(T/dt),0,0]))
elif simulation == 2 or simulation == 8:
    print(np.var(psi_history[int(T/6/dt):int(T/3/dt),0,0]))
    print(np.var(psi_history[int(T/2/dt):int(2*T/3/dt),0,0]))
    print(np.var(psi_history[int(5*T/6/dt):int(T/dt),0,0]))
        
if simulation == 1 or simulation == 2 or simulation == 5:
    plt.figure(figsize=(9, 6))
    plt.xlim((0., T))
#    plt.ylim((-7., 7.))
#    plt.title('Log-precision, $\gamma_z$ (= integral gain, $k_i$)')
    plt.title('Log-precisions, $\gamma_z - \gamma_{z\'}$ (= I-P gains, $k_i - k_p$)')
    plt.plot(np.arange(0, T-dt, dt), mu_gamma_z_history[:-1, 0], 'r', label='Estimated log-precision, $\mu_{\gamma_z}$')
    plt.plot(np.arange(0, T-dt, dt), gamma_z_history[:-1, 0], 'b', label='Real log-precision, $\gamma_z$')
    
    plt.plot(np.arange(0, T-dt, dt), mu_gamma_z_history[:-1, 1], color='orange', label='Expected log-precision, $\mu_{\gamma_{z\'}}$')
    plt.plot(np.arange(0, T-dt, dt), gamma_z_history[:-1, 1], 'c', label='Real log-precision, $\gamma_{z\'}$')
#    plt.plot(np.arange(0, T-dt, dt), eta_gamma_z_history[:-1, 0, 0], 'g', label='Prior precision, $\eta_{\gamma_z}$')
#    plt.axhline(y=-np.log(np.var(psi_history[int(T/(4*dt)):-1,0,0])), xmin=0.0, xmax=T, color='k', label='Measured precision')
    plt.xlabel('Time ($s$)')
    plt.legend(loc=4)
    if simulation == 1:
        plt.axvline(x=switch_condition_time, linewidth=3, color='k', linestyle='-.')
        plt.savefig("figures/activeInferencePIDTuning_d.pdf")
    if simulation == 2:
        plt.savefig("figures/activeInferencePIDTuningHyperPriors_e.pdf")
    if simulation == 5:
        plt.savefig("figures/activeInferencePIDMeasurementNoise_e.pdf")
    
    
#    plt.figure(figsize=(9, 6))
#    plt.xlim((0., T))
##    plt.ylim((-5., 7.))
#    plt.title('Log-precision, $\gamma_{z\'}$ (= proportional gain, $k_p$)')
#    plt.plot(np.arange(0, T-dt, dt), mu_gamma_z_history[:-1, 1], 'r', label='Expected log-precision, $\mu_{\gamma_{z\'}}$')
#    plt.plot(np.arange(0, T-dt, dt), gamma_z_history[:-1, 1], 'b', label='Real log-precision, $\gamma_{z\'}$')
##    plt.plot(np.arange(0, T-dt, dt), eta_gamma_z_history[:-1, 0, 1], 'g', label='Prior precision, $\eta_{\gamma_{z\'}}$')
##    plt.axhline(y=-np.log(np.var(psi_history[int(T/(4*dt)):-1,0,1])), xmin=0.0, xmax=T, color='k', label='Measured precision')
#    plt.xlabel('Time ($s$)')
#    plt.legend(loc=1)
#    if simulation == 1:
#        plt.axvline(x=iterations*dt/5, linewidth=3, color='k', linestyle='-.')
#        plt.savefig("figures/activeInferencePIDTuning_f.pdf")
#    if simulation == 2:
#        plt.savefig("figures/activeInferencePIDTuningHyperPriors_f.pdf")
#    if simulation == 5:
#        plt.savefig("figures/activeInferencePIDMeasurementNoise_f.pdf")
    
if simulation == 6:
    plt.figure(figsize=(9, 6))
    plt.title('Log-precision, $\gamma_w$')
    plt.plot(np.arange(0, T-dt, dt), mu_gamma_w_history[:-1, 0], 'r', label='Estimated log-precision, $\mu_{\gamma_w}$')
    plt.plot(np.arange(0, T-dt, dt), gamma_w_history[:-1, 0], 'b', label='Real log-precision, $\gamma_w$')
    #plt.axhline(y=-np.log(np.var(psi_history[int(T/(4*dt)):-1,0,0])), xmin=0.0, xmax=T, color='g', label='Measured precision')
    plt.plot(np.arange(0, T-dt, dt), eta_gamma_w_history[:-1, 0, 0], 'g', label='Prior precision, $\eta_{\gamma_{w\'}}$')
    plt.xlabel('Time ($s$)')
    plt.legend()
    plt.savefig("figures/activeInferencePIDModelUncertainty_e.pdf")
    
    plt.figure(figsize=(9, 6))
    plt.title('Log-precision,, $\gamma_{w\'}$')
    plt.plot(np.arange(0, T-dt, dt), mu_gamma_w_history[:-1, 1], 'r', label='Expected log-precision, $\mu_{\gamma_{w\'}}$')
    plt.plot(np.arange(0, T-dt, dt), gamma_w_history[:-1, 1], 'b', label='Real log-precision, $\gamma_{w\'}$')
    #plt.axhline(y=-np.log(np.var(psi_history[int(T/(4*dt)):-1,0,1])), xmin=0.0, xmax=T, color='g', label='Measured precision')
    plt.plot(np.arange(0, T-dt, dt), eta_gamma_w_history[:-1, 0, 1], 'g', label='Prior precision, $\eta_{\gamma_{w\'}}$')
    plt.xlabel('Time ($s$)')
    plt.legend()
    plt.savefig("figures/activeInferencePIDModelUncertainty_f.pdf")
    
    #plt.figure()
    #plt.title('dFdmu_gamma_z')
    #plt.plot(range(iterations-1), dFdmu_gamma_z_history[:-1, 0])
    #
    #plt.figure()
    #plt.title('Phi')
    #plt.plot(range(iterations-1), phi_history[:-1, 0])
    
#plt.figure()
#plt.title('Mu_pi_z0')
#plt.plot(range(iterations-1), mu_pi_z_history[:-1, 0])
#
#plt.figure()
#plt.title('Mu_pi_z1')
#plt.plot(range(iterations-1), mu_pi_z_history[:-1, 1])
#    
#plt.figure()
#plt.plot(kappa_z_history[:-1])
    #
    #plt.figure()
    #plt.title('Mu_pi_w0')
    #plt.plot(range(iterations-1), mu_pi_w_history[:-1, 0])
    #
    #plt.figure()
    #plt.title('Mu_pi_w1')
    #plt.plot(range(iterations-1), mu_pi_w_history[:-1, 1])
    
    #plt.figure()
    #plt.title('Free energy')
    #plt.plot(np.arange(0, T-dt, dt), FE_history[:-1])



#plt.figure(figsize=(9, 6))
#plt.title('Log-precision, $\gamma_w$')
#plt.plot(np.arange(0, T-dt, dt), mu_gamma_w_history[:-1, 0], 'r', label='Estimated log-precision, $\mu_{\gamma_w}$')
##    plt.plot(np.arange(0, T-dt, dt), gamma_w_history[:-1, 0], 'b', label='Real log-precision, $\gamma_w$')
##plt.axhline(y=-np.log(np.var(psi_history[int(T/(4*dt)):-1,0,0])), xmin=0.0, xmax=T, color='g', label='Measured precision')
#plt.plot(np.arange(0, T-dt, dt), eta_gamma_w_history[:-1, 0, 0], 'g', label='Prior precision, $\eta_{\gamma_{w\'}}$')
#plt.xlabel('Time ($s$)')
#plt.legend()
#plt.savefig("figures/activeInferencePIDModelUncertainty_e.pdf")
#
#plt.figure(figsize=(9, 6))
#plt.title('Log-precision,, $\gamma_{w\'}$')
#plt.plot(np.arange(0, T-dt, dt), mu_gamma_w_history[:-1, 1], 'r', label='Expected log-precision, $\mu_{\gamma_{w\'}}$')
##    plt.plot(np.arange(0, T-dt, dt), gamma_w_history[:-1, 1], 'b', label='Real log-precision, $\gamma_{w\'}$')
##plt.axhline(y=-np.log(np.var(psi_history[int(T/(4*dt)):-1,0,1])), xmin=0.0, xmax=T, color='g', label='Measured precision')
#plt.plot(np.arange(0, T-dt, dt), eta_gamma_w_history[:-1, 0, 1], 'g', label='Prior precision, $\eta_{\gamma_{w\'}}$')
#plt.xlabel('Time ($s$)')
#plt.legend()
#plt.savefig("figures/activeInferencePIDModelUncertainty_f.pdf")


