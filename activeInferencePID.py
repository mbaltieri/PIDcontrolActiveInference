#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:52:21 2018

Active inference version of a PID controller

@author: manuelbaltieri
"""

import numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 30
iterations = int(T / dt)
plt.close('all')

obs_states = 1
hidden_states = 1                               # x, in Friston's work
hidden_causes = 1                               # v, in Friston's work
states = obs_states + hidden_states
temp_orders_states = 3
temp_orders_causes = 3



### cruise control problem from Astrom and Murray (2010), pp 65-69

# environment parameters
ga = 9.81                                   # gravitational acceleration
theta = 4.                                  # hill angle
C_r = .01                                   # rolling friction coefficient
C_d = .32                                   # drag coefficient
rho_air = 1.3                               # air density
A = 2.4                                     # frontal area agent

# car's parameters
m = 1000                                    # car mass
T_m = 190                                   # maximum torque
omega_m = 420                               # engine speed to reach T_m
alpha_n = 12                                # = gear ration/wheel radius,
                                            # a1 = 40, a2 = 25, a3 = 16, a4 = 12, a5 = 10
beta = .4

x = np.zeros((hidden_states, temp_orders_states))   # position
v = np.zeros((hidden_causes, temp_orders_states - 1))
y = np.zeros((obs_states, temp_orders_states - 1))
eta = np.zeros((hidden_causes, temp_orders_states - 1))

eta[0, 0] = 10              # desired velocity




### free energy variables

FE = np.zeros((iterations,))
a = np.zeros((obs_states, temp_orders_states))

mu_x = np.random.randn(hidden_states, temp_orders_states)
mu_x = np.zeros((hidden_states, temp_orders_states))
mu_v = np.random.randn(hidden_causes, temp_orders_states)
mu_v = np.zeros((hidden_causes, temp_orders_states))
mu_gamma_z = 10 * np.ones((obs_states, temp_orders_states))

# minimisation variables and parameters
dFdmu_x = np.zeros((hidden_states, temp_orders_states))
dFdmu_v = np.zeros((hidden_causes, temp_orders_states))
Dmu_x = np.zeros((hidden_states, temp_orders_states))
Dmu_v = np.zeros((hidden_causes, temp_orders_states))
eta_mu_x = 1
eta_a = 1

# noise on sensory input (world - generative process)
#gamma_z = -16 * np.ones((obs_states, temp_orders_states - 1))  # log-precisions
#gamma_z[0, 0] = 4
gamma_z = 0 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
pi_z = np.exp(gamma_z) * np.ones((obs_states, temp_orders_states - 1))
sigma_z = 1 / (np.sqrt(pi_z))
z = np.zeros((iterations, obs_states, temp_orders_states - 1))
for i in range(obs_states):
    for j in range(temp_orders_states - 1):
        z[:, i, j] = sigma_z[i, j] * np.random.randn(1, iterations)

# noise on motion of hidden states (world - generative process)
gamma_w = 0                                                  # log-precision
pi_w = np.exp(gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
sigma_w = 1 / (np.sqrt(pi_w))
w = np.zeros((iterations, hidden_states, temp_orders_states - 1))
for i in range(hidden_states):
    for j in range(temp_orders_states - 1):
        w[:, i, j] = sigma_w[i, j] * np.random.randn(1, iterations)


# agent's estimates of the noise (agent - generative model)
#mu_gamma_z = -16 * np.ones((obs_states, temp_orders_states - 1))  # log-precisions
#mu_gamma_z[0, 0] = -8
mu_gamma_z = 4 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
pi_z = np.exp(mu_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
mu_gamma_w = 2                                                  # log-precision
pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))



# history
x_history = np.zeros((iterations, hidden_states, temp_orders_states))
y_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
v_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))
rho_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
mu_x_history = np.zeros((iterations, hidden_states, temp_orders_states))
eta_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))
a_history = np.zeros((iterations, temp_orders_states))

xi_z_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
xi_w_history = np.zeros((iterations, hidden_states, temp_orders_states - 1))




### functions ###
## cruise control ##

def force_gravitation(theta):
    return m * ga * np.sin(theta)

def force_friction(v):
    return m * ga * C_r * np.sign(v)

def force_drag(v):
    return .5 * rho_air * C_d * A * v**2

def force_disturbance(v, theta):
    return force_gravitation(theta) + force_friction(v) + force_drag(v)

def force_drive(v, u):
    return alpha_n * u * torque(v)

def torque(v):
    return T_m * (1 - beta * (omega(v) / omega_m)**2)

def omega(v):
    return alpha_n * v

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

def getObservation(x, v, a):
    x[:, 1:] = f(x[:, :-1], v, a[:, 1:]) + w[i, :, :]
    x[:, 0] += dt * x[:, 1]
    return g(x[:, :-1], v)



for i in range(iterations - 1):
    print(i)
    
    # include an external disturbance to test integral term
    if (i > iterations/3) and (i < 2*iterations/3):
        v[0,0] = 50.0
    else:
        v[0,0] = 0.0
    
    z[i+1, 0, 0] = z[i, 0, 0] + dt * (z[i, 0, 1])
    w[i+1, 0, 0] = w[i, 0, 0] + dt * (w[i, 0, 1])
    
    rho = getObservation(x, v, a) + z[i, 0, :]
    
    # prediction errors (only used for plotting)
    eps_z = y - g_gm(mu_x[:, :-1], mu_v)
    xi_z = pi_z * eps_z
    eps_w = mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1])
    xi_w = pi_w * eps_w
    
    # minimise free energy
    Dmu_x[0, :-1] = mu_x[0, 1:]
    dFdmu_x[0, :-1] = np.array([pi_z * -(rho - mu_x[0, :-1]) + pi_w * [mu_x[0, 1:] + mu_x[0, :-1] - eta]])
    
    dFda = np.array([0., np.sum(pi_z * (rho - mu_x[0, :-1])), 0.])
    
    
    # update equations
    mu_x += dt * (Dmu_x - eta_mu_x * dFdmu_x)
    a += dt * - eta_a * dFda
    
    
    # save history
    rho_history[i, :] = rho
    mu_x_history[i, :, :] = mu_x
    eta_history[i] = eta
    a_history[i] = a
    v_history[i] = v
    
    xi_z_history[i, :, :] = xi_z
    xi_w_history[i, :, :] = xi_w
    FE[i] = np.sum(xi_z) + np.sum(xi_w)


plt.figure()
plt.plot(np.arange(0, T-dt, dt), rho_history[:-1,0,0], 'b', label='Measured velocity')
plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,0], 'r', label='Estimated velocity')
plt.plot(np.arange(0, T-dt, dt), eta_history[:-1,0,0], 'g', label='Desired velocity')
plt.title('Car velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (km/h)')
plt.legend()

plt.figure()
plt.plot(np.arange(0, T-dt, dt), v_history[:-1,0,0], 'k')
plt.title('External disturbance')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (km/h)')

#plt.figure()
#plt.title('Velocity')
#plt.plot(range(iterations-1), rho_history[:-1,0,1])
#plt.plot(range(iterations-1), mu_x_history[:-1,0,1])
#plt.plot(range(iterations-1), eta_history[:-1,0,1])

plt.figure()
plt.title('Action')
plt.plot(range(iterations-1), a_history[:-1,1])

    
    
    
    