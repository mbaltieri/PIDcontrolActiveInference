# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:10:05 2016

Cruise control model (elaborated from 'Feedback systems' by Astrom)

@author: manuelbaltieri
"""

import numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 10
iterations = int(T/dt)

variables = 1
temp_orders = 2

# environment parameters
g = 9.81                                    # gravitational acceleration
theta = 4                                  # hill angle
C_r = .01                                   # rolling friction coefficient
C_d = .32                                   # drag coefficient
rho = 1.3                                   # air density
A = 2.4                                     # frontal area agent

# agent's parameters
m = 100                                    # car mass
T_m = 190                                   # maximum torque
omega_m = 420                               # engine speed to reach T_m
alpha_n = 12                                # = gear ration/wheel radius, a1 = 40, a2 = 25, a3 = 16, a4 = 12, a5 = 10
beta = .4

x = np.zeros((variables,temp_orders))       # hidden state (made observable to simplify things, y = x), x[0]=v x[1]=v_dot
y = np.zeros((variables,temp_orders))       # measured state
u = 0                                       # input
v = np.zeros((variables,temp_orders))       # z[0]: error, z[1]: integral of the error

n = np.random.randn(iterations,temp_orders+1)
n[:,1] *= np.sqrt(2)

k_p = 0.5
k_i = 1.0

v_ref = 10

ext_input = np.zeros((1, iterations))

# history
y_history = np.zeros((iterations,variables,temp_orders))
x_history = np.zeros((iterations,variables,temp_orders))
u_history = np.zeros((iterations,1))
a_history = np.zeros((iterations,1))
v_history = np.zeros((iterations,variables,temp_orders))

v_ref_history = v_ref*np.ones((iterations))

# functions

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def force_gravitation(theta):
    return m*g*np.sin(theta)

def force_friction(v):
    return m*g*C_r*np.sign(v)
    
def force_drag(v):
    return .5*rho*C_d*A*v**2
    
def force_disturbance(v, theta):
    return force_gravitation(theta) + force_friction(v) + force_drag(v)
    
def force_drive(v,u):
    return alpha_n*u*torque(v)

def torque(v):
    return T_m*(1 - beta*(omega(v)/omega_m)**2)
    
def omega(v):
    return alpha_n*v
    
    
for i in range(iterations-1):
    print(i)
    
    dn2 = - 1.0 * n[i, 1] + n[i, 2] / np.sqrt(dt)
    n[i+1, 1] = n[i, 1] + dt * dn2
    
    dn = - 1.0 * n[i, 0] + n[i, 1]
    n[i+1, 0] = n[i, 0] + dt * dn
    
#    v[0,1] = y[0,0] - v_ref
#    v[0,0] += dt*v[0,1]
#    
#    u = k_p*v[0,1] + k_i*v[0,0]
    
    v[0,1] = y[0,1] - 0.0
    v[0,0] = y[0,0] - v_ref
    
    u += dt*(k_p*v[0,1] + k_i*v[0,0])
    a = sigmoid(u)
    
#    ext_input[0,i] = .01*np.exp((i+iterations/2)*dt**1.3)
    x[0,1] = (force_drive(x[0,0],-a) - force_disturbance(x[0,0],theta))/m
#    if (i>iterations/4) and (i<iterations/2):
#        x[0,1] = (force_drive(x[0,0],-u) - force_disturbance(x[0,0],theta))/m + n[0,i] + ext_input[0,i]
#    else:
#        x[0,1] = (force_drive(x[0,0],-u) - force_disturbance(x[0,0],theta))/m + n[0,i]
    x[0,0] += dt*x[0,1]
    y = x + n[i,:-1]
    
    # save data
    y_history[i,:,:] = y
    x_history[i,:,:] = x
    u_history[i] = u
    a_history[i] = a
    v_history[i,:,:] = v
    
    
plt.close('all')

plt.figure(0)
plt.suptitle('Proportional Control + Noise')
plt.subplot(2,2,1)
plt.plot(range(iterations), x_history[:,0,0], 'b', range(iterations), v_ref_history, 'k')
plt.title('Velocity')
plt.subplot(2,2,2)
plt.plot(range(iterations), u_history, 'r')
plt.title('Control')
plt.subplot(2,2,3)
plt.plot(range(iterations), v_history[:,0,1], 'g')
plt.title('Error')
plt.subplot(2,2,4)
plt.plot(range(iterations), v_history[:,0,0], 'k')
plt.title('Error integral')    


plt.figure()
plt.plot(ext_input[0,int(iterations/4):int(iterations/2)])

plt.figure()
plt.plot(np.arange(0, T-dt, dt), y_history[:-1,0,0], 'b', np.arange(0, T-dt, dt), v_ref_history[:-1], 'k')

plt.figure()
plt.plot(np.arange(0, T-dt, dt), a_history[:-1], 'b')
plt.title('Control')

print(np.var(y_history[int(T/(4*dt)):-1,0,0]))
