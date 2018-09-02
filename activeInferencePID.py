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


simulation = 1
# 0: PID control as active inference, 1: PID tuning based on means, 2: PID tuning based on means and variances

dt = .01
T = 5000
T_switch = int(T/3)
iterations = int(T / dt)
alpha = 100000.                                              # drift in Generative Model
gamma = 1                                                   # drift in OU process
plt.close('all')

obs_states = 1
hidden_states = 1                                           # x, in Friston's work
hidden_causes = 1                                           # v, in Friston's work
states = obs_states + hidden_states
temp_orders_states = 3                                      # generalised coordinates for hidden states x, but only using n-1
temp_orders_causes = 3                                      # generalised coordinates for hidden causes v (or \eta in biorxiv manuscript), but only using n-1



### cruise control problem from Astrom and Murray (2010), pp 65-69

# environment parameters
ga = 9.81                                                   # gravitational acceleration
theta = 4.                                                  # hill angle
C_r = .01                                                   # rolling friction coefficient
C_d = .32                                                   # drag coefficient
rho_air = 1.3                                               # air density
A = 2.4                                                     # frontal area agent

# car's parameters
#m = 1000                                                    # car mass (book example)
m = 500                                                    # car mass
T_m = 190                                                   # maximum torque
omega_m = 420                                               # engine speed to reach T_m
alpha_n = 12                                                # = gear ration/wheel radius,
                                                            # a1 = 40, a2 = 25, a3 = 16, a4 = 12, a5 = 10
beta = .4

x = np.zeros((hidden_states, temp_orders_states))           # position
v = np.zeros((hidden_causes, temp_orders_states - 1))
y = np.zeros((obs_states, temp_orders_states - 1))
eta_x = np.zeros((hidden_causes, temp_orders_states - 1))
eta_gamma_z = np.zeros((obs_states, temp_orders_states - 1))

eta_x[0, 0] = 10*alpha                                         # desired velocity , 0 to avoid mischaracterisation of precision from the start (large initial fluctuation is similar to large variance)
eta_gamma_z[0, 0] = 3.
eta_gamma_z[0, 1] = 1.




### free energy variables
a = np.zeros((obs_states, temp_orders_states-1))
phi = np.zeros((obs_states, temp_orders_states-1))
psi = np.zeros((obs_states, temp_orders_states-1))

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
kappa_z = 1                                                 # damping on precisions minimisation
kappa_w = 10                                                 # damping on precisions minimisation

# noise on sensory input (world - generative process)
#gamma_z = -16 * np.ones((obs_states, temp_orders_states - 1))  # log-precisions
#gamma_z[0, 0] = 4
gamma_z = 0 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
gamma_z[:,1] = gamma_z[:,0] - np.log(2 * gamma)
pi_z = np.exp(gamma_z) * np.ones((obs_states, temp_orders_states - 1))
#pi_z[0, 1] = pi_z[0, 0] / (2 * gamma)
sigma_z = 1 / (np.sqrt(pi_z))
z = np.zeros((iterations, obs_states, temp_orders_states - 1))
for i in range(obs_states):
    for j in range(temp_orders_states - 1):
        z[:, i, j] = sigma_z[i, j] * np.random.randn(1, iterations)

# noise on motion of hidden states (world - generative process)
gamma_w = 2                                                  # log-precision
pi_w = np.exp(gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
pi_w[0, 1] = pi_w[0, 0] / (2 * gamma)
sigma_w = 1 / (np.sqrt(pi_w))
w = np.zeros((iterations, hidden_states, temp_orders_states - 1))
for i in range(hidden_states):
    for j in range(temp_orders_states - 1):
        w[:, i, j] = sigma_w[i, j] * np.random.randn(1, iterations)


# agent's estimates of the noise (agent - generative model)
#mu_gamma_z = -16 * np.ones((obs_states, temp_orders_states - 1))  # log-precisions
#mu_gamma_z[0, 0] = -8
if simulation == 0:
    mu_gamma_z = 0.0 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
else:
    mu_gamma_z = -3.0 * np.ones((obs_states, temp_orders_states - 1))    # log-precisions
mu_gamma_z[0, 1] = mu_gamma_z[0, 0] - np.log(2 * gamma)
#mu_gamma_z[0, 0] = 3.708050201
#mu_gamma_z[0, 1] = -.6931471806
#mu_gamma_z[0, 1] = 1
#mu_gamma_z[0, 1] = -0.7
mu_pi_z = np.exp(mu_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
mu_gamma_w = -20 * np.ones((obs_states, temp_orders_states - 1))   # log-precision
mu_gamma_w[0, 1] = mu_gamma_w[0, 0] - np.log(2)
mu_pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))

if simulation == 2:
    mu_gamma_gamma_z = 5 * np.ones((obs_states, temp_orders_states - 1))
else:
    mu_gamma_gamma_z = - large_value * np.ones((obs_states, temp_orders_states - 1))
mu_c_gamma_z = np.exp(mu_gamma_gamma_z) * np.ones((obs_states, temp_orders_states - 1))


# history
x_history = np.zeros((iterations, hidden_states, temp_orders_states))
y_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
v_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))
rho_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
mu_x_history = np.zeros((iterations, hidden_states, temp_orders_states))
eta_x_history = np.zeros((iterations, hidden_causes, temp_orders_states - 1))
a_history = np.zeros((iterations, temp_orders_states - 1))
mu_gamma_z_history = np.zeros((iterations, temp_orders_states-1))
mu_gamma_w_history = np.zeros((iterations, temp_orders_states-1))
mu_pi_z_history = np.zeros((iterations, temp_orders_states-1))
mu_pi_w_history = np.zeros((iterations, temp_orders_states-1))
dFdmu_gamma_z_history = np.zeros((iterations, temp_orders_states-1))
dFdmu_gamma_w_history = np.zeros((iterations, temp_orders_states-1))
phi_history = np.zeros((iterations, temp_orders_states-1))
psi_history = np.zeros((iterations, temp_orders_states-1))


xi_z_history = np.zeros((iterations, obs_states, temp_orders_states - 1))
xi_w_history = np.zeros((iterations, hidden_states, temp_orders_states - 1))

kappa_z_history = np.zeros((iterations,1))

FE_history = np.zeros((iterations,))




### FUNCTIONS ###

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

def getObservation(x, v, a, w):
    x[:, 1:] = f(x[:, :-1], v, a)# + w[i, :, :]
    x[:, 0] += dt * x[:, 1]
    return g(x[:, :-1], v)

def F(rho, mu_x, mu_gamma_z, mu_pi_w):
    return .5 * (np.sum(np.exp(mu_gamma_z) * (rho - mu_x[0, :-1])**2) +
                 np.sum(mu_pi_w * (mu_x[0, 1:] + alpha * (mu_x[0, :-1] - eta_x))**2) -
                 np.log(np.prod(np.exp(mu_gamma_z)) * np.prod(mu_pi_w)))

mu_gamma_w_init = mu_gamma_w[0, 0]

for i in range(iterations - 1):
    print(i)
#    kappa_z = 70 * np.tanh(.01*i/T) + 10.0
    
    # re-encode precisions
#    mu_gamma_z[0, 1] = mu_gamma_z[0, 0] - np.log(2 * gamma)
    mu_pi_z = np.exp(mu_gamma_z) * np.ones((obs_states, temp_orders_states - 1))
    mu_pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
#    mu_pi_z[0, 1] = mu_pi_z[0, 0] / (2 * gamma)
    
#    if i > int(50/dt):
#        kappa_z = 100
    
    # include an external disturbance to test integral term
    if (simulation ==0) and (i > iterations/3) and (i < 2*iterations/3):
        v[0,0] = 5.0
    else:
        v[0,0] = 0.0
    
    
    # Analytical noise, for one extra level of generalised cooordinates, this is equivalent to an ornstein-uhlenbeck process
    dw = - gamma * w[i, 0, 0] + w[i, 0, 1] / np.sqrt(dt)
    dz = - gamma * z[i, 0, 0] + z[i, 0, 1] / np.sqrt(dt)
    
    w[i+1, 0, 0] = w[i, 0, 0] + dt * dw                               # noise in dynamics, at the moment not used in generative process
    z[i+1, 0, 0] = z[i, 0, 0] + dt * dz
    
    y = getObservation(x, v, a, w)
    rho = y + z[i, 0, :]
    
    # prediction errors (only used for plotting)
    eps_z = y - g_gm(mu_x[:, :-1], mu_v)
    xi_z = mu_pi_z * eps_z
    eps_w = mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1])
    xi_w = mu_pi_w * eps_w
    
    ### minimise free energy ###
    # perception
    Dmu_x[0, :-1] = mu_x[0, 1:]
    dFdmu_x[0, :-1] = np.array([mu_pi_z * - (rho - mu_x[0, :-1]) + mu_pi_w * alpha * [mu_x[0, 1:] + alpha * mu_x[0, :-1] - eta_x]])
#    dFdmu_x[0, 1:] += np.squeeze(mu_pi_w * [mu_x[0, :-1] + alpha * mu_x[0, :-1] - eta_x])
    
    # action
    dFdy = mu_pi_z * (rho - eta_x/alpha)
    dyda = np.ones((obs_states, temp_orders_states-1))
#    dyda = np.array([0., 1.])
    dFda = np.zeros((obs_states, temp_orders_states-1))
    dFda[0, 0] = np.sum(dFdy * dyda)
    
    # attention
    dFdmu_gamma_z = .5 * (np.exp(mu_gamma_z) * (rho - mu_x[0, :-1])**2 - 1) + mu_c_gamma_z * (mu_gamma_z - eta_gamma_z)
    dFdmu_gamma_w = .5 * (np.exp(mu_gamma_w) * (mu_x[0, :-1] + alpha * mu_x[0, :-1] - eta_x)**2 - 1)
    
    
    # update equations
    mu_x += dt * (Dmu_x - k_mu_x * dFdmu_x)
    a += dt * - k_a * dFda

    phi += dt * (- dFdmu_gamma_z - kappa_z * phi)
    psi += dt * (- dFdmu_gamma_w - kappa_w * psi)
    
    if (simulation == (1 or 2)) and (i > iterations/2):
        mu_gamma_z += dt * k_mu_gamma_z * phi
        kappa_z = 70 * np.tanh((i - iterations/2)/(10*T))+10
#        kappa_z = 50 * i/iterations
        kappa_z_history[i] = kappa_z
    
#    if (i > iterations/3) and (i < 2*iterations/3):
#        mu_gamma_w += dt * k_mu_gamma_w * psi
#    if i > iterations/2:
#        mu_gamma_z += dt * k_mu_gamma_z * phi
#        mu_gamma_w += dt * k_mu_gamma_w * psi
#        mu_gamma_w = -20 * np.ones((obs_states, temp_orders_states - 1))   # log-precision
#        mu_gamma_w[0, 1] = mu_gamma_w[0, 0] - np.log(2)
#        mu_pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
    
#    if i <= T_switch/dt:
##        mu_gamma_z += dt * k_mu_gamma_z * phi
#        ac = 1
#    else:
##        mu_gamma_w[0, 0] = mu_gamma_w_init + np.tanh(.5*dt*(i*dt - T_switch))
##        mu_gamma_w[0, 1] = mu_gamma_w[0, 0] - np.log(2)
#        mu_gamma_w[0, 0] = -20
#        mu_gamma_w[0, 1] = mu_gamma_w[0, 0] - np.log(2)
#        mu_pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
    
    
#    mu_gamma_z[0,1] += dt * k_mu_gamma_z * phi[0,1]
#    if i > iterations/3:
#        mu_gamma_z += dt * k_mu_gamma_z * phi
#        a += dt * - k_a * dFda
#        kappa_z = 10
#        mu_gamma_w = -21 * np.ones((obs_states, temp_orders_states - 1))   # log-precision
#        mu_gamma_w[0, 1] = mu_gamma_w[0, 0] - np.log(2)
#        mu_pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_states - 1))
#    if i == int(2*iterations/3):
#        mu_gamma_gamma_z = 5 * np.ones((obs_states, temp_orders_states - 1))
#        mu_c_gamma_z = np.exp(mu_gamma_gamma_z) * np.ones((obs_states, temp_orders_states - 1))

    
    # save history
    rho_history[i, :] = rho
    mu_x_history[i, :, :] = mu_x
    eta_x_history[i] = eta_x/alpha
    a_history[i] = a
    v_history[i] = v
    mu_gamma_z_history[i] = mu_gamma_z
    mu_gamma_w_history[i] = mu_gamma_w
    
    xi_z_history[i, :, :] = xi_z
    xi_w_history[i, :, :] = xi_w
    FE_history[i] = F(rho, mu_x, mu_gamma_z, mu_pi_w)
    
    phi_history[i] = phi
    psi_history[i] = psi
    dFdmu_gamma_z_history[i] = dFdmu_gamma_z
    dFdmu_gamma_w_history[i] = dFdmu_gamma_z
    mu_pi_z_history[i] = mu_pi_z
    mu_pi_w_history[i] = mu_pi_w

plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, T-dt, dt), rho_history[:-1,0,0], 'b', label='Measured velocity, $\\rho$')
plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,0], 'r', label='Expected velocity, $\mu_x$')
plt.plot(np.arange(0, T-dt, dt), eta_x_history[:-1,0,0], 'g', label='Desired velocity, $\eta_x$')
plt.title('Hidden state, x (car velocity)')
#plt.xlabel('Time ($s$)')
plt.ylabel('Velocity ($km/h$)')
plt.legend(loc=1)
if simulation == 0:
    plt.savefig("figures/activeInferencePID_a.pdf")
elif simulation == 1:
    plt.savefig("figures/activeInferencePIDTuning_a.pdf")
    

plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, T-dt, dt), rho_history[:-1,0,1], 'b', label='Measured acceleration, $\\rho\'$')
plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,1], 'r', label='Estimated acceleration, $\mu_x\'$')
plt.plot(np.arange(0, T-dt, dt), eta_x_history[:-1,0,1], 'g', label='Desired acceleration, $\eta_x\'$')
plt.title('Hidden state, x\' (car acceleration)')
#plt.xlabel('Time ($s$)')
plt.ylabel('Acceleration ($km/h^2$)')
plt.legend(loc=1)
if simulation == 0:
    plt.savefig("figures/activeInferencePID_b.pdf")
elif simulation == 1:
    plt.savefig("figures/activeInferencePIDTuning_b.pdf")

plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, T-dt, dt), v_history[:-1,0,0], 'k')
plt.title('Hidden input, v (external disturbance)')
plt.xlabel('Time ($s$)')
plt.ylabel('Acceleration ($km/h^2$)')
if simulation == 0:
    plt.savefig("figures/activeInferencePID_c.pdf")
elif simulation == 1:
    plt.savefig("figures/activeInferencePIDTuning_c.pdf")

plt.figure(figsize=(8, 6))
plt.title('Action, a (control)')
plt.plot(np.arange(0, T-dt, dt), a_history[:-1,0])
plt.xlabel('Time ($s$)')
plt.ylabel('Acceleration ($km/h^2$)')
if simulation == 0:
    plt.savefig("figures/activeInferencePID_d.pdf")
elif simulation == 1:
    plt.savefig("figures/activeInferencePIDTuning_d.pdf")

#print(np.var(rho_history[int(2/dt):int(T/3/dt),0,0]))
print(np.var(rho_history[int(T/4/dt):int(2*T/4/dt),0,0]))
print(np.var(rho_history[int(T/2/dt):int(T/dt),0,0]))

if simulation > 0:
    plt.figure(figsize=(8, 6))
    plt.title('Log-precision, $\gamma_z$ (= integral gain, $k_i$)')
    plt.plot(np.arange(0, T-dt, dt), mu_gamma_z_history[:-1, 0], 'r', label='Estimated log-precision, $\mu_{\gamma_z}$')
    plt.axhline(y=gamma_z[0,0], xmin=0.0, xmax=T, color='b', label='Real log-precision, $\gamma_z$')
#    plt.axhline(y=-np.log(np.var(rho_history[int(T/(4*dt)):-1,0,0])), xmin=0.0, xmax=T, color='g', label='Measured precision')
    if simulation == 2:
        plt.axhline(y=eta_gamma_z[0,0], xmin=0.0, xmax=T, color='k', label='Desired precision')
    plt.legend(loc=1)
    plt.savefig("figures/activeInferencePIDTuning_e.pdf")
    
    plt.figure(figsize=(8, 6))
    plt.title('Log-precision, $\gamma_{z\'}$ (= integral gain, $k_p$)')
    plt.plot(np.arange(0, T-dt, dt), mu_gamma_z_history[:-1, 1], 'r', label='Expected log-precision, $\mu_{\gamma_{z\'}}$')
    plt.axhline(y=gamma_z[0,1], xmin=0.0, xmax=T, color='b', label='Real log-precision, $\gamma_{z\'}$')
#    plt.axhline(y=-np.log(np.var(rho_history[int(T/(4*dt)):-1,0,1])), xmin=0.0, xmax=T, color='g', label='Measured precision')
    if simulation == 2:
        plt.axhline(y=eta_gamma_z[0,1], xmin=0.0, xmax=T, color='k', label='Desired precision')
    plt.legend(loc=1)
    plt.savefig("figures/activeInferencePIDTuning_f.pdf")
    
    #
    #plt.figure()
    #plt.title('Log-precision w0')
    #plt.plot(range(iterations-1), mu_gamma_w_history[:-1, 0], 'r', label='Estimated precision')
    #plt.axhline(y=gamma_w, xmin=0.0, xmax=T, color='b', label='Theoretical precision')
    ##plt.axhline(y=-np.log(np.var(rho_history[int(T/(4*dt)):-1,0,0])), xmin=0.0, xmax=T, color='g', label='Measured precision')
    #plt.legend()
    
    #plt.figure()
    #plt.title('Log-precision w1')
    #plt.plot(range(iterations-1), mu_gamma_w_history[:-1, 1], 'r', label='Estimated precision')
    #plt.axhline(y=gamma_w[0,1], xmin=0.0, xmax=T, color='b', label='Theoretical precision')
    ##plt.axhline(y=-np.log(np.var(rho_history[int(T/(4*dt)):-1,0,1])), xmin=0.0, xmax=T, color='g', label='Measured precision')
    #plt.legend()
    
    #plt.figure()
    #plt.title('dFdmu_gamma_z')
    #plt.plot(range(iterations-1), dFdmu_gamma_z_history[:-1, 0])
    #
    #plt.figure()
    #plt.title('Phi')
    #plt.plot(range(iterations-1), phi_history[:-1, 0])
    
    plt.figure()
    plt.title('Mu_pi_z0')
    plt.plot(range(iterations-1), mu_pi_z_history[:-1, 0])
    
    plt.figure()
    plt.title('Mu_pi_z1')
    plt.plot(range(iterations-1), mu_pi_z_history[:-1, 1])
    
    plt.figure()
    plt.plot(kappa_z_history[:-1])
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






