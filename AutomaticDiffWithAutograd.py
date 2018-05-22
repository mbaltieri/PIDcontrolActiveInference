#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 22:30:12 2018

Testing autograd

@author: manuelbaltieri
"""

import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad, jacobian

def tanh(x):                 # Define a function
    y = np.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)       # Obtain its gradient function
a = grad_tanh(1.0)               # Evaluate the gradient at x = 1.0


def f(x,y):
    d = np.zeros((2,))
    d[0] = x[0]**2 * x[1]
    d[1] = 5 * x[0] + np.sin(x[1])
    return d
    return np.array([x[0]**2 * x[1], 5 * x[0] + np.sin(x[1])])

x0 = np.array([1.0, 0.0])
jac_f = jacobian(f)
bb = f(x0,0)
b = jac_f(x0,0)

c = np.eye(3, k=1)