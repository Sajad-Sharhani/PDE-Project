#!/usr/bin/env python3
import numpy as np
import scipy.sparse.linalg as spsplg
import scipy.linalg as splg
import operators as ops
import matplotlib.pyplot as plt
import time

######################################################################################
##                                                                                  ##
##  Lab "Introduction to Finite Difference Methods", part 1, for course             ##
##  "Scientific computing for PDEs" at Uppsala University.                          ##
##                                                                                  ##
##  Author: Gustav Eriksson                                                         ##
##  Date:   2022-08-31                                                              ##
##                                                                                  ##
##  Based on Matlab code written by Ken Mattsson in June 2022.                      ##
##                                                                                  ##
##  Solves the inviscid Burger's equation u_t + 0.5(u^2)_x = 0 with periodic        ##
##  boundary conditions using summation-by-parts finite differences. Non-smooth     ##
##  initial data is used to compare the dispersion errors (difference between the   ##
##  exact and numerical solution) of the stencils with and without artificial       ##
##  dissipation.                                                                    ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##  - Python     3.9.2                                                              ##
##  - Numpy      1.19.5                                                             ##
##  - Scipy      1.7.0                                                              ##
##  - Matplotlib 3.3.4                                                              ##
##                                                                                  ##
######################################################################################

# Method parameters
# Number of grid points, integer > 15.
mx = 201

# Order of accuracy. 2, 4, 6, 8, 10, or 12.
order = 4

# If using implicit operator, boolean. If true, 'order' is not used.
use_implicit = False

# If using artificial dissipation.
use_AD = False

# Initial data


def f(x):
    return (np.abs(2*x - 0.3) <= 0.25)*np.exp(-200*(2*x - 0.3)**2) + 0.5


# Model parameters
T = 2  # end time
xl = 0
xr = 1

# Space discretization
hx = (xr - xl)/mx
xvec = np.linspace(xl, xr-hx, mx)  # periodic, u(x=0) = u(x=1)

if use_implicit:
    H, Q = ops.periodic_imp(mx, hx, use_AD)
else:
    H, Q = ops.periodic_expl(mx, hx, order, use_AD)

# Time discretization
ht_try = 0.04*hx
mt = int(np.ceil(T/ht_try) + 1)  # round up so that (mt-1)*ht = T
tvec, ht = np.linspace(0, T, mt, retstep=True)

LUP = spsplg.splu(H)


def rhs(v):
    return LUP.solve(-Q@(0.5*v*v))


# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
v = f(xvec)
[line] = ax.plot(xvec, v, label='Approximation')
ax.set_xlim([xl, xr-hx])
ax.set_ylim([0, 2])
title = plt.title("t = " + "{:.2f}".format(0))
plt.draw()
plt.pause(1)

# Runge-Kutta 4
t = 0
for tidx in range(mt-1):
    k1 = ht*rhs(v)
    k2 = ht*rhs(v + 0.5*k1)
    k3 = ht*rhs(v + 0.5*k2)
    k4 = ht*rhs(v + k3)

    v = v + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + ht

    # Update plot every 50th time step
    if tidx % 50 == 0:
        line.set_ydata(v)
        title.set_text("t = " + "{:.2f}".format(tvec[tidx+1]))
        plt.draw()
        plt.pause(1e-8)

# Plot final approximation and exact solution
line.set_ydata(v)
title.set_text("t = " + "{:.2f}".format(T))
plt.draw()
plt.show()
