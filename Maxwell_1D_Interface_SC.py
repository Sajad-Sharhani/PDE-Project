#!/usr/bin/env python3

######################################################################################
##                                                                                  ##
##  Lab "Introduction to Finite Difference Methods", part 3, for course             ##
##  "Scientific computing for PDEs" at Uppsala University.                          ##
##                                                                                  ##
##  Author: Gustav Eriksson                                                         ##
##  Date:   2022-08-31                                                              ##
##                                                                                  ##
##  Based on Matlab code written by Ken Mattsson in June 2022.                      ##
##                                                                                  ##
##  Solves the 1D Maxwell's equations with piecewise constant coefficients using    ##
##  summation-by-parts finite differences and the projection method to              ##
##  impose the boundary and interface conditions. Demonstrates the reflected and    ##
##  transmitted waves due to the discontinuous jump in permittivity.                ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##  - Python     3.9.2                                                              ##
##  - Numpy      1.19.5                                                             ##
##  - Scipy      1.7.0                                                              ##
##  - Matplotlib 3.3.4                                                              ##
##                                                                                  ##
######################################################################################

import numpy as np
from scipy.sparse import kron, csc_matrix, eye, vstack, bmat
from scipy.sparse.linalg import inv
from math import sqrt, ceil
import operators as ops
import matplotlib.pyplot as plt

# Method parameters
# Number of grid points, integer > 15.
mx = 201

# Order of accuracy: 2, 3, 4, 5, 6, or 7. Odd orders are upwind operators
order = 4

# Model parameters
T = 1.8 # end time

# Block boundaries
xl = -2
xi = 1
xr = 4

# Material properties
mu1 = 1
eps1 = 1
mu2 = 1
eps2 = 2

A = csc_matrix([[0,1],[1,0]])
C1 = csc_matrix([[eps1,0],[0,mu1]])
C2 = csc_matrix([[eps2,0],[0,mu2]])

# Speed of light
c1 = 1/sqrt(eps1*mu1)
c2 = 1/sqrt(eps2*mu2)

# Space discretization
# Assume same discretization in both blocks
N = 2*mx # number of degrees of freedom per block
NN = 2*N # total number of degrees of freedom 

xvec1,hx1 = np.linspace(xl,xi,mx,retstep=True)
xvec2,hx2 = np.linspace(xi,xr,mx,retstep=True)
assert(hx1 == hx2)

if order == 2:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_2nd(mx,hx1)
elif order == 3:
    H,HI,Dp,Dm,e_l,e_r = ops.sbp_upwind_3rd(mx,hx1)
elif order == 4:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_4th(mx,hx1)
elif order == 5:
    H,HI,Dp,Dm,e_l,e_r = ops.sbp_upwind_5th(mx,hx1)
elif order == 6:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_6th(mx,hx1)
elif order == 7:
    H,HI,Dp,Dm,e_l,e_r = ops.sbp_upwind_7th(mx,hx1)
else:
    raise NotImplementedError('Order not implemented.')

e1 = csc_matrix([1,0])
e2 = csc_matrix([0,1])

# Identity matrices
I_N = eye(N)
I_NN = eye(NN)
I_m = eye(mx)
I_2 = eye(2)

# Zero matrices
z_1_N = csc_matrix((1,N))
z_N_N = csc_matrix((N,N))

H_bar = kron(I_2,H)
HI_bar = kron(I_2,HI)

HI_bar_bar = kron(I_2,HI_bar)

Lbc1 = kron(e1,e_l)
Lbc2 = kron(e1,e_r)
Lbc = bmat([[Lbc1,z_1_N],[z_1_N,Lbc2]])

Lic = bmat([
    [kron(e1,e_r),-kron(e1,e_l)],
    [kron(e2,e_r),-kron(e2,e_l)]
    ])

L = vstack((Lbc,Lic),format="csc")

# Construct RHS using the projection method
P = I_NN - HI_bar_bar@L.T@inv(L@HI_bar_bar@L.T)@L
CI = bmat([
    [kron(inv(C1),I_m),z_N_N],
    [z_N_N,kron(inv(C2),I_m)],
    ],format="csc")
if order == 2 or order == 4 or order == 6:
    D = P@CI@kron(I_2,kron(A,D1))@P
else:
    D1 = 0.5*(Dp + Dm)
    DI = 0.5*(Dp - Dm)
    D = P@CI@kron(I_2,kron(A,D1) + kron(I_2,DI))@P

# Initial data
def gauss(x):
    rstar = 0.1
    return np.exp(-(x/rstar)**2)

Ey1 = -gauss(xvec1) - gauss(xvec1)
Hz1 = -gauss(xvec1) + gauss(xvec1)
Ey2 = -gauss(xvec2) - gauss(xvec2)
Hz2 = -gauss(xvec2) + gauss(xvec2)
v = np.hstack((Ey1,Hz1,Ey2,Hz2))

# Time discretization
CFL = 0.1/max(c1,c2)
ht_try = CFL*hx1
mt = int(ceil(T/ht_try) + 1) # round up so that (mt-1)*ht = T
tvec,ht = np.linspace(0,T,mt,retstep=True)

def rhs(v):
    return D@v

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
Ey = v[0:mx]
Hz = v[mx:]
[line11] = ax.plot(xvec1,Ey1,color='g',label='Electric field')
[line21] = ax.plot(xvec1,Hz1,'--',color='r',label='Magnetic field')
plt.legend()
[line12] = ax.plot(xvec2,Ey2,color='g',label='Approximation E')
[line22] = ax.plot(xvec2,Hz2,'--',color='r',label='Approximation H')
ax.set_xlim([xl,xr])
ax.set_ylim([-2,2])
title = plt.title("t = " + "{:.2f}".format(0))
plt.draw()
plt.pause(0.5)

# Runge-Kutta 4
t = 0
for tidx in range(mt-1):
    k1 = ht*rhs(v)
    k2 = ht*rhs(v + 0.5*k1)
    k3 = ht*rhs(v + 0.5*k2)
    k4 = ht*rhs(v + k3)

    v = v + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + ht

    # Update plot every 20th time step
    if tidx % ceil(5*max(c2,c1)) == 0 or tidx == mt-2:
        Ey1 = v[0:mx]
        Hz1 = v[mx:2*mx]
        Ey2 = v[2*mx:3*mx]
        Hz2 = v[3*mx:4*mx]
        line11.set_ydata(Ey1)
        line21.set_ydata(Hz1)
        line12.set_ydata(Ey2)
        line22.set_ydata(Hz2)
        title.set_text("t = " + "{:.2f}".format(tvec[tidx+1]))
        plt.draw()
        plt.pause(1e-3)

# Electric field reflection error
Ey1 = v[0:mx]
eta1 = sqrt(eps1)
eta2 = sqrt(eps2)
R_exact = (eta2 - eta1)/(eta1 + eta2)

ref_pos = 0.2 # position of reflected wave, assuming xi = 1, c1 = 1, T = 1.8
Ey1_check = Ey1[np.abs(xvec1 - ref_pos) < 0.1] # look for the peak 0.1 around the position
R_approx = np.max(np.abs(Ey1_check)) # magnitude of peak
if R_exact < 0: # swap sign if negative reflection
    R_approx = -R_approx

error = abs(R_exact - R_approx)
print("Reflection error: {:.4e}".format(error))

plt.show()
