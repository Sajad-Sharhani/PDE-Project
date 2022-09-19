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
##  Solves the 2D Maxwell's equations with piecewise constant coefficients using    ##
##  summation-by-parts finite differences and the projection method to              ##
##  impose the boundary and interface conditions. Demonstrates the behavior of the  ##
##  electric field divergence. The domain consists of four blocks where one block   ##
##  (block 2) has a jump in material properties. The blocks are ordered as follows: ##
##                                                                                  ##
##              #############                                                       ##
##              #     #     #                                                       ##
##              #  3  #  4  #                                                       ##
##              #     #     #                                                       ##
##              #############                                                       ##
##              #     #     #                                                       ##
##              #  1  #  2  #                                                       ##
##              #     #     #                                                       ##
##              #############                                                       ##
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
# Number of grid points in each direction and block, integer > 15.
m = 61

# Order of accuracy: 2, 3, 4, 5, 6, or 7. Odd orders are upwind operators
order = 5

# Type of boundary condition.
# 1 - Tangential electric field
# 2 - Magnetic field
# 3 - Non-reflecting
bc_type = 2

# Model parameters
T = 4 # end time

# Block boundaries
xl = -2
xi = 0
xr = 2
yl = -2
yi = 0
yr = 2

# Material properties
eps1 = 1
eps2 = 2
mu1 = 1
mu2 = 1

A = csc_matrix([[0,0,0],[0,0,-1],[0,-1,0]])
B = csc_matrix([[0,1,0],[1,0,0],[0,0,0]])
C1 = csc_matrix([[eps1,0,0],[0,mu1,0],[0,0,eps1]])
C2 = csc_matrix([[eps2,0,0],[0,mu2,0],[0,0,eps2]])

# Speed of light
c1 = 1/sqrt(eps1*mu1)
c2 = 1/sqrt(eps2*mu2)

# Space discretization
# Assume same discretization in all blocks and both dimensions
mtot = m*m # number of grid points per block
N = 3*mtot # number of degrees of freedom per block
NN = 4*N # total number of degrees of freedom

# Bottom left
xvec1, hx1 = np.linspace(xl,xi,m,retstep=True)
yvec1, hy1 = np.linspace(yl,yi,m,retstep=True)

# Bottom right
xvec2, hx2 = np.linspace(xi,xr,m,retstep=True)
yvec2, hy2 = np.linspace(yl,yi,m,retstep=True)

# Top left
xvec3, hx3 = np.linspace(xl,xi,m,retstep=True)
yvec3, hy3 = np.linspace(yi,yr,m,retstep=True)

# Top right
xvec4, hx4 = np.linspace(xi,xr,m,retstep=True)
yvec4, hy4 = np.linspace(yi,yr,m,retstep=True)

assert(hx1 == hx2)
assert(hx1 == hx3)
assert(hx1 == hx4)
assert(hy1 == hy2)
assert(hy1 == hy3)
assert(hy1 == hy4)

X1,Y1 = np.meshgrid(xvec1,yvec1)
X2,Y2 = np.meshgrid(xvec2,yvec2)
X3,Y3 = np.meshgrid(xvec3,yvec3)
X4,Y4 = np.meshgrid(xvec4,yvec4)

if order == 2:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_2nd(m,hx1)
elif order == 3:
    H,HI,Dp,Dm,e_l,e_r = ops.sbp_upwind_3rd(m,hx1)
elif order == 4:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_4th(m,hx1)
elif order == 5:
    H,HI,Dp,Dm,e_l,e_r = ops.sbp_upwind_5th(m,hx1)
elif order == 6:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_6th(m,hx1)
elif order == 7:
    H,HI,Dp,Dm,e_l,e_r = ops.sbp_upwind_7th(m,hx1)
else:
    raise NotImplementedError('Order not implemented.')

Im = eye(m,format="csc")
Im_bar = Im[1:-1,:] # remove first and last rows to exclude corner points later

e1 = csc_matrix([1,0,0])
e2 = csc_matrix([0,1,0])
e3 = csc_matrix([0,0,1])

# Identity matrices
I_3 = eye(3,format="csc")
I_4 = eye(4,format="csc")
I_N = eye(N,format="csc")
I_NN = eye(NN,format="csc")
I_mtot = eye(mtot,format="csc")

# Zero matrices
z_N_N = csc_matrix((N,N))
z_2m_N = csc_matrix((2*m,N))
z_mm2_N = csc_matrix((m-2,N))
z_2mm2_N = csc_matrix((2*m-2,N))
z_1_N = csc_matrix((1,N))

if order == 3 or order == 5 or order == 7:
    D1 = 0.5*(Dp + Dm)
    DI = 0.5*(Dp - Dm)
    DIx = kron(DI,Im)
    DIy = kron(Im,DI)
Dx = kron(D1,Im)
Dy = kron(Im,D1)

# One block one component
H = kron(H,H)
HI = kron(HI,HI)

# One block three components
H_bar = kron(I_3,H)
HI_bar = kron(I_3,HI)

# Four blocks three components
H_bar_bar = kron(I_4,H_bar)
HI_bar_bar = kron(I_4,HI_bar)

eW = kron(e_l,Im_bar) # note: not the corner points
eE = kron(e_r,Im_bar)
eS = kron(Im_bar,e_l)
eN = kron(Im_bar,e_r)
eWS = kron(e_l,e_l)
eES = kron(e_r,e_l)
eWN = kron(e_l,e_r)
eEN = kron(e_r,e_r)

# Stack the boundary operators. Boundary condition: Lbc*v = 0.
if bc_type == 1:
    Lbc1 = vstack((kron(e3,eW),kron(e1,eS),kron(e3,eWS),kron(e1,eWS),kron(e3,eWN),kron(e1,eES)))
    Lbc2 = vstack((kron(e3,eE),kron(e1,eS),kron(e1,eWS),kron(e3,eES),kron(e3,eEN),kron(e1,eES)))
    Lbc3 = vstack((kron(e3,eW),kron(e1,eN),kron(e3,eWS),kron(e3,eWN),kron(e1,eWN),kron(e1,eEN)))
    Lbc4 = vstack((kron(e3,eE),kron(e1,eN),kron(e3,eES),kron(e3,eEN),kron(e1,eWN),kron(e1,eEN)))
    Lbc = bmat([[Lbc1,z_2m_N,z_2m_N,z_2m_N],[z_2m_N,Lbc2,z_2m_N,z_2m_N],[z_2m_N,z_2m_N,Lbc3,z_2m_N],[z_2m_N,z_2m_N,z_2m_N,Lbc4]],format="csc")   
elif bc_type == 2:
    Lbc1 = vstack((kron(e2,eW),kron(e2,eS),kron(e2,eWS),kron(e2,eES)))
    Lbc2 = vstack((kron(e2,eE),kron(e2,eS),kron(e2,eES),kron(e2,eEN)))
    Lbc3 = vstack((kron(e2,eW),kron(e2,eN),kron(e2,eWN),kron(e2,eWS)))
    Lbc4 = vstack((kron(e2,eE),kron(e2,eN),kron(e2,eEN),kron(e2,eWN)))
    Lbc = bmat([[Lbc1,z_2mm2_N,z_2mm2_N,z_2mm2_N],[z_2mm2_N,Lbc2,z_2mm2_N,z_2mm2_N],[z_2mm2_N,z_2mm2_N,Lbc3,z_2mm2_N],[z_2mm2_N,z_2mm2_N,z_2mm2_N,Lbc4]],format="csc")   
elif bc_type == 3:
    Lbc1 = vstack((kron(e2+c1*e3,eW),kron(e1-c1*e2,eS),kron(e2+c1*e3,eWS),kron(e1-c1*e2,eWS),kron(e2+c1*e3,eWN),kron(e1-c1*e2,eES)))
    Lbc2 = vstack((kron(e2-c2*e3,eE),kron(e1-c2*e2,eS),kron(e1-c2*e2,eWS),kron(e1-c2*e2,eES),kron(e2-c2*e3,eEN),kron(e2-c2*e3,eES)))
    Lbc3 = vstack((kron(e2+c1*e3,eW),kron(e1+c1*e2,eN),kron(e2+c1*e3,eWS),kron(e2+c1*e3,eWN),kron(e1+c1*e2,eWN),kron(e1+c1*e2,eEN)))
    Lbc4 = vstack((kron(e2-c1*e3,eE),kron(e1+c1*e2,eN),kron(e2-c1*e3,eES),kron(e2-c1*e3,eEN),kron(e1+c1*e2,eWN),kron(e1+c1*e2,eEN)))
    Lbc = bmat([[Lbc1,z_2m_N,z_2m_N,z_2m_N],[z_2m_N,Lbc2,z_2m_N,z_2m_N],[z_2m_N,z_2m_N,Lbc3,z_2m_N],[z_2m_N,z_2m_N,z_2m_N,Lbc4]],format="csc")   
else:
    raise NotImplementedError('Boundary condition type not implemented.')

# Stack the interface operators. Interface condition: Lic*v = 0
Lic12 = bmat([
    [kron(e3,eE),-kron(e3,eW),z_mm2_N,z_mm2_N],
    [kron(e2,eE),-kron(e2,eW),z_mm2_N,z_mm2_N],
    [kron(e3,eES),-kron(e3,eWS),z_1_N,z_1_N],
    [kron(e2,eES),-kron(e2,eWS),z_1_N,z_1_N],
    [kron(e3,eEN),-kron(e3,eWN),z_1_N,z_1_N],
    [kron(e2,eEN),-kron(e2,eWN),z_1_N,z_1_N]
    ],format="csc") 

Lic34 = bmat([
    [z_mm2_N,z_mm2_N,kron(e3,eE),-kron(e3,eW)],
    [z_mm2_N,z_mm2_N,kron(e2,eE),-kron(e2,eW)],
    [z_1_N,z_1_N,kron(e3,eES),-kron(e3,eWS)],
    [z_1_N,z_1_N,kron(e2,eES),-kron(e2,eWS)],
    [z_1_N,z_1_N,kron(e3,eEN),-kron(e3,eWN)],
    [z_1_N,z_1_N,kron(e2,eEN),-kron(e2,eWN)]
    ],format="csc") 

Lic13 = bmat([
    [kron(e1,eN),z_mm2_N,-kron(e1,eS),z_mm2_N],
    [kron(e2,eN),z_mm2_N,-kron(e2,eS),z_mm2_N],
    [kron(e1,eWN),z_1_N,-kron(e1,eWS),z_1_N],
    [kron(e2,eWN),z_1_N,-kron(e2,eWS),z_1_N],
    [kron(e1,eEN),z_1_N,-kron(e1,eES),z_1_N],
    [kron(e2,eEN),z_1_N,-kron(e2,eES),z_1_N]
    ],format="csc") 

Lic24 = bmat([
    [z_mm2_N,kron(e1,eN),z_mm2_N,-kron(e1,eS)],
    [z_mm2_N,kron(e2,eN),z_mm2_N,-kron(e2,eS)],
    [z_1_N,kron(e1,eEN),z_1_N,-kron(e1,eES)],
    [z_1_N,kron(e2,eEN),z_1_N,-kron(e2,eES)],
    [z_1_N,kron(e1,eWN),z_1_N,-kron(e1,eWS)]
    ],format="csc") 

Lic = vstack((Lic12,Lic34,Lic13,Lic24),format="csc")

L = vstack((Lbc,Lic),format="csc")

# Construct RHS using the projection method
P = I_NN - HI_bar_bar@L.T@inv(L@HI_bar_bar@L.T)@L
CI = bmat([
    [kron(inv(C1),I_mtot),z_N_N,z_N_N,z_N_N],
    [z_N_N,kron(inv(C2),I_mtot),z_N_N,z_N_N],
    [z_N_N,z_N_N,kron(inv(C1),I_mtot),z_N_N],
    [z_N_N,z_N_N,z_N_N,kron(inv(C1),I_mtot)]
    ],format="csc")
# D = P@CI@(kron(I4,kron(A,Dx) + kron(B,Dy)))@P

if order == 2 or order == 4 or order == 6:
    D = P@CI@(kron(I_4,kron(A,Dx) + kron(B,Dy)))@P
else:
    # D = P@CI@kron(I_2,kron(A,D1) + kron(I_2,DI))@P
    D = P@CI@(kron(I_4,kron(A,Dx) + kron(B,Dy) + kron(I_3,DIx + DIy)))@P

# Initial data
rstar = 0.1
x0 = -0.5
y0 = 0.5
Ex1 = np.zeros(mtot)
Hz1 = np.exp(-(X1 - x0)**2/rstar**2 - (Y1 - y0)**2/rstar**2).reshape(mtot,order="F")
Ey1 = np.zeros(mtot)

Ex2 = np.zeros(mtot)
Hz2 = np.exp(-(X2 - x0)**2/rstar**2 - (Y2 - y0)**2/rstar**2).reshape(mtot,order="F")
Ey2 = np.zeros(mtot)

Ex3 = np.zeros(mtot)
Hz3 = np.exp(-(X3 - x0)**2/rstar**2 - (Y3 - y0)**2/rstar**2).reshape(mtot,order="F")
Ey3 = np.zeros(mtot)

Ex4 = np.zeros(mtot)
Hz4 = np.exp(-(X4 - x0)**2/rstar**2 - (Y4 - y0)**2/rstar**2).reshape(mtot,order="F")
Ey4 = np.zeros(mtot)

v = np.hstack((Ex1,Hz1,Ey1,Ex2,Hz2,Ey2,Ex3,Hz3,Ey3,Ex4,Hz4,Ey4))

# Time discretization
CFL = 0.1/max(c1,c2)
ht_try = CFL*hx1
mt = int(ceil(T/ht_try) + 1) # round up so that (mt-1)*ht = T
tvec,ht = np.linspace(0,T,mt,retstep=True)

def rhs(v):
    return D@v

def norm(v):
    return sqrt(v.T@H@v)

# Plot
fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8,6))
zlow = -0.2
zhigh = 0.2
plt.xlabel("x")
plt.ylabel("y")
title = plt.title("Magnetic field at t = " + str(0))
srf1 = ax.pcolor(X1,Y1,Hz1.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
srf2 = ax.pcolor(X2,Y2,Hz2.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
srf3 = ax.pcolor(X3,Y3,Hz3.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
srf4 = ax.pcolor(X4,Y4,Hz4.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
fig.colorbar(srf4, ax=ax)
fig.tight_layout()
plt.pause(0.5)
plt.draw()

# Runge-Kutta 4
div = np.zeros(mt)
t = 0
for tidx in range(mt-1):
    k1 = ht*rhs(v)
    k2 = ht*rhs(v + 0.5*k1)
    k3 = ht*rhs(v + 0.5*k2)
    k4 = ht*rhs(v + k3)

    v = v + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + ht

    Ex1 = v[0:mtot]
    Hz1 = v[mtot:2*mtot]
    Ey1 = v[2*mtot:3*mtot]
    Ex2 = v[3*mtot:4*mtot]
    Hz2 = v[4*mtot:5*mtot]
    Ey2 = v[5*mtot:6*mtot]
    Ex3 = v[6*mtot:7*mtot]
    Hz3 = v[7*mtot:8*mtot]
    Ey3 = v[8*mtot:9*mtot]
    Ex4 = v[9*mtot:10*mtot]
    Hz4 = v[10*mtot:11*mtot]
    Ey4 = v[11*mtot:12*mtot]

    div[tidx+1] = norm(Dx@Ex1 + Dy@Ey1) + norm(Dx@Ex2 + Dy@Ey2) + norm(Dx@Ex3 + Dy@Ey3) + norm(Dx@Ex4 + Dy@Ey4)

    # Update plot every Xth time step
    if tidx % ceil(20*max(c2,c1)) == 0 or tidx == mt-2: 
        srf1.remove()
        srf2.remove()
        srf3.remove()
        srf4.remove()

        srf1 = ax.pcolor(X1,Y1,Hz1.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
        srf2 = ax.pcolor(X2,Y2,Hz2.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
        srf3 = ax.pcolor(X3,Y3,Hz3.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
        srf4 = ax.pcolor(X4,Y4,Hz4.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)

        title.set_text("Magnetic field at t = {:.2f}".format(t))
        plt.draw()
        plt.pause(1e-6)

# Norm of divergence over time
plt.figure()
plt.plot(tvec,div)
plt.title("Norm of divergence")
plt.xlabel("t")
plt.ylabel("||div||")

# Final divergence plot
fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8,6))
div1 = Dx@Ex1 + Dy@Ey1
div2 = Dx@Ex2 + Dy@Ey2
div3 = Dx@Ex3 + Dy@Ey3
div4 = Dx@Ex4 + Dy@Ey4
zlow = np.min(np.stack((div1,div2,div3,div4)))
zhigh = np.max(np.stack((div1,div2,div3,div4)))
plt.xlabel("x")
plt.ylabel("y")
title = plt.title("Divergence at t = {:.2f}".format(t))
srf1 = ax.pcolor(X1,Y1,div1.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
srf2 = ax.pcolor(X2,Y2,div2.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
srf3 = ax.pcolor(X3,Y3,div3.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
srf4 = ax.pcolor(X4,Y4,div4.reshape((m,m),order="F"),shading="nearest",vmin=zlow,vmax=zhigh)
fig.colorbar(srf4, ax=ax)

plt.show()
