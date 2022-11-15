#!/usr/bin/env python3
import numpy as np
from scipy.sparse import kron, csc_matrix, eye, vstack
from scipy.sparse.linalg import inv
from math import sqrt, ceil
import operators as ops
import matplotlib.pyplot as plt

# Method parameters
# Number of grid points, integer > 15
mx = 201

# Order of accuracy: 2, 3, 4, 5, 6, or 7. Odd orders are upwind operators
order = 4

# Type of boundary condition.
# 1 - Electric field at both boundaries
# 2 - Magnetic field at both boundaries
# 3 - Linear combination E + beta*H and E - beta*H at left and right boundary respectively
# 4 - Both electric and magnetic field at both boundaries
bc_type = 1

# If bc_type 3, specify beta
beta = 1

# Model parameters
T = 1.8  # end time

# Domain boundaries
xl = -1
xr = 1

# Material properties


# Speed of light


# Space discretization
N = 2*mx  # number of degrees of freedom

hx = (xr - xl)/(mx-1)
xvec = np.linspace(xl, xr, mx)

if order == 2:
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_2nd(mx, hx)
elif order == 3:
    H, HI, Dp, Dm, e_l, e_r = ops.sbp_upwind_3rd(mx, hx)
elif order == 4:
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_4th(mx, hx)
elif order == 5:
    H, HI, Dp, Dm, e_l, e_r = ops.sbp_upwind_5th(mx, hx)
elif order == 6:
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(mx, hx)
elif order == 7:
    H, HI, Dp, Dm, e_l, e_r = ops.sbp_upwind_7th(mx, hx)
else:
    raise NotImplementedError('Order not implemented.')

e1 = csc_matrix([1, 0])
e2 = csc_matrix([0, 1])
I_N = eye(N)
I_m = eye(mx)
I_2 = eye(2)
H_bar = kron(I_2, H)
HI_bar = kron(I_2, HI)


def norm(v):
    return (1/np.sqrt(hx))*np.sqrt(np.sum(v**2))


if bc_type == 1:
    L = vstack((kron(e1, e_l), kron(e1, e_r)), format="csc")
elif bc_type == 2:
    L = vstack((kron(e2, e_l), kron(e2, e_r)), format="csc")
elif bc_type == 3:
    L = vstack((kron(e1 + beta*e2, e_l), kron(e1 - beta*e2, e_r)), format="csc")
elif bc_type == 4:
    L = vstack((kron(e1, e_l), kron(e1, e_r), kron(
        e2, e_l), kron(e2, e_r)), format="csc")
else:
    raise NotImplementedError('Boundary condition type not implemented.')

# Construct RHS matrix using the projection method
P = I_N - HI_bar@L.T@inv(L@HI_bar@L.T)@L


# Initial data
# def gauss(x):
#     rstar = 0.1
#     return np.exp(-(x/rstar)**2)

def gaussl(x, t):
    rstar = 0.1
    return np.exp(-((x - t)/rstar)**2)


def gaussr(x, t):
    rstar = 0.1
    return -np.exp(-((x + t)/rstar)**2)


def exactNeumann(x, t):
    return 0.5*gaussl(x+2, t) - 0.5*gaussr(x-2, t)


def exactDirichlet(x, t):
    return -0.5*gaussl(x+2, t) + 0.5*gaussr(x-2, t)


Ey = -gauss(xvec) - gauss(xvec)
Hz = -gauss(xvec) + gauss(xvec)
v = np.hstack((Ey, Hz))

# Time discretization
CFL = 0.1/c
ht_try = CFL*hx
mt = int(ceil(T/ht_try) + 1)  # round up so that (mt-1)*ht = T
tvec, ht = np.linspace(0, T, mt, retstep=True)


def rhs(v):
    return D@v


# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
Ey = v[0:mx]
Hz = v[mx:]
[line1] = ax.plot(xvec, Ey, label='Electric field')
[line2] = ax.plot(xvec, Hz, label='Magnetic field')
plt.legend()
ax.set_xlim([xl, xr])
ax.set_ylim([-2, 2])
title = plt.title("t = " + "{:.2f}".format(0))
plt.draw()
plt.pause(0.5)

# Runge-Kutta 4 for 1st order system of ODEs
# u' = v
# v' = rhs(u)
for i in range(1, mt):
    m1 = ht*v
    k1 = ht*rhs(v)
    m2 = ht*(v + 0.5*k1)
    k2 = ht*rhs(v + 0.5*k1)
    m3 = ht*(v + 0.5*k2)
    k3 = ht*rhs(v + 0.5*k2)
    m4 = ht*(v + k3)
    k4 = ht*rhs(v + k3)
    v = v + (m1 + 2*m2 + 2*m3 + m4)/6
    v = P@v


# Error for bc_type 1
if bc_type == 1:
    tt = 2 - T
    Ey_exact = gauss(xvec - tt) + gauss(xvec + tt)
    Hz_exact = gauss(xvec - tt) - gauss(xvec + tt)
    Ey = v[0:mx]
    Hz = v[mx:]
    error = norm(Ey_exact - Ey) + norm(Hz_exact - Hz)
    print("Error: " + "{:.4e}".format(error))

plt.show()
