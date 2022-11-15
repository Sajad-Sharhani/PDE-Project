#!/usr/bin/env python3
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import eye, vstack
from Problem_Solving2 import B
import operators as ops
import matplotlib.pyplot as plt

# Method parameters
# Number of grid points, integer > 15.
mx = 101

# Initial data


def f(x):
    return np.exp(-(6*x)**2)


# Model parameters
a = 1
b = 0.1
T = 1.5  # end time
xl = -1
xr = 1

# Space discretization
hx = (xr - xl)/(mx-1)
xvec = np.linspace(xl, xr, mx)

# The 4th order SBP operator
H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_4th(mx, hx)

# Construct the Boundary Operators
L = vstack([e_l,  e_r])
I = eye(mx)

# Construct the Projection Operator
P = I - HI@L.T@inv(L@HI@L.T)@L


# Time discretization
ht_try = 0.1*hx
mt = int(np.ceil(T/ht_try) + 1)  # round up so that (mt-1)*ht = T
tvec, ht = np.linspace(0, T, mt, retstep=True)

# The SBP-Projection Method
# imaginary unit
j = 1j

# F function is alpha*exp(-(6*(x-1))^2)


def F(v):
    return a*np.exp(-(6*(xvec-1))**2)


def rhs(v):
    return (P@(j*D2 - j*F(xvec))@P@v)


v = f(xvec)
B = P@D2@P@v
# convert B to a matrix
B = B.toarray()
print(B.shape)

fig = plt.figure(1)
ee, vv = np.linalg.eig(hx*B)

# plot all the real and imaginary parts of the eigenvalues
plt.plot(np.real(ee), np.imag(ee), 'x')
plt.show()


# # Plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# v = f(xvec)
# [line] = ax.plot(xvec, v, label='Approximation')
# ax.set_xlim([xl, xr-hx])
# ax.set_ylim([0, 1])
# title = plt.title("t = " + "{:.2f}".format(0))
# plt.draw()
# plt.pause(1)

# # Runge-Kutta 4
# t = 0
# for tidx in range(mt-1):
#     k1 = ht*rhs(v)
#     k2 = ht*rhs(v + 0.5*k1)
#     k3 = ht*rhs(v + 0.5*k2)
#     k4 = ht*rhs(v + k3)

#     v = v + 1/6*(k1 + 2*k2 + 2*k3 + k4)
#     t = t + ht

#     # Plot for the 0.5 second mark and save the figure
#     if tidx == 250:
#         ax.set_ylim([0, 0.35])
#         line.set_ydata(v)
#         title.set_text("t = " + "{:.2f}".format(t))
#         plt.draw()
#         plt.pause(1)
#         plt.savefig("0.5s.png")

#      # Plot for the 1.5 second mark and save the figure
#     if tidx == 749:
#         ax.set_ylim([0, 0.08])
#         line.set_ydata(v)
#         title.set_text("t = " + "{:.2f}".format(t))
#         plt.draw()
#         plt.pause(1)
#         plt.savefig("1.5s.png")
