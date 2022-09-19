#!/usr/bin/env python3
import numpy as np
from scipy.sparse.linalg import inv, eigs
from scipy.sparse import eye, vstack, kron
import operators as ops
import matplotlib.pyplot as plt

# Method parameters
# Number of grid points, integer > 15.
m = 101
n = 2*m

# Initial data


def f(x):
    return np.exp(-(6*x)**2)


# Model parameters
T = 1  # end time
xl = -1
xr = 1

# Space discretization
hx = (xr - xl)/(m-1)
xvec = np.linspace(xl, xr, m)

A = np.array([[2, 1], [1, 0]])
I2 = eye(2)
I = eye(n)
e1 = np.array([1, 0])
e2 = np.array([0, 1])


# 6th order SBP operator
H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(m, hx)

# Construct the Boundary Operators
L = vstack([kron(e1, e_l), kron(e1+e2, e_r)])
HII = kron(I2, HI)

# Construct the Projection Operator
P = I - HII@L.T@inv(L@HII@L.T)@L

# Solution Matrix
B = P @ (kron(A, D1)) @ P

# Numerical Solution
V = np.zeros((n, 1))

V[0:m] = f(xvec).reshape(m, 1)

# Time discretization
ht_try = 0.1*hx
mt = int(np.ceil(T/ht_try) + 1)  # round up so that (mt-1)*ht = T
tvec, ht = np.linspace(0, T, mt, retstep=True)


# Runge-Kutta 4
t = 0
for tidx in range(mt-1):
    k1 = ht*B@V
    k2 = ht*B@(V + 0.5*k1)
    k3 = ht*B@(V + 0.5*k2)
    k4 = ht*B@(V + k3)

    V = V + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + ht

    if tidx % 150 == 0:
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        ax.plot(xvec, V[0:m], label='Approximation')
        ax.plot(xvec, V[m:n], label='Approximation')

        ax.set_xlim([xl, xr-hx])
        ax.set_ylim([-1, 1])
        title = plt.title("t = " + "{:.2f}".format(t))
        plt.show()

fig = plt.figure(1)
ee, vv = np.linalg.eig(hx*B.toarray())

# plot all the real and imaginary parts of the eigenvalues
plt.plot(np.real(ee), np.imag(ee), 'x')
plt.show()
