#!/usr/bin/env python3
from math import fabs
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import eye, vstack, kron, coo_matrix, csc_matrix
import operators as ops
import matplotlib.pyplot as plt

# Method parameters
# Number of grid points, integer > 15.
m = 101
n = 2*m


# Model parameters
c = 1.  # for simplicity


T = 11  # end time
xl = -1.
xr = 1.
yt = 1.
yb = -1.


# Initial data
def f(x, y):
    return np.exp(-100*(x**2 + y**2))


# Space discretization
h = (xr - xl) / (m-1)
xvec = np.linspace(xl, xr, m)
yvec = np.linspace(yb, yt, m)
X, Y = np.meshgrid(xvec, yvec)
e1 = np.array([1, 0])
e2 = np.array([0, 1])

I2 = eye(2)
Im = eye(m, dtype='uint8')


# The 4th order SBP operator
H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_4th(m, h)

# Construct the Boundary Operators
# Dirichlet BC
# L = vstack([e_l, e_r])

# Neumann BC
L = vstack([d1_l, d1_r])

# Construct the Projection Operator

P = Im - HI@L.T@inv(coo_matrix(L@HI@L.T).tocsc())@L

# Solution Matrix
A = c**2*P@D2@P

B = np.block([
    [csc_matrix(np.zeros((m*m, m*m), dtype='uint8')).toarray(),
     np.eye(m*m, dtype='uint8')],
    [coo_matrix(kron(A, Im)+kron(Im, A)).toarray(),
     csc_matrix((np.zeros((m*m, m*m), dtype='uint8'))).toarray()]
])
B = coo_matrix(B)
print(B)

# Numerical Solution
V = np.zeros((2*m**2, 1))

# Initial conditons
for i in range(1, m+1):
    V[m*(i-1):m*i] = f(xvec[i-1], yvec).reshape(m, 1)


# Time discretization
k_try = 0.01  # !!!
mt = int(np.ceil(T/k_try) + 1)  # round up so that (mt-1)*k = T
print("sdsdsdfa", mt)
tvec, ht = np.linspace(0, T, mt, retstep=True)
# print(ht)

# Runge-Kutta 4
t = 0
for tidx in range(mt-1):
    k1 = ht*B@V
    k2 = ht*B@(V + 0.5*k1)
    k3 = ht*B@(V + 0.5*k2)
    k4 = ht*B@(V + k3)

    V = V + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + ht

    if tidx % 50 == 0:
        print(tidx)
        V1 = V[0:m**2].reshape(m, m)
        V2 = V[m**2:2*m**2].reshape(m, m)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, V1, cmap='viridis', linewidth=0)
        ax.set_zlim(0, 1)
        ax.text2D(0.05, 0.95, "t = %f" % t, transform=ax.transAxes)
        plt.show()
        # plt.pause(0.5)
        # plt.close()
