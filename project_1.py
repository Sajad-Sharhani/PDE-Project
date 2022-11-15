#!/usr/bin/env python3
from cmath import log
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import eye, vstack, kron, coo_matrix
import operators as ops
import matplotlib.pyplot as plt

# Method parameters
# Number of grid points, integer > 15.
mx = 200
n = 2*mx


# Model parameters
c = 1.  # for simplicity
alphal = 0.
alphar = 0.
betal = 1  # Dirichlet BC if it's the only one set to non-zero
betar = 1  # Dirichlet BC if it's the only one set to non-zero
gammal = 0  # Neumann BC if it's the only one set to non-zero
gammar = 0  # Neumann BC if it's the only one set to non-zero

T = 1.8  # end time
xl = -1.
xr = 1.

# Analytic solutions


def theta1(x, t):
    return np.exp(-((x-c*t)/0.2)**2)


def theta2(x, t):
    return -np.exp(-((x+c*t)/0.2)**2)


def analytic_N(x, t):  # valid after the Gaussian pulses have been reflected
    return 0.5*theta1(x+2, t) - 0.5*theta2(x-2, t)


def analytic_D(x, t):  # valid after the Gaussian pulses have been reflected
    return -0.5*theta1(x+2, t) + 0.5*theta2(x-2, t)


# Initial data
def f(x):
    #    return np.exp(-(5*x)**2)
    return theta1(x, 0)


# Space discretization
hx = (xr - xl)/(mx-1)
xvec = np.linspace(xl, xr, mx)
e1 = np.array([1, 0])
e2 = np.array([0, 1])

I2 = eye(2)
I = eye(n)

e1 = np.array([1, 0])
e2 = np.array([0, 1])


# The 4th order SBP operator
H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(mx, hx)

# Construct the Boundary Operators
L = vstack([alphal*kron(e2, e_l)+betal*kron(e1, e_l)+gammal*kron(e1, d1_l),
            alphar*kron(e2, e_r)+betar*kron(e1, e_r)+gammar*kron(e1, d1_r)])


# Construct the Projection Operator
HII = kron(I2, HI)
P = I - HII@L.T@inv(L@HII@L.T)@L

# Solution Matrix
A = np.block([
    [np.zeros((mx, mx)), np.eye(mx)],
    [coo_matrix(c**2*D2).toarray(), np.zeros((mx, mx))]
])
B = P @ A @ P

# Numerical Solution
V = np.zeros((n, 1))

# Initial conditons
V[0:mx] = f(xvec).reshape(mx, 1)
#V[0:mx] = theta1(xvec, 0).reshape(mx, 1)

# Time discretization
k_try = 0.0001
mt = int(np.ceil(T/k_try) + 1)  # round up so that (mt-1)*k = T
print("mt = ", mt)
tvec, ht = np.linspace(0, T, mt, retstep=True)
print("ht = ", ht)


# Runge-Kutta 4
t = 0
for tidx in range(mt):

    k1 = ht*B@V
    k2 = ht*B@(V + 0.5*k1)
    k3 = ht*B@(V + 0.5*k2)
    k4 = ht*B@(V + k3)

    V = V + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + ht
    if tidx % 800 == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xvec, V[0:mx], label='Approximation')
        # ax.plot(xvec, analytic_D(xvec, t),
        #         label='Analytic solution', linestyle='--')
        error = (1/np.sqrt(mx)) * \
            np.sqrt(np.sum((analytic_D(xvec, t).reshape(mx, 1) - V[0:mx])**2))
        print('Error at t = ', t, ': ', error)
        # print(analytic_D(xvec, t).shape)
        # print(V[0:mx].shape)
        error1 = np.sqrt(hx) * np.linalg.norm(analytic_D(xvec, t) - V[0:mx])
        ax.set_title('t = ' + str(t) + ', error = ' + str(error))
        ax.plot(xvec, analytic_D(xvec, t),
                label='Analytic solution', linestyle='--')
        plt.xlabel('x')
        plt.ylabel('u', rotation=0)
        ax.set_xlim([xl, xr-hx])
        ax.set_ylim([-1, 1])
        # title = plt.title("t = " + "{:.2f}".format(t))
        plt.grid(linewidth=0.4)
        plt.legend()
        plt.show()
        # plt.pause(0.5)
        # plt.close()


# convergence rate for the error
# l2 norm
# tt = 1.8


# q = log(0.01399/0.00359)/log(101/51)
# print('Convergence rate: ', q)
