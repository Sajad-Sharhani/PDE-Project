import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as la

# Finite Element Method Solver

# stiffness matrix assembler
# The input is a vector x of node coordinates
# The output is the stiffness matrix A

T = 10


def stiffness(x):
    n = len(x) - 1
    A = np.zeros((n+1, n+1))
    for i in range(n):
        h = x[i+1] - x[i]
        A[i, i] = 1/h
        A[i+1, i+1] = 1/h
        A[i, i+1] = -1/h
        A[i+1, i] = -1/h

    # adjust for boundary conditions A[0,0] = 1.e+6
    A[0, 0] = 1.e+6
    A[n, n] = 1.e+6
    return A

# load vector assembler
# The input is a vector x of node coordinates
# The output is the load vector b


def f(x): return np.exp(-x**2)


def load(x):
    n = len(x) - 1
    b = np.zeros(n+1)
    for i in range(n):
        h = x[i+1] - x[i]
        b[i] = f(x[i])*h/2
        b[i+1] = f(x[i+1])*h/2
    return b

# FEM solver


def solve(x):
    A = stiffness(x)
    b = load(x)
    u = la.solve(A, b)
    return u

# RK4 time-stepping scheme


# def rk4(u, x, t, dt):
#     k1 = dt*solve(x)
#     k2 = dt*solve(x + 0.5*dt)
#     k3 = dt*solve(x + 0.5*dt)
#     k4 = dt*solve(x + dt)
#     u = u + (k1 + 2*k2 + 2*k3 + k4)/6
#     return u

# loop with RK4 time-stepping scheme


xvec = np.linspace(-1, 1, 101)
x = f(xvec)
k_try = 0.01
mt = int(np.ceil(T/k_try) + 1)  # round up so that (mt-1)*k = T
tvec, ht = np.linspace(0, T, mt, retstep=True)
fig = plt.figure()
plt.tight_layout(rect=[0, 0, 0.75, 1])
ax = fig.add_subplot(111)
[line] = ax.plot(xvec, x[0:101], label='Numerical solution at t='+str(T))
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xlabel("x")
ax.set_ylabel("u")
title = plt.title("t = " + "{:.2f}".format(0))
plt.grid()
plt.draw()
plt.pause(1)

t = 0
for i in range(101):
    k1 = ht*solve(x)
    k2 = ht*solve(x + 0.5*k1)
    k3 = ht*solve(x + 0.5*k2)
    k4 = ht*solve(x + k3)
    x = x + (k1 + 2*k2 + 2*k3 + k4)/6

    t = t + ht
    # plot every 10th step
    if i % 20 == 0:
        line.set_ydata(x[0:101])
        title.set_text("t = " + "{:.2f}".format(t))
        plt.draw()
        plt.pause(1e-8)


# plot the solution at the final time
# plt.plot(xvec, x)
# plt.show()


# # main program
# # set up the grid
# n = 100
# xl = -5.0
# xr = 5.0
# x = np.linspace(xl, xr, n+1)
# # set up the time grid
# t0 = 0.0
# tf = 1.0
# dt = 0.01
# t = np.arange(t0, tf+dt, dt)
# # solve the problem
# u = loop(x, t, dt)
# # plot the solution
# plt.plot(x, u)
# plt.show()


# x = np.linspace(0, 1, 101)
# u = solve(x)
# # plot the solution
# plt.plot(x, u, 'o-')
# plt.show()
