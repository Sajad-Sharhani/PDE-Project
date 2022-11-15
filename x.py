"""
Created on Fri Sep 16 14:24:53 2022

@author: Jannes
Title: Project 1 Scientific Computing for PDEs, 1D Wave equation

The code is inspired by the work of Gustav Eriksson
as provided in laboration 1 on the course page.
"""

import numpy as np
from scipy.sparse import kron, csc_matrix, eye, vstack, bmat
from scipy.sparse.linalg import inv
import operators as ops
import matplotlib.pyplot as plt
#plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Set parameters
T = 1.8
r = 0.2
# Set BC=1 for Dirichlet BC, BC=2 for Neumann BC,
# BC=3 for mixed BC, BC=4 for non-reflective BC
BC = 1
c = 1
order = 6  # choose order of the operators, between 2, 4 and 6

# Boundary points
xr = 1
xl = -1

# Space+time discretization & Number of grid points
m = 301  # for convergence study use m=51, 101, 201, 301
h = (xr-xl)/(m-1)
# k=0.01;
k = 0.1*h  # for stable solutions in the convergence study
mt = int(np.ceil(T/k)+1)
xvector = np.linspace(xl, xr, m)
tvector, ht = np.linspace(0, T, mt, retstep=True)
# tlist=[0.2,0.5,0.7,1.8]
tlist = []

# Get SBP/relevant operators from library
if order == 2:
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_2nd(m, h)

if order == 4:
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_4th(m, h)

if order == 6:
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(m, h)

e1 = csc_matrix([1, 0])
e2 = csc_matrix([0, 1])
H_bar = kron(eye(2), H)
HI_bar = kron(eye(2), HI)
zeros = csc_matrix((m, m))
D2_b = bmat([[zeros, eye(m)], [c**2*D2, zeros]])

# Define gaussians, initial profile, BC and construct projection


def gaussianr(x, t, c):
    return np.exp(-((x-c*t)/r)**2)


def gaussianl(x, t, c):
    return -np.exp(-((x+c*t)/r)**2)


def f(x, t, c):
    return gaussianr(x, t, c)


if BC == 1:  # DIRICHLET
    L = vstack([kron(e1, e_l), kron(e1, e_r)], format="csc").T

    def sol(x, t, c):
        return -(1/2)*gaussianr(x+2, t, c)+(1/2)*gaussianl(x-2, t, c)

if BC == 2:  # NEUMANN
    L = vstack([kron(e1, d1_l), kron(e1, d1_r)], format="csc").T

    def sol(x, t, c):
        return (1/2)*gaussianr(x+2, t, c)-(1/2)*gaussianl(x-2, t, c)

if BC == 3:  # MIXED
    L = vstack([kron(e1, d1_l), kron(e1, e_r)], format="csc").T


if BC == 4:  # NONREFLECTIVE
    L = vstack((kron(-c*e1, d1_l)+kron(e2, e_l),
                kron(c*e1, d1_r)+kron(e2, e_r)), format="csc").T

P = eye(2*m) - HI_bar @ L @ inv(L.T @ HI_bar @ L) @ L.T


# $l_2$ norm for error/convergence analysis
def norm(v):
    return (1/np.sqrt(m))*np.sqrt(np.sum(v**2))


# Define right hand site of system of ODE
def rhs(v):
    return P @ D2_b @ P @ v


w = np.hstack([f(xvector, 0, c), np.zeros(m)])

# Set up plot
fig = plt.figure()
plt.tight_layout(rect=[0, 0, 0.75, 1])
ax = fig.add_subplot(111)
[line] = ax.plot(xvector, w[0:m], label='Numerical solution at t='+str(T))
ax.set_xlim([xl, xr])
ax.set_ylim([-1, 1])
ax.set_xlabel("x")
ax.set_ylabel("u")
title = plt.title("t = " + "{:.2f}".format(0))
plt.grid()
plt.draw()
plt.pause(1)

#RK4 & Plot
t = 0
for tidx in range(mt):
    k1 = ht*rhs(w)
    k2 = ht*rhs(w + 0.5*k1)
    k3 = ht*rhs(w + 0.5*k2)
    k4 = ht*rhs(w + k3)

    w = w + 1/6*(k1 + 2*k2 + 2*k3 + k4)

    if round(tidx*k, 3) in tlist:
        ax.plot(xvector, w[0:m], '--',
                label='Numerical solution at t=' + str(round(tidx*k, 3)))

    t = t + ht

    if tidx % 20 == 0:
        line.set_ydata(w[0:m])
        title.set_text("t = " + "{:.2f}".format(tvector[tidx]))
        plt.draw()
        plt.pause(1e-8)

# Compute l2 error & Plot final profile
if BC == 1:
    solvector = sol(xvector, T, c)
    print(norm(solvector-w[0:m]))
    title.set_text("Solutions to wave equations at various timesteps")
    ax.plot(xvector, solvector, '--', label='Exact solution at t='+str(T))
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.draw()
    #plt.savefig("sol_dirichlet.png", bbox_inches="tight")
    plt.show()
if BC == 2:
    solvector = sol(xvector, T, c)
    print(norm(solvector-w[0:m]))
    title.set_text("Solutions to wave equations at various timesteps")
    ax.plot(xvector, solvector, label='Exact solution at t=' + str(T))
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.draw()
    #plt.savefig("sol_neumann.png", bbox_inches="tight")
    plt.show()

if BC == 4:
    title.set_text("Solutions to wave equations at various timesteps")
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.draw()
    #plt.savefig("sol_nonrefl.png", bbox_inches="tight")
    plt.show()
