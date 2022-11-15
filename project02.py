import math
from re import M
from tkinter import N
import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as la
# import rk4 from scipy
from scipy.integrate import odeint, solve_ivp


# Finite Element Method Solver

# mass matrix assembler
# The input is a vector x of node coordinates
# The output is the mass matrix M

def mass(n, h):
    M = np.zeros((n+1, n+1))
    for i in range(n):

        M[i, i] = 2*h/3
        M[i+1, i+1] = 2*h/3
        M[i, i+1] = h/6
        M[i+1, i] = h/6
    return M


# advection matrix assembler
# The input is a vector x of node coordinates
# The output is the advection matrix A

def advection(n):
    A = np.zeros((n+1, n+1))
    for i in range(n):
        # h = x[i+1] - x[i]
        A[i, i] = -1/2
        A[i+1, i+1] = 1/2
        A[i, i+1] = 1/2
        A[i+1, i] = -1/2

    return A


# diffusion matrix assembler
# The input is a vector x of node coordinates
# The output is the diffusion matrix S

def diffusion(n, h):
    S = np.zeros((n+1, n+1))
    for i in range(n):
        S[i, i] = 2/h
        S[i+1, i+1] = 2/h
        S[i, i+1] = -1/h
        S[i+1, i] = -1/h

    return S


a = -1.
b = 1.
c = 2.
epsilon = 0.1
def belta(U): return 1/2*U**2


belta = np.vectorize(belta)

T = 0.4
ht = 0.001
N_list = [5*2**i for i in range(5)]


def u_0(x): return c - math.tanh((x + 1/2)/(2*epsilon))


u_0 = np.vectorize(u_0)
def u_exact(x, t): return c - math.tanh((x + 1/2 - c*t)/(2*epsilon))
def g_a(t): return u_exact(a, t)
def g_b(t): return u_exact(b, t)


def fun(U_vec):
    return np.linalg.inv(M)@(-A@belta(U_vec) - epsilon*S@U_vec)


if u_exact != None:
    err_list = []
    h_list = []

for N in N_list:
    h = (b-a) / N
    x_vec = np.linspace(a, b, N+1)
    U_vec = u_0(x_vec).reshape(N+1, 1)
    M = mass(N, h)
    A = advection(N)
    S = diffusion(N, h)

    Nt = int(T / ht)
    t = 0.

    # Runge-Kutta 4
    for _ in range(Nt):
        k1 = fun(U_vec)
        k2 = fun(U_vec + 0.5*ht*k1)
        k3 = fun(U_vec + 0.5*ht*k2)
        k4 = fun(U_vec + ht*k3)
        U_vec = U_vec + 1/6*(k1 + 2*k2 + 2*k3 + k4)*ht
        t = t + ht
        U_vec[0] = g_a(t)
        U_vec[N] = g_b(t)

    if u_exact != None:
        # now t may be different from T. To be modified
        def solution_exact(x): return u_exact(x, t)
        solution_exact = np.vectorize(solution_exact)
        err = 1 / \
            math.sqrt(
                N+1) * math.sqrt(sum((solution_exact(x_vec)-U_vec.reshape(N+1))**2))
        h_list.append(h)
        err_list.append(err)

    plt.figure()
    plt.title("t = " + "{:.2f}".format(t) + ", N = " +
              str(N) + ", epsilon = " + "{:.5f}".format(epsilon))
    plt.plot(x_vec, U_vec, label='approximation')
    if u_exact != None:
        x_vec_exact = np.linspace(a, b, 1000+1)
        plt.plot(x_vec_exact, solution_exact(
            x_vec_exact), label='exact solution')
    plt.legend()
    plt.show()

if u_exact != None:
    plt.plot(h_list, err_list)
    plt.xlabel('h')
    plt.ylabel('l2 norm', rotation=0)
    plt.title("the error as a function of mesh-size")
    q = [math.log10(err_list[i]/err_list[i+1])/math.log10((N_list[i+1]+1) /
                                                          (N_list[i]+1)) for i in range(len(N_list)-1)]
    print("the convergence rate q is " + str(q))


# M = diffusion(np.linspace(0, 1, 100))
# # spy plot of the advection matrix
# plt.spy(M)
# plt.show()


# xl = -1
# xr = 1
# T = 0.4
# c = 2
# epsilon = 0.1
# N = 41
# # h = 1/(N-1)

# xvec = np.linspace(xl, xr, N+1)
# k_try = 0.01
# mt = int(np.ceil(T/k_try) + 1)  # round up so that (mt-1)*k = T
# tvec, ht = np.linspace(0, T, mt, retstep=True)
# u0 = c - np.tanh((xvec+1/2)/(2*epsilon))
# # print(u0)
# plt.plot(xvec, u0)
# plt.show()
# uexact = c - np.tanh((xvec - c*ht + 1/2)/(2*epsilon))
# plt.plot(xvec, uexact)
# plt.show()

# M = mass(xvec)
# A = advection(xvec)
# S = diffusion(xvec)
# M[0, 0] = 1.e+6
# M[N, N] = 1.e+6

# # du/dt = M^{-1}A (1/2 * u^2) + M^{-1} epsilon S u

# # RK4 using scipy


# def f(u, t):
#     return np.dot(la.inv(M), np.dot(A, 0.5 * u**2) - np.dot(S, epsilon * u))


# # print("sajad", f(u0, 0))
# # solve with RK4
# # u = solve_ivp(f, [0, T], u0, method='RK45', t_eval=tvec)

# u = odeint(f, u0, tvec)
# # plot the solution as a moving wave
# for i in range(mt):
#     plt.plot(xvec, u[i, :])
# plt.show()


# t = 0
# dt = 0.01
# u = u0
# for i in range(int(T/dt)):
#     k1 = ht*(la.solve(M, A @ ((u)/2) + epsilon * S @ u))
#     k2 = ht*(la.solve(M, A @ (((u + k1/2))/2) + epsilon * S @ (u + k1/2)))
#     k3 = ht*(la.solve(M, A @ (((u + k2/2))/2) + epsilon * S @ (u + k2/2)))
#     k4 = ht*(la.solve(M, A @ (((u + k3))/2) + epsilon * S @ (u + k3)))
#     u = u + (k1 + 2*k2 + 2*k3 + k4)/6
#     t = t + ht


# plt.plot(xvec, u, label='numerical')
# plt.plot(xvec, uexact, label='exact')
# plt.legend()
# plt.show()


# a = -1
# b = 1
# T = 0.4
# c = 2
# eps = 0.1
# Nlist = 41

# mt = 200
# t, ht = np.linspace(0, T, mt+1, retstep=True)
# x = np.linspace(a, b, Nlist)
# M = mass(x)
# A = advection(x)
# S = diffusion(x)


# def uexact1(x, t, c, eps):
#     return c - np.tanh((x - c*t + 1/2)/2*eps)


# def u2(x):
#     return np.sin(x)


# def ga(t):
#     return uexact1(a, t, c, eps)


# def gb(t):
#     return uexact1(b, t, c, eps)


# def f(U):
#     return (-eps*S@U - 1/2*A@(U**2))


# M[0, 0] = 1
# M[0, 1] = 0
# M[Nlist-1, Nlist-1] = 1
# M[Nlist-1, Nlist-2] = 0

# U = uexact1(x, 0, c, eps)
# tind = 0
# for i in range(mt):
#     k1 = ht*(la.solve(M, f(U)))
#     k2 = ht*(la.solve(M, f(U + k1/2)))
#     k3 = ht*(la.solve(M, f(U + k2/2)))
#     k4 = ht*(la.solve(M, f(U + k3)))
#     U = U + (k1 + 2*k2 + 2*k3 + k4)/6
#     tind = tind + 1
#     if tind % 10 == 0:
#         plt.plot(x, U, label='t = ' + str(tind*ht))
#         plt.legend()
#         plt.show()
