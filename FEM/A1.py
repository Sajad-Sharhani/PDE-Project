import math
import numpy as np
from matplotlib import pyplot as plt

a = -1.
b = 1.
c = 2.
epsilon = 0.1
def belta(U): return 1/2*U**2


belta = np.vectorize(belta)

T = 0.4
ht = 0.001
N_list = [10, 20, 40, 80]


def u_0(x): return c - math.tanh((x + 1/2)/(2*epsilon))


u_0 = np.vectorize(u_0)
def u_exact(x, t): return c - math.tanh((x + 1/2 - c*t)/(2*epsilon))
def g_a(t): return u_exact(a, t)
def g_b(t): return u_exact(b, t)


def matrix_M(N, h):
    M = np.zeros((N+1, N+1))
    M[0][:2] = np.array([1/3, 1/6])
    for i in range(1, N):
        M[i][i-1:i+2] = np.array([1/6, 2/3, 1/6])
    M[N][-2:] = np.array([1/6, 1/3])
    return h*M


def matrix_A(N):
    A = np.zeros((N+1, N+1))
    A[0][:2] = np.array([-1/2, 1/2])
    for i in range(1, N):
        A[i][i-1:i+2] = np.array([-1/2, 0, 1/2])
    A[N][-2:] = np.array([-1/2, 1/2])
    return A


def matrix_S(N, h):
    S = np.zeros((N+1, N+1))
    S[0][:2] = np.array([1, -1])
    for i in range(1, N):
        S[i][i-1:i+2] = np.array([-1, 2, -1])
    S[N][-2:] = np.array([-1, 1])
    return 1 / h * S


def fun(U_vec):
    return np.linalg.inv(M)@(-A@belta(U_vec) - epsilon*S@U_vec)


if u_exact is not None:
    err_list = []
    h_list = []

for N in N_list:
    h = (b-a) / N
    x_vec = np.linspace(a, b, N+1)
    U_vec = u_0(x_vec).reshape(N+1, 1)
    M = matrix_M(N, h)
    A = matrix_A(N)
    S = matrix_S(N, h)

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

    if u_exact is not None:
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
    if u_exact is not None:
        x_vec_exact = np.linspace(a, b, 1000+1)
        plt.plot(x_vec_exact, solution_exact(
            x_vec_exact), label='exact solution')
    plt.legend()
    plt.show()

if u_exact is not None:
    plt.loglog(h_list, err_list)
    plt.xlabel('h')
    plt.ylabel('l2 norm', rotation=0)
    plt.title("the error as a function of mesh-size")
    plt.show()
    q = [math.log10(err_list[i]/err_list[i+1])/math.log10((N_list[i+1]+1) /
                                                          (N_list[i]+1)) for i in range(len(N_list)-1)]
    print("the convergence rate q is " + str(q))
