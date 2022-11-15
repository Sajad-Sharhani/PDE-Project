import math
import numpy as np
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt

# Finite Element Method Solver

# mass matrix assembler
# The input is a vector x of node coordinates
# The output is the mass matrix M


def mass(N, h):
    M = np.zeros((N+1, N+1))
    M[0][:2] = np.array([1/3, 1/6])
    for i in range(1, N):
        M[i][i-1:i+2] = np.array([1/6, 2/3, 1/6])
    M[N][-2:] = np.array([1/6, 1/3])
    return h*M


# advection matrix assembler
# The input is a vector x of node coordinates
# The output is the advection matrix A

def advection(N):
    A = np.zeros((N+1, N+1))
    A[0][:2] = np.array([-1/2, 1/2])
    for i in range(1, N):
        A[i][i-1:i+2] = np.array([-1/2, 0, 1/2])
    A[N][-2:] = np.array([-1/2, 1/2])
    return A


# diffusion matrix assembler
# The input is a vector x of node coordinates
# The output is the diffusion matrix S

def diffusion(N, h):
    S = np.zeros((N+1, N+1))
    S[0][:2] = np.array([1, -1])
    for i in range(1, N):
        S[i][i-1:i+2] = np.array([-1, 2, -1])
    S[N][-2:] = np.array([-1, 1])
    return 1 / h * S


# Set parameters
a = -1
b = 1
c = 2
T = 0.4
N_list = [41, 81, 161, 321, 641]
eps_list = [0.1]


hlist = [(b-a)/(N-1) for N in N_list]
errors = []


def uexact(x, t, c, eps):
    return c-np.tanh((x+1/2-c*t)/(2*eps))


def f(U):
    return inv_M @ (-eps*S @ U - 1/2 * A @ U**2)


# Initialize plot
plt.grid()
plt.xlabel("x")
plt.ylabel("u")
plt.title("Solutions at t="+str(T))


for N in N_list:
    h = (b-a)/(N-1)
    ht = 1.e-5
    x = np.linspace(a, b, N)
    t = 0
    M = mass(N-1, h)
    M = csc_matrix(M)
    inv_M = inv(M)
    S = diffusion(N-1, h)
    A = advection(N-1)

    for eps in eps_list:
        U = uexact(x, 0, c, eps)
        t = 0
        for i in range(np.int_(T/ht)):
            k1 = f(U)
            k2 = f(U+ht*k1/2)
            k3 = f(U+ht*k2/2)
            k4 = f(U+ht*k3)
            U += ht/6*(k1+2*k2+2*k3+k4)
            U[0] = uexact(a, t, c, eps)
            U[-1] = uexact(b, t, c, eps)
            t += ht
        errors.append(norm(U-uexact(x, T, c, eps)))
        plt.plot(x, U, label='N=' + str(N))

plt.plot(x, uexact(x, T, c, eps), label='Exact solution')
plt.legend()
plt.show()
plt.clf()


plt.xlabel("h")
plt.ylabel("Error and h**2")
plt.title("Error h**2 vs h at T = 0.4")
plt.loglog(hlist, errors, label='Error')
plt.loglog(hlist, [h**2 for h in hlist], label='h**2')
plt.show()
for i in N_list:

    q = [math.log10(errors[i]/errors[i+1])/math.log10((N_list[i+1]+1) /
                                                      (N_list[i]+1)) for i in range(len(N_list)-1)]
print("the convergence rate q is " + str(q))

fit = np.polyfit(np.log10(hlist), np.log10(errors), 1)
order = fit
print("Order of convergence="+str(order))
# Order of convergence=[ 1.18297351 -0.6612053 ]
# the convergence rate q is [1.5423324240811547, 1.313277776058959, 0.9549964094969702, 1.0762413158817745]
