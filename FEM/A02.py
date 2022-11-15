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
a = 0
b = 2*np.pi
T = 2
N_list = [201]
eps_list = [1, 0.1, 0.001]


hlist = [(b-a)/(N-1) for N in N_list]


def uexact(x):
    return np.sin(x)


def f(U):
    return inv_M @ (-eps*S @ U - 1/2 * A @ U**2)


# Initialize plot
plt.grid()
plt.xlabel("x")
plt.ylabel("u")
plt.title("Solutions at T="+str(T))


for N in N_list:
    h = (b-a)/(N-1)
    ht = 0.0002
    x = np.linspace(a, b, N)
    t = 0

    M = mass(N-1, h)
    M = csc_matrix(M)
    inv_M = inv(M)
    S = diffusion(N-1, h)
    A = advection(N-1)

    for eps in eps_list:
        U = uexact(x)

        t = 0
        for i in range(np.int_(T/ht)):
            k1 = f(U)
            k2 = f(U+ht*k1/2)
            k3 = f(U+ht*k2/2)
            k4 = f(U+ht*k3)
            U += ht/6*(k1+2*k2+2*k3+k4)
            U[0] = 0
            U[-1] = 0
            t += ht
        plt.plot(x, U, label='epsilon=' + str(eps))

plt.legend()
plt.show()
