import time
import numpy as np
import scipy.linalg as la


def tic():
    # Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        return time.time() - startTime_for_tictoc
    else:
        print("Toc: start time not set")


# Write your own LU factorization function here
def myownLU(A):
    """
    LU factorization of a square matrix A.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
    return L, U

# Forward substitution and back substitution functions


def forward_substitution(L, b):
    """
    Forward substitution for solving the linear system Lx = b.
    """
    x = np.zeros_like(b, dtype=np.double)
    for i in range(L.shape[0]):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x


def back_substitution(U, b):
    """
    Back substitution for solving the linear system Ux = b.
    """
    x = np.zeros_like(b, dtype=np.double)
    for i in range(U.shape[0] - 1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, (i + 1):], x[(i + 1):])) / U[i, i]
    return x


# LU solver function
def lu_solver(A, b):
    """
    LU solver for solving the linear system Ax = b.
    """
    L, U = myownLU(A)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x


A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
b = np.array([6, 25, -11, 15])

tic()
x = la.solve(A, b)
print(x)
print("Sovle", toc())

# scipy LU solver and iteration number
tic()
x1 = la.lu_solve(la.lu_factor(A), b)
print(x1)
print("Scipy LU", toc())

tic()
x = lu_solver(A, b)
print(x)
print("LU", toc())
