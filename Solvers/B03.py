import time
import numpy as np
import scipy.linalg as la
import jacobi
import gauss_seidel
import cg
import myownLU


def tic():
    # Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        return time.time() - startTime_for_tictoc
    else:
        print("Toc: start time not set")


def matrix(x, alpha):
    n = len(x) - 1
    S = np.zeros((n+1, n+1))
    for i in range(n):
        h = x[i+1] - x[i]
        S[i, i] = 2+alpha
        S[i+1, i+1] = 2+alpha
        S[i, i+1] = -1
        S[i+1, i] = -1
    return S


N = 10000
b = np.random.rand(N+1)
xl = 0
xr = 1
alpha = 0.00001
xvec = np.linspace(xl, xr, N+1)

A = matrix(xvec, alpha)


tic()
i, x = cg.cg(A, b, tol=1.e-5)
print("Iteration", i)
print("CG", toc())

tic()
i, x = gauss_seidel.gauss_seidel(A, b, tol=1.e-5)
print(x)
print("Iteration", i)
print("Gauss Seidel", toc())

tic()
i, x = jacobi.jacobi(A, b, tol=1.e-5)
print(x)
print("Iteration", i)
print("Jacobi", toc())


# Scipy Solve
tic()
x = la.solve(A, b)
print(x)
print("Solve", toc())

# scipy LU solver and iteration number
tic()
x1 = la.lu_solve(la.lu_factor(A), b)
print("Scipy LU", toc())

# LU decomposition
tic()
x = myownLU.lu_solver(A, b)
print("LU", toc())
