import time
import cg
import jacobi
import gauss_seidel
import myownLU
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


# generate a random matrix of size 1000x1000
N = 1000

w = 100
F = w*np.eye(N)
print(F)

A = np.random.rand(N, N) + w * np.eye(N)

b = np.random.rand(N)


# solve the linear system
tic()
i, x = cg.cg(A, b, tol=1.e-5)


print("Iteration", i)
print("CG", toc())

tic()
i, x = gauss_seidel.gauss_seidel(A, b, tol=1.e-5)
# print(x)
print("Iteration", i)
print("Gauss Seidel", toc())

tic()
i, x = jacobi.jacobi(A, b, tol=1.e-5)
# print(x)
print("Iteration", i)
print("Jacobi", toc())


tic()
x = la.solve(A, b)
# print(x)
print("Solve", toc())

tic()
x = la.lu_solve(la.lu_factor(A), b)
# print(x)
print("Scipy LU", toc())

tic()
x = myownLU.lu_solver(A, b)
# print(x)
print("LU", toc())
