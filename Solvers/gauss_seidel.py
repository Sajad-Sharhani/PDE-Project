import time
import numpy as np


def tic():
    # Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        return time.time() - startTime_for_tictoc
    else:
        print("Toc: start time not set")


def gauss_seidel(A, b, tol=1.e-5, maxiter=10000):
    """
    Gauss-Seidel method for solving the linear system Ax = b.
    """
    x = np.zeros_like(b)

    for i in range(maxiter):

        x_old = x.copy()

        for j in range(A.shape[0]):
            x[j] = (b[j] - np.dot(A[j, :j], x[:j]) -
                    np.dot(A[j, (j + 1):], x_old[(j + 1):])) / A[j, j]

        if np.linalg.norm(x - x_old) / np.linalg.norm(x) < tol:
            break

    return i, x


A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
b = np.array([6, 25, -11, 15])


tic()
i, x = gauss_seidel(A, b)
print("Solution: ", x)
print("Iteration: ", i)
print("Time: ", toc())
