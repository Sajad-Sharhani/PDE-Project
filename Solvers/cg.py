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
# conjugate gradient method


def cg(A, b, tol=1.e-5, maxiter=100000):
    """
    Conjugate gradient method for solving the linear system Ax = b.
    """
    x = np.zeros_like(b, dtype=np.double)
    r = b - np.dot(A, x)
    p = r.copy()
    rsold = np.dot(r, r)

    for i in range(maxiter):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return i, x


A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
b = np.array([6, 25, -11, 15])


tic()
i, x = cg(A, b)
print("Solution: ", x)
print("Iteration: ", i)
print("Time: ", toc())
