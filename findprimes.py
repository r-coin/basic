from numba import njit
import numpy as np
import time

@njit
def setdiff1d_numba(ar1, ar2):
    return np.array([x for x in ar1 if x not in ar2])

@njit(parallel=True)
def find_primes(x, y):
    n = np.arange(1, y + 1)
    a = 6 * n - 1
    b = 6 * n + 1
    c = np.append(a, b)
    d = np.sort(c)
    m = np.arange(x, y + 1)
    o = 6 * m - 1
    p = 6 * m + 1
    q = np.append(o, p)
    r = np.sort(q)
    e = []
    for i in range(0, y):
        if int(d[i]) ** 2 <= 6 * y + 1:
            for j in range(int((6 * x - 2) / (int(d[i]) * 3)) - 1, int((6 * y + 2) / (int(d[i]) * 3))):
                e.append(int(d[i]) * int(d[j]))
    f = setdiff1d_numba(r, np.array(e))
    return f

x = int(input("Find primes from 6n-1, n= "))
y = int(input("Find primes to 6n+1, n= "))
start = time.time()
f = find_primes(x, y)
t = time.time() - start
print(f)
print(len(f))
print(t)
