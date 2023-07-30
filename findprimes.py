import asyncio
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

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
    f = np.setdiff1d(r, e)
    return f

async def main():
    x = int(input("Find prime numbers from 6n-1, n= "))
    y = int(input("Find prime numbers to 6n+1, n= "))
    start = time.time()
    
    loop = asyncio.get_running_loop()
    
    with ProcessPoolExecutor() as executor:
        f = await loop.run_in_executor(executor, find_primes, x, y)

    t = time.time() - start
    print(f)
    print(len(f))
    print(t)

if __name__ == '__main__':
    asyncio.run(main())
