import numpy as np
import time
from numba import cuda
import cupy as cp

def get_optimal_batch_size(free_memory, dtype):
    # Calculate the size of a single element in bytes
    element_size = cp.dtype(dtype).itemsize
    # Calculate the number of elements that can fit in the available memory
    num_elements = free_memory // element_size
    # Set the batch size to a fraction of the available memory: // 2
    batch_size = num_elements // 2
    return max(1, batch_size)

x = int(input("Find prime numbers from 6n-1, n= "))
y = int(input("Find prime numbers to 6n+1, n= "))

start = time.time()

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

e = np.zeros(6 * y + 2, dtype=np.int32)

@cuda.jit
def calculate_e(d, e, x, y):
    pos = cuda.grid(1)
    if pos < y:
        if int(d[pos]) ** 2 <= 6 * y + 1:
            for j in range(pos, y):
                if d[pos] * d[j] <= 6 * y + 1:
                    e[d[pos] * d[j]] = 1

d_device = cuda.to_device(d)
e_device = cuda.to_device(e)

blockdim = 256 # Set the block size to 256 threads per block
griddim = (y + blockdim - 1) // blockdim

# Calculate the optimal batch size based on the available memory and data type
free_memory = 8 * 1024 * 1024 * 1024 # 8GB in bytes, The amount of free memory available in bytes
batch_size = get_optimal_batch_size(free_memory, d.dtype)

# Split the data into batches and process each batch separately
for i in range(0, len(d), batch_size):
    d_batch = d[i:i+batch_size]
    d_batch_device = cuda.to_device(d_batch)
    calculate_e[griddim, blockdim](d_batch_device, e_device, x, y)

e_host = e_device.copy_to_host()

f = np.setdiff1d(r, np.where(e_host == 1)[0])
t = time.time() - start

print(f)
print(len(f))
print(t)