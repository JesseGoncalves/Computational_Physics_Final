import multiprocessing as mp
import numpy as np
import time as tm

# Start timer
start = tm.time()


# Define an output queue
output = mp.Queue()

# Define monte carlo integration function
def mc_int(a,b,n):
	xr = np.random.uniform(low=a, high=b, size=n)
	yr = np.sinc(xr)**2
	I = 2 * 3.5 * np.mean(yr)
	output.put(I)
 
a = -3.5
b = 3.5
n = 100000
cores = 4

# Setup a list of processes that we want to run
processes = [mp.Process(target=mc_int, args=(a,b,n)) for x in range(cores)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = np.mean([output.get() for p in processes])

# End timer
end = tm.time()

print(results)
print(cores)
print(n*cores)
print(end - start)
