import multiprocessing as mp
import numpy as np
import time as tm
import matplotlib.pyplot as plt

start = tm.time()

def mc_int(func,domain,n):

        # func: dy/dx = func(x)
        # domain: Endpoints of integration domain
        # n: Number of random samples for Monte Carlo estimate
        
        samples = np.random.uniform(domain[0],domain[1],n)
        volume = abs(domain[1] - domain[0])
        return np.mean(func(samples)) * volume

def mc_solver(func,y0,x,n,proc):

        # func: dy/dx = func(x)
        # y0: Initial value func(x0) = y0
        # x: Domain points over which func should be integrated
        # n: Number of random samples for Monte Carlo estimate
        # proc: Number of processes to pool

	vals = [y0]
        pool = mp.Pool(processes=proc)
	results = [pool.apply_async(mc_int, args=(func,[lo,hi],n)) for lo, hi in zip(x[:-1], x[1:])]
	output = [p.get() for p in results]
	for i in output:
		vals.append(vals[-1] + i)
	
	return np.asarray(vals)

def f(x):
	return 4 * x**3 - 3 * x**2 + 2 * x - 1

a = 0
b = 1
h = 0.01
x = np.arange(a,b,h)
y0 = 0
n = 1000
proc = mp.cpu_count()
sol = mc_solver(f,y0,x,n,proc)

end = tm.time()

exact = x**4 - x**3 + x**2 - x

#plt.plot(x,sol,label='approx')
#plt.plot(x,exact,label='exact')
#plt.legend()
#plt.show()
print("Number of processes: ", proc)
print("time: ", end - start)
print("error: ", np.mean((exact - sol)**2))
