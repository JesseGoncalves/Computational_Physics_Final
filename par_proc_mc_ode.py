import multiprocessing as mp
import numpy as np
import time as tm
import matplotlib.pyplot as plt

start = tm.time()

# Define an output queue
output = mp.Queue()

def mc_int(func, domain, n):
        """
        :param func: dy/dx = func(x)
        :param domain: Endpoints of integration domain
        :param n: Number of random samples for Monte Carlo estimate
        """
        samples = np.random.uniform(domain[0],domain[1],n)
        volume = abs(domain[1] - domain[0])
        output.put(np.mean(func(samples)) * volume)

def mc_solver(func, y0, x, n):
        """
        :param func: dy/dx = func(x)
        :param y0: Initial value func(x0) = y0
        :param x: Domain points over which func should be integrated
        :param n: Number of random samples for Monte Carlo estimate
        """
        vals = [y0]
	processes = [mp.Process(target=mc_int, args=(func,[lo,hi],n)) for lo, hi in zip(x[:-1], x[1:])]
		
	# Run processes
	for p in processes:
    		p.start()

	# Exit the completed processes
	for p in processes:
     		p.join()

	# Get process results from the output queue
	results = [output.get() for p in processes]

	for i in results:
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

sol = mc_solver(f,y0,x,n)

end = tm.time()

exact = x**4 - x**3 + x**2 - x

#plt.plot(x,sol,label='approx')
#plt.plot(x,exact,label='exact')
#plt.legend()
#plt.show()
print(end - start)
print(np.mean((exact - sol)**2))
