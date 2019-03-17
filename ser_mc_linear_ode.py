import numpy as np
import time as tm
import matplotlib.pyplot as plt

start = tm.time()

def mc_int(func, domain, n):
    	"""
    	:param func: dy/dx = func(x)
    	:param domain: Endpoints of integration domain
    	:param n: Number of random samples for Monte Carlo estimate
    	"""
    	samples = np.random.uniform(domain[0],domain[1],n)
    	volume = abs(domain[1] - domain[0])
    	return np.mean(func(samples)) * volume

def mc_solver(func, y0, x, n):
    	"""
    	:param func: dy/dx = func(x)
    	:param y0: Initial value func(x0) = y0
    	:param x: Domain points over which func should be integrated
    	:param n: Number of random samples for Monte Carlo estimate
    	"""
    	vals = [y0]
    	for lo, hi in zip(x[:-1], x[1:]):
        	vals.append(vals[-1] + mc_int(func, [lo, hi], n))
    	
	return np.asarray(vals)

def f(x):
    	return x**3 + 2 * x**2 - 3**x

a = -2
b = 2
h = 0.01
x = np.arange(a,b,h)
y0 = 0
n = 100000

sol = mc_solver(f,y0,x,n)

end = tm.time()

plt.plot(x,sol)
plt.show()
print(end - start)
