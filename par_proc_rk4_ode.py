import numpy as np
import multiprocessing as mp
import time as tm
import matplotlib.pyplot as plt

start = tm.time()

# Define an output queue
output = mp.Queue()

def k_func(func,domain,h):
        """
        : param func: dy/dx = func(x)
        : param domain: Evaluation points [xn, xn+h] of func
        : param h: x step size
	"""
        k = h * (func(domain[0]) + 4 * func(domain[1]) + func(domain[2])) / 6
        output.put(k)
                 
def rk4_solver(func,x,y0,h):
        """      
        : param func: dy/dx = func(x)
        : param x: Domain points of func    
        : param y0: y(x0) = y0
        : param h: x step size
        """      
        vals = [y0]
	processes = [mp.Process(target=k_func, args=(func,[lo,mid,hi],h)) for lo, mid, hi in zip(x[:-1], x[1:], x[2:])]
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

sol = rk4_solver(f,x,y0,h)

end = tm.time()

exact = x[:-1]**4 - x[:-1]**3 + x[:-1]**2 - x[:-1]

#plt.plot(x[:-1],sol,label='approx')
#plt.plot(x[:-1],exact,label='exact')
#plt.legend()
#plt.show()
print(end - start)
print(np.mean((exact - sol)**2))
