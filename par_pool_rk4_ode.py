import numpy as np
import multiprocessing as mp
import time as tm
import matplotlib.pyplot as plt

start = tm.time()

def k_func(func,domain,h):
        """
        : param func: dy/dx = func(x)
        : param domain: Evaluation points [xn, xn+h] of func
        : param h: x step size
        """
        k = h * (func(domain[0]) + 4 * func(domain[1]) + func(domain[2])) / 6
        return k

def rk4_solver(func,x,y0,h,proc):

        # func: dy/dx = func(x)
        # x: Domain points of func
        # y0: y(x0) = y0
        # h: x step size
	# proc: Number of processes to pool

        vals = [y0]
	pool = mp.Pool(processes=proc)
	results = [pool.apply_async(k_func, args=(func,[lo,mid,hi],h)) for lo, mid, hi in zip(x[:-1], x[1:], x[2:])]
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
proc = mp.cpu_count()
sol = rk4_solver(f,x,y0,h,proc)

end = tm.time()

exact = x[:-1]**4 - x[:-1]**3 + x[:-1]**2 - x[:-1]

#plt.plot(x[:-1],sol,label='approx')
#plt.plot(x[:-1],exact,label='exact')
#plt.legend()
#plt.show()
print("Number of processes: ", proc)
print("time: ", end - start)
print("error: ", np.mean((exact - sol)**2))
