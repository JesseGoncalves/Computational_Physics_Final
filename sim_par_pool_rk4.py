import numpy as np
import multiprocessing as mp
import time as tm
import csv

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

with open("par_pool_rk4_sim_data.csv","a") as csvfile:
	for proc in np.arange(10,11,1):
		for i in np.arange(0,10,1):
			start = tm.time()
			sol = rk4_solver(f,x,y0,h,proc)
			end = tm.time()
			ex_time = end - start
			data = [proc, ex_time]
                	writer = csv.writer(csvfile)
                	writer.writerow(data)

csvfile.close()
