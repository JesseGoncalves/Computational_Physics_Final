import numpy as np
import scipy as sp
from scipy import special
import time as tm
import matplotlib.pyplot as plt

start = tm.time()

def km(func,domain,m):
	"""
	: param func: dy/dx = func(x)
	: param domain: Evaluation points [xn, xn+h] of func
	: param m: number corresponding to k
	"""
	
	if m == 1:
		k = func(domain[0])
	elif m == 4:
		k = func(domain[2])
	else:
		k = func(domain[1])

	return k

def rk4_solver(func,x,y0,h):
	"""
	: param func: dy/dx = func(x)
	: param x: Domain points of func
	: param y0: y(x0) = y0
	: param h: x step size
	"""
	
	vals = [y0]
	for lo, mid, hi in zip(x[:-1], x[1:], x[2:]):
		vals.append(vals[-1] + h * (km(func,[lo,mid,hi],1) + 2 * km(func,[lo,mid,hi],2) + 2 * km(func,[lo,mid,hi],3) + km(func,[lo,mid,hi],4)) / 6)
	
	return np.asarray(vals) 

def f(x):
	return x**2

a = 0
b = 1
h = 0.01
x = np.arange(a,b,h)
y0 = 0 		 

sol = rk4_solver(f,x,y0,h)

end = tm.time()

exact = x[:-1]**3 / 3

plt.plot(x[:-1],sol,label='approx')
plt.plot(x[:-1],exact,label='exact')
plt.legend()
plt.show()
print(end - start)
print(np.mean((exact - sol)**2))
