import numpy as np
import matplotlib.pyplot as plt

def mc_int(func, domain, n_samples):
    """
    :param func: Lambda function of the gradient field. Signature should be `func(x)`
    :param domain: Endpoints of the integration domain
    :param n_samples: Number of random samples for the estimate
    """
    samples = np.random.uniform(low=domain[0], high=domain[1], size=n_samples)
    volume = abs(domain[1] - domain[0])
    return np.mean(func(samples)) * volume


def nonlinear_mc_solver(func, y0, t):
	"""
	:param func: Lambda function of the nonlinear gradient field. Signature should be `func(y0, x)`
	:param y0: Initial function value corresponding to the x0. The latter is given as the first element of x
	:param t: Domain points over which the field `func` should be integrated
	"""

	n_samples = 100
    	sols = [y0]
    	for lo, hi in zip(t[:-1], t[1:]) :
        	part_func = lambda v: func(x=v, y=sols[-1])
       	 	sols.append(sols[-1] + mc_int(part_func, (lo, hi), n_samples=n_samples))
    	return np.asarray(sols)

def f(y, x):
    	return  x * np.sqrt(np.abs(y)) + np.sin(x * np.pi/2)**3 - 5 * (x > 2)

x = np.arange(-4,4,0.01)
x0 = -4
y0 = 4

sol = nonlinear_mc_solver(f,y0,x)
plt.plot(x,sol)
plt.show()
