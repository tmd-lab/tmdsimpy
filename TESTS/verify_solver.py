# Verification of Solver and Iteration Callback Function for Convergence

import sys
import numpy as np

# Path to Solver Routine
sys.path.append('../')

from solvers import NonlinearSolver




# Solve for root of x^2 - 9
fun = lambda x : (np.cos(x), -np.sin(x))

x0 = 3

solver = NonlinearSolver

#x = solver.nsolve(fun, x0)

x, R, dRdX, sol = solver.nsolve(fun, x0, verbose=True)




