"""
Script for verifying the accuracy of the solver function / used for development of interface

Outline:
    1. Try to solve a simple nonlinear problem
    
failed_flag = False, changes to true if a test fails at any point 
""" 


import sys
import numpy as np

# Path to Solver Routine
sys.path.append('../ROUTINES/')

from solvers import NonlinearSolver

failed_flag = False

###############################################################################
##### Test a Simple Case                                                  #####
###############################################################################

# Solve for root of a function
# x^2 - 9
fun = lambda x : (x**2-9, 2*x)
# fun = lambda x : (np.cos(x), -np.sin(x)) # - poor test since can jump to multiple roots

x0 = np.array([3.5])

solver = NonlinearSolver

#x = solver.nsolve(fun, x0)

x, R, dRdX, sol = solver.nsolve(fun, x0, verbose=True)


failed_flag = failed_flag or np.abs(R) > 1e-12
failed_flag = failed_flag or np.abs(x - 3.0) > 1e-12

###############################################################################
##### Test Results                                                        #####
###############################################################################

if failed_flag:
    print('\n\nTest FAILED, investigate results further!\n')
else:
    print('\n\nTest passed.\n')
    



