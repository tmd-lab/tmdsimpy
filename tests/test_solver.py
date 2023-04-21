"""
Script for verifying the accuracy of the solver function / used for 
development of interface

Outline:
    1. Try to solve a simple nonlinear problem
    
""" 



import sys
import numpy as np
import unittest

sys.path.append('..')
from tmdsimpy.solvers import NonlinearSolver


class TestSolver(unittest.TestCase):
    
    def test_solve(self):
        """
        Test a simple case with the solver routine.

        Returns
        -------
        None.

        """

        # Solve for root of a function
        # x^2 - 9
        fun = lambda x : (x**2-9, 2*x)
        # fun = lambda x : (np.cos(x), -np.sin(x)) # - poor test since can jump to multiple roots

        x0 = np.array([3.5])

        solver = NonlinearSolver

        #x = solver.nsolve(fun, x0)

        x, R, dRdX, sol = solver.nsolve(fun, x0, verbose=False)

        self.assertLess(np.abs(R), 1e-12, 'Residual error is too high.')
        self.assertLess(np.abs(x - 3.0), 1e-12, 'Solution error is too high.')

        
    
        
if __name__ == '__main__':
    unittest.main()