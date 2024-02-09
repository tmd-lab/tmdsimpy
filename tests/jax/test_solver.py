"""
Script for verifying the accuracy of the solver function / used for 
development of interface

Outline:
    1. Try to solve a simple nonlinear problem
    
""" 



import sys
import numpy as np
import unittest

sys.path.append('../..')
from tmdsimpy.jax.solvers import NonlinearSolverOMP

sys.path.append('..')
import verification_utils as vutils

class TestSolverOMP(unittest.TestCase):
    
    def test_solve(self):
        """
        Test a simple case with the solver routine.

        Returns
        -------
        None.

        """

        # Solve for root of a function
        # x^2 - 9
        fun = lambda x : (x**2-9, 2*np.atleast_2d(x))
        # fun = lambda x : (np.cos(x), -np.sin(x)) # - poor test since can jump to multiple roots

        x0 = np.array([3.5])

        config={'xtol': 1e-9}

        solver = NonlinearSolverOMP(config=config)

        x, R, dRdX, sol = solver.nsolve(fun, x0, verbose=False)

        self.assertLess(np.abs(R), 1e-12, 'Residual error is too high.')
        self.assertLess(np.abs(x - 3.0), 1e-12, 'Solution error is too high.')

    def test_bfgs_solve(self):
        """
        Test a simple case with the solver routine.

        Returns
        -------
        None.

        """

        # Solve for root of a function
        # x^2 - 9
        fun = lambda x, calc_grad : (x**2-9, 2*np.atleast_2d(x))
        # fun = lambda x : (np.cos(x), -np.sin(x)) # - poor test since can jump to multiple roots

        x0 = np.array([3.5])

        config={'xtol': 1e-9,
                'reform_freq' : 4}
        
        solver = NonlinearSolverOMP(config=config)

        x, R, dRdX, sol = solver.nsolve(fun, x0, verbose=False)

        self.assertLess(np.abs(R), 1e-12, 'Residual error is too high.')
        
        self.assertLess(np.abs(x - 3.0), 1e-12, 'Solution error is too high.')
        
        self.assertLess(sol['njev'], sol['nfev'], 
                        'Did not use fewer jacobian evaluations than function'\
                        +'evaluations, so not using BFGS')
        
    def test_bfgs_solve_mdof(self):
        """
        Test numerical solver for multidimensional problem

        Returns
        -------
        None.

        """

        # Solve for root of a complex polynomial function
        # [(x-3)*(x+y-z)
        #  (y+5)*(x^2*y^2*z^2 + x*y*z)
        #  (z-2.5)*(3*x^2 - 5*y^2 + 2*z^2)]
        #
        # A solution is obviously [3, -5, 2.5]
        #
        fun = lambda x, calc_grad : (\
                 np.array([(x[0]-3)*(x[0]+x[1]-x[2]),
                        (x[1]+5)*(x[0]**2*x[1]**2*x[2]**2 + x[0]*x[1]*x[2]),
                        (x[2]-2.5)*(3*x[0]**2 - 5*x[1]**2 + 2*x[2]**2)])\
                     , 
                 np.array([[-x[2]+x[1]+2*x[0]-3,
                             x[0]-3,
                             3-x[0]], \
                       [(x[1]+5)*(2*x[0]*x[1]**2*x[2]**2+x[1]*x[2]),
                            (x[1]+5)*(2*x[0]**2*x[1]*x[2]**2+x[0]*x[2])+x[0]**2*x[1]**2*x[2]**2+x[0]*x[1]*x[2],
                            (x[1]+5)*(2*x[0]**2*x[1]**2*x[2]+x[0]*x[1])], \
                       [6*x[0]*(x[2]-2.5), 
                            -10*x[1]*(x[2]-2.5), 
                            2*x[2]**2+4*(x[2]-2.5)*x[2]-5*x[1]**2+3*x[0]**2]]))


        x0 = np.array([3.25, -5.15, 2.5])

        config={'rtol': 1e-13,
                'stopping_tol' : ['rtol'],
                'reform_freq' : 4}
        
        # Check that the gradient is correctly implemented for the problem
        
        grad_failed = vutils.check_grad(lambda x : fun(x, True), 
                                        x0, verbose=False, 
                                        rtol=1e-10)
        
        self.assertFalse(grad_failed, 'Bad test, need to give correct gradient.')
        
        solver = NonlinearSolverOMP(config=config)

        x, R, dRdX, sol = solver.nsolve(fun, x0, verbose=False)

        self.assertLess(np.linalg.norm(R), 5e-12, 'Residual error is too high.')
        
        self.assertLess(np.linalg.norm(x - np.array([3.0, -5.0, 2.5])), 1e-12, 
                        'Solution error is too high.')
        
        self.assertLess(sol['njev'], sol['nfev'], 
                        'Did not use fewer jacobian evaluations than function'\
                        +'evaluations, so not using BFGS')
            
    def test_bfgs_solve_nan(self):
        """
        Test that BFGS/NR exit immediately upon calculating a nan valued
        residual or jacobian step.

        Returns
        -------
        None.

        """
        
        ################
        # BFGS Version
        
        fun = lambda x, calc_grad : (np.array([np.nan]), np.array([[np.nan]]))
            
        config={'xtol': 1e-11,
                'reform_freq' : 4,
                'max_steps' : 20}
        
        solver = NonlinearSolverOMP(config=config)
        
        x0 = np.array([0.0])
        
        x, R, dRdX, sol = solver.nsolve(fun, x0, verbose=False)
        
        self.assertEqual(sol['nfev'], 1, 
                         'Solver did not stop when it found nan values')

        ################
        # Newton Raphson Version
        
        fun = lambda x : (np.array([np.nan]), np.array([[np.nan]]))

        config={'xtol': 1e-11,
                'reform_freq' : 1,
                'max_steps' : 20}
        
        solver = NonlinearSolverOMP(config=config)
        
        x0 = np.array([0.0])
        
        x, R, dRdX, sol = solver.nsolve(fun, x0, verbose=False)
        
        self.assertEqual(sol['nfev'], 1, 
                         'Solver did not stop when it found nan values')
        
    def test_lin_solve(self):
        """
        Test linear solve interface on solver object

        Returns
        -------
        None.

        """
        # Seed for repeatability
        np.random.seed(1023)
        
        # System to solve
        A = np.random.rand(4,4)
        b = np.random.rand(4)
        
        solver = NonlinearSolverOMP()
        
        x_lin = solver.lin_solve(A, b)
        
        self.assertLess(np.linalg.norm(A @ x_lin - b), 1e-12, 
                        'Linear solve function is wrong')
        
        # Factor and solve
        factor_res = solver.lin_factor(A)
        
        x_factor_res = solver.lin_factored_solve(factor_res, b)
        
        self.assertLess(np.linalg.norm(A @ x_factor_res - b), 1e-12, 
                        'Factor and factored solve function is wrong')
    
        
if __name__ == '__main__':
    unittest.main()