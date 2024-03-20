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
        
    def test_line_search(self):
        """
        Test that the line search function shows the appropriate behavior for
        a simple case for the different flags when called independently
        
        Only considers a scalar equation so it is easy to analytically check.
        
        This should test all combinations of input arguments and logical flow
        through the function.
        """
        
        ###############
        # Baseline problem being solved
        
        fun = lambda x, calc_grad=True : (np.arctan(x), np.diag(1 / (x**2 + 1)))
        
        X = np.array([11.1])
        
        Rx, dRdX = fun(X)
        
        # Utilize an easy deltaX for analytical solution checking
        # deltaX = -np.linalg.solve(dRdX, Rx)
        deltaX = np.array([-16])
        
        ###############
        # Case 1 (max iterations, same sign)
        
        config = {'line_search_iters' : 3,
                  'line_search_tol' : 0.25,
                  'line_search_same_sign' : True,
                  'verbose' : False}
        
        solver = NonlinearSolverOMP(config=config)
        
        alpha, sol = solver.line_search(fun, X, Rx, deltaX)
        
        self.assertEqual(alpha, 0.625, 
                         'alpha does not give expected analytical solution.')
        
        self.assertGreater(np.abs(sol['G(alpha)_bracket'][0]), 
                           np.abs(sol['G(alpha)_bracket'][1]),
                           'Test does not check the same sign argument fully.')
        
        ###############
        # Case 2 (max iterations, no need for same sign)
        
        config = {'line_search_iters' : 3,
                  'line_search_tol' : 0.25,
                  'line_search_same_sign' : False,
                  'verbose' : False}
        
        solver = NonlinearSolverOMP(config=config)
        
        alpha, sol = solver.line_search(fun, X, Rx, deltaX)
        
        self.assertEqual(alpha, 0.75, 
                         'alpha does not give expected analytical solution.')
        
        self.assertGreater(np.abs(sol['G(alpha)_bracket'][0]), 
                           np.abs(sol['G(alpha)_bracket'][1]),
                           'Test does not verify that same sign is violated.')

        ###############
        # Case 3 (converged, same sign)
        
        config = {'line_search_iters' : 5,
                  'line_search_tol' : 0.25,
                  'line_search_same_sign' : True,
                  'verbose' : False}
        
        solver = NonlinearSolverOMP(config=config)
        
        X = np.array([11.9])
        Rx, dRdX = fun(X)
        
        alpha, sol = solver.line_search(fun, X, Rx, deltaX)
        
        # alpha * 2**5 should be an integer.
        self.assertEqual(alpha, 0.71875, 
                          'alpha does not give expected analytical solution.')
        
        self.assertGreater(np.abs(sol['G(alpha)_bracket'][0]), 
                           np.abs(sol['G(alpha)_bracket'][1]),
                           'Test does not verify that same sign is violated.')

        ###############
        # Case 4 (converged, different sign)
        
        config = {'line_search_iters' : 5,
                  'line_search_tol' : 0.25,
                  'line_search_same_sign' : False,
                  'verbose' : False}
        
        solver = NonlinearSolverOMP(config=config)
        
        X = np.array([11.9])
        Rx, dRdX = fun(X)
        
        alpha, sol = solver.line_search(fun, X, Rx, deltaX)
        
        # alpha * 2**5 should be an integer.
        self.assertEqual(alpha, 0.75, 
                          'alpha does not give expected analytical solution.')
        
        self.assertGreater(np.abs(sol['G(alpha)_bracket'][0]), 
                           np.abs(sol['G(alpha)_bracket'][1]),
                           'Test does not verify that same sign is violated.')
        
        ###############
        # Case 5 (no zero steps, same sign)
        deltaX = np.array([32])
        X = np.array([-2.5])
        Rx, dRdX = fun(X)
        
        config = {'line_search_iters' : 3,
                  'line_search_tol' : 0.25,
                  'line_search_same_sign' : True,
                  'verbose' : False}
        
        solver = NonlinearSolverOMP(config=config)
        
        alpha, sol = solver.line_search(fun, X, Rx, deltaX)

        self.assertEqual(alpha, 0.125, 
                          'alpha does not give expected analytical solution.')
        
        self.assertGreater(np.abs(sol['G(alpha)_bracket'][0]), 
                           np.abs(sol['G(alpha)_bracket'][1]),
                           'Test does not verify that same sign is violated.')
        
        ###############
        # Case 6 (no zero steps, different sign)
        deltaX = np.array([100])
        X = np.array([-1])
        Rx, dRdX = fun(X)
        
        config = {'line_search_iters' : 5,
                  'line_search_tol' : 0.25,
                  'line_search_same_sign' : False,
                  'verbose' : False}
        
        solver = NonlinearSolverOMP(config=config)
        
        alpha, sol = solver.line_search(fun, X, Rx, deltaX)

        self.assertEqual(alpha, 0.03125, 
                          'alpha does not give expected analytical solution.')
        
        ###############
        # Case 7 (no need to call, safe, positive step)
        deltaX = np.array([5])
        X = np.array([-10])
        Rx, dRdX = fun(X)
        
        config = {'line_search_iters' : 5,
                  'line_search_tol' : 0.25,
                  'line_search_same_sign' : False,
                  'verbose' : False}
        
        solver = NonlinearSolverOMP(config=config)
        
        alpha, sol = solver.line_search(fun, X, Rx, deltaX)
        
        self.assertEqual(alpha, 1.0, 
                  'alpha does not give expected analytical solution.')

        ###############
        # Case 8 (no need to call, safe, negative step)
        deltaX = np.array([-10])
        X = np.array([20])
        Rx, dRdX = fun(X)
        
        config = {'line_search_iters' : 5,
                  'line_search_tol' : 0.25,
                  'line_search_same_sign' : False,
                  'verbose' : False}
        
        solver = NonlinearSolverOMP(config=config)
        
        alpha, sol = solver.line_search(fun, X, Rx, deltaX)
        
        self.assertEqual(alpha, 1.0, 
                  'alpha does not give expected analytical solution.')
        
    
    def test_line_search_nsolve(self):
        """
        Test line search integration into nsolve routine for a simple case
        Uses 3 DOF system to verify vectors are appropriately handled
        """
        
        ###############
        # Baseline problem being solved
        
        coef = np.array([-1.0, 5.0, 2.5])
        offset = np.array([0.5, 0.0, -5.0])
        
        fun = lambda x, calc_grad=True : (np.arctan(coef*x - offset), \
                                          np.diag(coef / ((coef*x - offset)**2 + 1)))
        
        X0 = np.array([10, -2.0, 20.0])
        X0 = 3.0/coef + offset
        
        ###############
        # Check that the test is giving appropriate inputs
        
        grad_failed = vutils.check_grad(fun, X0, verbose=False, rtol=1e-9)
        
        self.assertFalse(grad_failed, 
                         'Test gives wrong Jacobian to nonlinear solver')
        grad_failed = vutils.check_grad(fun, offset, verbose=False, rtol=1e-9)
        
        self.assertFalse(grad_failed, 
                         'Test gives wrong Jacobian to nonlinear solver')
        
        ###############
        # Setup Solvers
        config_base = {'verbose' : False}
        
        solver_baseline = NonlinearSolverOMP(config=config_base)
        
        
        config_ls = {'line_search_iters' : 5,
                  'line_search_tol' : 0.25,
                  'line_search_same_sign' : False,
                  'verbose' : False,
                  'xtol' : 1e-7}
        
        solver_linesearch = NonlinearSolverOMP(config=config_ls)
        
        
        config_ls_bfgs = {'reform_freq' : 2,
                          'line_search_iters' : 5,
                          'line_search_tol' : 0.25,
                          'line_search_same_sign' : False,
                          'verbose' : True,
                          'xtol' : 1e-7}
        
        solver_ls_bfgs = NonlinearSolverOMP(config=config_ls_bfgs)
        
        ###############
        # Try Solutions with both
        
        X_base,_,_,sol_base = solver_baseline.nsolve(fun, X0)
        
        X_ls,_,_,sol_ls = solver_linesearch.nsolve(fun, X0)
        
        X_ls_bfgs,_,_,sol_ls_bfgs = solver_ls_bfgs.nsolve(fun, X0)
        
        self.assertFalse(sol_base['success'], 
                         'Baseline solution converged, so test does not check'\
                         +' line search well.')
            
        self.assertTrue(sol_ls['success'], 
                    'Line search did not allow solver to find the solution.')
        
        self.assertLess(np.linalg.norm(X_ls - offset/coef), 1e-6,
                    'Line search solution does not meet expected tolerance.')

        self.assertTrue(sol_ls['success'], 
                    'Line search + BFGS did not find the solution.')
        
        self.assertLess(np.linalg.norm(X_ls - offset/coef), 1e-6,
                    'Line search + BFGS solution does not meet expected tolerance.')
        
if __name__ == '__main__':
    unittest.main()