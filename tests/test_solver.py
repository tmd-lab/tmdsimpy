"""
Script for verifying the accuracy of the solver function / used for 
development of interface

Outline:
    1. Try to solve a simple nonlinear problem
    
""" 



import sys
import numpy as np
import unittest

import verification_utils as vutils

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

        solver = NonlinearSolver()

        #x = solver.nsolve(fun, x0)

        x, R, dRdX, sol = solver.nsolve(fun, x0, verbose=False)

        self.assertLess(np.abs(R), 1e-12, 'Residual error is too high.')
        self.assertLess(np.abs(x - 3.0), 1e-12, 'Solution error is too high.')

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
        
        solver = NonlinearSolver()
        
        x_lin = solver.lin_solve(A, b)
        
        self.assertLess(np.linalg.norm(A @ x_lin - b), 1e-12, 
                        'Linear solve function is wrong')
        
        # Factor and solve
        factor_res = solver.lin_factor(A)
        
        x_factor_res = solver.lin_factored_solve(factor_res, b)
        
        self.assertLess(np.linalg.norm(A @ x_factor_res - b), 1e-12, 
                        'Factor and factored solve function is wrong')
        
    def test_eigs(self):
        """
        Test eigen-analysis interface of solver

        Returns
        -------
        None.

        """
        
        solver = NonlinearSolver()
        
        # Generate a K,M eigenvalue problem
        np.random.seed(1023)
        Ndof = 20
        Ncalc = 10
        w = np.sort(np.abs(np.random.rand(Ndof)))
        Phi = np.random.rand(Ndof, Ndof)
        Phi_inv = np.linalg.inv(Phi)
        
        K = Phi_inv.T @ np.diag(w**2) @ Phi_inv
        M = Phi_inv.T @ Phi_inv
        
        
        eigvals, eigvecs = solver.eigs(K, M=M, subset_by_index=[0, Ncalc-1])
        
        self.assertEqual(eigvals.shape[0], Ncalc, 'Incorrect number of eigenvalues.')
        self.assertEqual(eigvecs.shape[0], Ndof, 'Incorrect length of eigenvectors.')
        self.assertEqual(eigvecs.shape[1], Ncalc, 'Incorrect number of eigenvalues.')
        
        self.assertLess(np.linalg.norm(eigvals - w[:Ncalc]**2), 1e-12,
                        'Incorrect eigenvalues.')
        
        self.assertLess(np.linalg.norm(eigvecs - Phi[:, :Ncalc]), 1e-9,
                        'Incorrect eigenvectors.')
        
    def test_condition_fun(self):
        """
        Test that the conditioning function with the solver returns an 
        appropriate function.

        Returns
        -------
        None.

        """
        
        ###########
        # Function and Input to test
        
        fun = lambda X, calc_grad=True : \
                        (np.array([(X[0]-1)**3, 1e6*(X[1] - 0.01)**3]),
                          np.diag([3*(X[0]-1)**2, 3e6*(X[1] - 0.01)**2]))
                          
        X = np.array([1.1, 0.02])
        
        CtoP = np.array([1.0, 0.01])
        RPtoC = 3.1415
        
        ###########
        # Create the conditioned function and check the solution
        
        solver = NonlinearSolver()
        
        fun_cond = solver.conditioning_wrapper(fun, CtoP, RPtoC=RPtoC)
        
        Rc_fun, dRcdXc_fun = fun_cond(X/CtoP)
        
        Rc_ref = RPtoC*fun(X)[0]
        
        dRcdXc_ref = RPtoC*fun(X)[1]*np.diag(CtoP)
        
        self.assertLess(np.linalg.norm(Rc_fun - Rc_ref), 1e-10, 
                        'Conditioned residual is wrong.')
        
        self.assertLess(np.linalg.norm(dRcdXc_fun - dRcdXc_ref), 1e-10,
                        'Conditioned derivative is not expected value')
        
        # Check the gradient numerically of the new function
        grad_failed = vutils.check_grad(fun_cond, X, rtol=1e-10, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 
                     'Numerical dRdX of conditioned function does not match')
        
        # Check calc_grad options
        Rc_grad_False = fun_cond(X/CtoP, calc_grad=False)[0]
        
        self.assertLess(np.linalg.norm(Rc_grad_False - Rc_ref), 1e-10, 
                        'Conditioned residual, calc_grad=False, is wrong.')

        
if __name__ == '__main__':
    unittest.main()