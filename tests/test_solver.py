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
        solver.lin_factor(A)
        
        x_factor_res = solver.lin_factored_solve(b)
        
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

        
if __name__ == '__main__':
    unittest.main()