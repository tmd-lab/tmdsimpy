"""
Verification of the AFT implementation with the Duffing (Cubic stiffness)
nonlinearity.
"""

import sys
import numpy as np
import unittest

# Python Utilities
sys.path.append('..')
import verification_utils as vutils

sys.path.append('../..')
from tmdsimpy.nlforces.cubic_stiffness import CubicForce


class TestDuffingAFT(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestDuffingAFT, self).__init__(*args, **kwargs)      
        
        
        self.analytical_sol_tol = 1e-13 # Tolerance comparing to analytical solution
        
        self.rtol_grad = 1e-11 # Relative gradient tolerance

    def test_simple_analytical(self):
        """
        First set of test cases. Move the first harmonic and check against 
        analytical expansions for AFT.
        
        Also Verify Gradients for this system

        Returns
        -------
        None.

        """        
        
        ####################
        # Simple Mapping to spring displacements
        Q = np.array([[1.0, 0], \
                      [-1.0, 1.0], \
                      [0, 1.0]])
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[1.0, 0.25, 0.0], \
                      [0.0, 0.25, 1.0]])
        
        kalpha = np.array([3, 0, 7])
        
        duff_force = CubicForce(Q, T, kalpha)
        
        """
        Case 1: 
            -Fix Center DOF
            -Only move the first Harmonic
            -Compare to analytical expansion of cos^3/sin^3
        """
        # h = np.array([0, 1, 2, 3]) # Manual Checking expansion / debugging
        h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) # Automate Checking with this
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        
        Nd = Q.shape[1]
        
        U = np.zeros((Nd*Nhc, 1))
        
        # First DOF, Cosine Term, Fundamental
        U[Nd+0, 0] = 4
        
        # Second DOF, Sine Term, Fundamental
        U[2*Nd+1, 0] = 3
        
        w = 1.5 # Test for various w
        
        # Calculation with simple first harmonic motion.        
        Fnl, dFnldU, dFnldw = duff_force.aft(U, w, h)
        
        # # Analytically Verify Force expansion:
        # # X^3*cos^3(x) = X^3*( 3/4*cos(x) + 1/4*cos(3x) )
        # # X^3*sin^3(x) = X^3*(3/4*sin(x) - 1/4*sin(3x)
        Fnl_analytical = np.zeros_like(Fnl) 
        Fnl_analytical[Nd+0] = T[0,0]*( 0.75*kalpha[0]*(Q[0,0]*U[Nd+0])**3 )
        Fnl_analytical[5*Nd+0] = T[0,0]*( 0.25*kalpha[0]*(Q[0,0]*U[Nd+0])**3 )
        
        Fnl_analytical[2*Nd+1] = T[1,2]*( 0.75*kalpha[2]*(Q[2,1]*U[2*Nd+1])**3 )
        Fnl_analytical[6*Nd+1] = T[1,2]*( -0.25*kalpha[2]*(Q[2,1]*U[2*Nd+1])**3 )
        
        analytical_sol_error = np.linalg.norm(Fnl - Fnl_analytical)
        
        self.assertLess(analytical_sol_error, self.analytical_sol_tol,
                        'Numerical AFT does not match analytical Duffing AFT solution.')
        
        
        # Numerically Verify Gradient
        fun = lambda U: duff_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect AFT gradient w.r.t. displacements')
                
        # Numerically Verify Frequency Gradient
        fun = lambda w: duff_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=self.rtol_grad)

        self.assertFalse(grad_failed, 'Incorrect AFT gradient w.r.t. frequency')

    def test_mixed_harmonics(self):
        """
        Test Gradients for cases where only some harmonics are included instead 
        of all of the harmonics from 0 to hmax

        Returns
        -------
        None.

        """     
        
        np.random.seed(1023)
        
        w = 1.5
        
        Q = np.array([[1.0, 0], \
                      [-1.0, 1.0], \
                      [0, 1.0]])
        
        # Skip the 4th harmonic
        h = np.array([0, 1, 2, 3, 5, 6, 7]) 
        kalpha = np.array([3, 5, 7])
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[1.0, 0.25, 0.0], \
                      [0.0, 0.25, 0.5]])
        
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        Nd = Q.shape[1]
        
        U = np.random.rand(Nd*Nhc, 1)
        
        
        duff_force = CubicForce(Q, T, kalpha)
        
        fun = lambda U: duff_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect AFT gradient w.r.t. displacements')
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: duff_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect AFT gradient w.r.t. frequency')
        
        ######################
        # Test without zeroth harmonic + skipping the 4th harmonic
        
        h = np.array([1, 2, 3, 5, 6, 7]) 
        
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        Nd = Q.shape[1]
        
        U = np.random.rand(Nd*Nhc, 1)
        
        fun = lambda U: duff_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect AFT gradient w.r.t. displacements')
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: duff_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect AFT gradient w.r.t. frequency')








if __name__ == '__main__':
    unittest.main()