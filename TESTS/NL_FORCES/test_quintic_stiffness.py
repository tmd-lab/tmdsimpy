"""
Test AFT Implementation of stiffness of the form X^5
"""


import sys
import numpy as np
import unittest

# Python Utilities
sys.path.append('../')
sys.path.append('../../ROUTINES/')
sys.path.append('../../ROUTINES/NL_FORCES')

import verification_utils as vutils
from quintic_stiffness import QuinticForce


class TestQuinticAFT(unittest.TestCase):

    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestQuinticAFT, self).__init__(*args, **kwargs)      
        
        self.analytical_tol = 1e-12 # Comparison to analytical solution tolerance
        
        self.rtol_grad = 1e-11 # Relative gradient tolerance


    def test_analytical_aft(self):
        """
        Test analytical expansion of nonlinear force for first harmonic motion
        against the AFT values.
        
        Also check gradients for the simple case

        Returns
        -------
        None.

        """
                
        #######################
        # Test System
        
        # Simple Mapping to spring displacements
        Q = np.array([[1.0, 0], \
                      [-1.0, 1.0], \
                      [0, 1.0]])
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[1.0, 0.25, 0.0], \
                      [0.0, 0.25, 1.0]])
        
        kalpha = np.array([3, 0, 7])
        
        nl_force = QuinticForce(Q, T, kalpha)
        
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
        
        w = 1 # Test for various w
        
        # Testing Simple First Harmonic Motion        
        Fnl, dFnldU = nl_force.aft(U, w, h)[0:2]
        
        # # Analytically Verify Force expansion:
        # # X^5*cos^5(x) = X^5*( 10/16*cos(x) + 5/16*cos(3x) + 1/16*cos(5x) )
        # # X^5*sin^5(x) = X^5*( 10/16*sin(x) - 5/16*sin(3x) + 1/16*sin(5x) )
        Fnl_analytical = np.zeros_like(Fnl) 
        Fnl_analytical[Nd+0] = T[0,0]*( 10/16*kalpha[0]*(Q[0,0]*U[Nd+0])**5 )
        Fnl_analytical[5*Nd+0] = T[0,0]*( 5/16*kalpha[0]*(Q[0,0]*U[Nd+0])**5 )
        Fnl_analytical[9*Nd+0] = T[0,0]*( 1/16*kalpha[0]*(Q[0,0]*U[Nd+0])**5 )
        
        Fnl_analytical[2*Nd+1] = T[1,2]*( 10/16*kalpha[2]*(Q[2,1]*U[2*Nd+1])**5 )
        Fnl_analytical[6*Nd+1] = T[1,2]*( -5/16*kalpha[2]*(Q[2,1]*U[2*Nd+1])**5 )
        Fnl_analytical[10*Nd+1] = T[1,2]*( 1/16*kalpha[2]*(Q[2,1]*U[2*Nd+1])**5 )
        
        
        error = np.linalg.norm(Fnl - Fnl_analytical)
        
        self.assertLess(error, self.analytical_tol, 
                        'Quintic stiffness does not match analytical AFT.')
        
        
        # Numerically Verify Gradient
        fun = lambda U: nl_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                        rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 
                         'Incorrect displacement w.r.t. displacement.')
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: nl_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, 
                                        rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 
                         'Incorrect displacement w.r.t. frequency.')


    def test_multiharmonic_grads(self):
        """
        Test derivatives for multiharmonic motion and for mixed sets / skipping
        some harmonic components

        Returns
        -------
        None.

        """
                
        # np.random.seed(42)
        np.random.seed(1023)
        # np.random.seed(0)
        
        # Simple Mapping to spring displacements
        Q = np.array([[1.0, 0], \
                      [-1.0, 1.0], \
                      [0, 1.0]])
        
        h = np.array([0, 1, 2, 3, 5, 6, 7]) # Automate Checking with this
        kalpha = np.array([3, 5, 7])
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[1.0, 0.25, 0.0], \
                      [0.0, 0.25, 0.5]])
        
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        Nd = Q.shape[1]
        
        U = np.random.rand(Nd*Nhc, 1)
        
        w = 1.375 # Test for various w
        
        nl_force = QuinticForce(Q, T, kalpha)
        
        
        ######################
        # Test without 4th harmonic
        
        fun = lambda U: nl_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                        rtol=self.rtol_grad)

        self.assertFalse(grad_failed, 'Incorrect displacement gradient.')        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: nl_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False,
                                        rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect displacement frequency.')   
        
        
        ######################
        # Test without zeroth harmonic + skipping 4th harmonic
        h = np.array([1, 2, 3, 5, 6, 7]) # Automate Checking with this
        
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        Nd = Q.shape[1]
        
        U = np.random.rand(Nd*Nhc, 1)
        
        fun = lambda U: nl_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                        rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect displacement gradient.')   
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: nl_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, 
                                        rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect frequency gradient.')   
        
        
        

if __name__ == '__main__':
    unittest.main()