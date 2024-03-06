"""
Test AFT Implementation of stiffness of the form X^5
"""


import sys
import numpy as np
import unittest

# Python Utilities
sys.path.append('..')
import verification_utils as vutils

sys.path.append('../..')
from tmdsimpy.nlforces.general_poly_stiffness import GenPolyForce


class TestGenPolyAFT(unittest.TestCase):

    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestGenPolyAFT, self).__init__(*args, **kwargs)      
        
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
        Q = np.array([[1, 0,0], \
                      [0, 1,0], \
                      [0, 0, 1 ]])
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[1, 0,0], \
                      [0, 1,0], \
                      [0, 0, 1 ]])
        
      
        qq = np.array([[2, 0, 0], \
                       [0, 2, 0], \
                       [3, 0, 0], \
                       [0, 0, 3],  \
                       [1, 1, 0], \
                       [0, 1, 1],  \
                       [2, 1, 0], \
                       [1, 1,1]])
            
        Emat = np.ones((3,qq.shape[0]))
            
        nl_force = GenPolyForce(Q, T, Emat, qq)
        
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
        
        # Third DOF, Sine Term, Fundamental
        U[2*Nd+2, 0] = 2
        
        w = 1 # Test for various w
        
        # Testing Simple First Harmonic Motion        
        Fnl, dFnldU = nl_force.aft(U, w, h)[0:2]
        
        # # Analytically Verify Force expansion:
        #4^3*cos^3+2^3*sin^3+ 4^2*cos^2+3^2*sin^2
        #4*3*cos*sin+3*2*sin*sin
        #4^2*3*cos^2*sin+4*3*2*cosx*sin^2
        
        #coefficients [ 0c 1c 1s 2c 2s 3c 3s 4c 4s 5c 6c 7c 7s ]
        #[15.5  54 18 0.5 6 10 10]
        
        
        temp = np.array ([15.5, 54, 18, 0.5, 6, 10, 10,0, 0, 0, 0, 0, 0, 0, 0])
        Fnl_analytical=np.repeat(temp, 3)


        
        
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