"""
Test AFT Implementation of Cubic and Quadratic Polynomial Nonlinearity 
(Used for geometric nonlinearity)
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
            
        Emat = np.ones((Q.shape[1],qq.shape[0]))
            
        nl_force = GenPolyForce(Q, T, Emat, qq)

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
        Fnl_analytical = np.repeat(temp, 3)
     
        error = np.linalg.norm(Fnl - Fnl_analytical)
        
        self.assertLess(error, self.analytical_tol, 
                        'Quintic stiffness does not match analytical AFT.')
        
    def test_force(self):   
        """
        Test analytical expansion of nonlinear force 
        
        Also check gradients for the simple case

        Returns
        -------
        None.

        """

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
        Nd = Q.shape[1]
            
        nl_force = GenPolyForce(Q, T, Emat, qq)
        
        """
        Case 1: 
            -Compare to analytical expansion
        """        
        Nd = Q.shape[1]
        u0 = [2,3,4]
            
        # Move to new function + numerically check this gradient
        # Verify Nonlinear force function analytical
        Fnl_f, dFnldU_f = nl_force.force(u0) #output from function
        
        # Analytical solution
        Fnl_a=Emat @ np.array([4,9,8,64,6,12,12,24])
        dFnldU_a = Emat @  np.array([[4, 0, 12, 0, 3, 0, 12, 12],\
                     [0, 6, 0, 0, 2, 4, 4, 8],\
                     [0, 0, 0, 48, 0, 3, 0, 6]]).T 
        
        error_force = np.max(np.abs(Fnl_f - Fnl_a))
        error_J = np.max(np.abs(dFnldU_f - dFnldU_a))
        
        force_failed = (error_force > 1e-10)
        J_failed = (error_J > 1e-10)
        
        self.assertFalse((force_failed > 1e-10), 
            'Incorrect force evaluation from nl_force.force, analytical check')
        
        self.assertFalse((J_failed > 1e-10), 
            'Incorrect jacobian evaluation from nl_force.force, analytical check')
        
        """
        Case 2: 
            -Compare to Numerical gradient
        """
        rng = np.random.default_rng()
        u0 = rng.random((Nd))
        Fnl_f, dFnldU_f = nl_force.force(u0) #output from function 
        Emat = rng.random((Nd,1))
        
        grad_check = vutils.check_grad(nl_force.force, u0, verbose=False, 
                                        rtol=100*self.rtol_grad)    
        
        self.assertFalse(grad_check, 
            'Incorrect jacobian evaluation from nl_force.force, numerical check')
              
    def test_aftgradient(self):   
        """
        Test analytical expansion of nonlinear force gradient for first harmonic motion
         for the simple case. Also validates numerical gradient for random values of u and Emat
        
        Returns
        -------
        None.
        """
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
        Nd = Q.shape[1]
            
        nl_force = GenPolyForce(Q, T, Emat, qq)
        
        # h = np.array([0, 1, 2, 3]) # Manual Checking expansion / debugging
        h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) # Automate Checking with this
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        w = 1 # Test for various w
        
        Nd = Q.shape[1]
       
        #U[:3] = 1.0
        rng = np.random.default_rng()
        U = rng.random((Nd*Nhc, 1))
        
        # Numerically Verify Gradient 
        fun = lambda U: nl_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                        rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 
                         'Incorrect gradiant w.r.t. displacement.')
        
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
        
        # h = np.array([0, 1, 2, 3]) # Manual Checking expansion / debugging
        h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) # Automate Checking with this
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components            
        Emat = np.ones((3,qq.shape[0]))
        Nd = Q.shape[1]
            
        nl_force = GenPolyForce(Q, T, Emat, qq)
        
        rng = np.random.default_rng()
        
        U = rng.random((Nd*Nhc, 1))
        
        w = 1.375 # Test for various w
        
        nl_force = GenPolyForce(Q, T, Emat, qq)
        
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
        
    def test_zeroU_input(self):    
        """
        Test derivatives when the input displacement is zero. Earlier gradiaent
        calculation in matlab was not calculating correct gradient at u=0   

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
            
        Emat = np.ones((Q.shape[1],qq.shape[0]))
            
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
        nl_force = GenPolyForce(Q, T, Emat, qq)        
        # Numerically Verify Gradient 
        fun = lambda U: nl_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                       rtol=self.rtol_grad)
           
        self.assertFalse(grad_failed, 
                        'Incorrect gradiant w.r.t. displacement when disp is 0.')       


if __name__ == '__main__':
    unittest.main()