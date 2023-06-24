"""
Test of Alternating Frequency-Time (AFT) with cubic damping nonlinearity

Includes analytical checks of cubic damping results for first harmonic motion.


System (all cubic dampers, fixed boundaries):
    Only checks AFT, so don't need the system to be physical by adding springs.
    /|        + ----+        + ----+        |\
    /|---c1---| M1  |---c2---| M2  |---c3---|\
    /|        +-----+        +-----+        |\


"""
import sys
import numpy as np
import unittest

# Python Utilities
sys.path.append('..')
import verification_utils as vutils

sys.path.append('../..')
from tmdsimpy.nlforces.cubic_damping import CubicDamping



class TestCubicDamping(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestCubicDamping, self).__init__(*args, **kwargs)      
        
        
        self.analytical_sol_tol = 1e-12 # Tolerance comparing to analytical solution
        
        self.rtol_grad = 1e-10 # Relative gradient tolerance
        
        
        
    def test_fundamental_motion(self):
        """
        Analytically check outputs with fundamental harmonic motion and
        check the gradients for this case.

        Returns
        -------
        None.

        """
        
        #####################
        # Simple Mapping to spring displacements
        Q = np.array([[1.0, 0], \
                      [-1.0, 1.0], \
                      [0, 1.0]])
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[1.0, 0.25, 0.0], \
                      [0.0, 0.25, 1.0]])
        
        calpha = np.array([3, 0, 7])
        
        nl_force = CubicDamping(Q, T, calpha)
        
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
        
        w = 1.732 # Test for various w, not 1.0 so velocity is clearly different from displacement.
        
        # Calculation for fundamental harmonic motion
        Fnl, dFnldU, dFnldw = nl_force.aft(U, w, h)
        
        # # Analytically Verify Force expansion:
        # # X^3*cos^3(x) = X^3*( 3/4*cos(x) + 1/4*cos(3x) )
        # # X^3*sin^3(x) = X^3*(3/4*sin(x) - 1/4*sin(3x)
        # # With Velocity shift
        # #     cos -> -sin
        # #     sin -> cos
        Fnl_analytical = np.zeros_like(Fnl) 
        Fnl_analytical[2*Nd+0] = -T[0,0]*( 0.75*calpha[0]*(Q[0,0]*U[Nd+0, 0]*w)**3 )
        Fnl_analytical[6*Nd+0] = T[0,0]*( 0.25*calpha[0]*(Q[0,0]*U[Nd+0, 0]*w)**3 )
        
        Fnl_analytical[1*Nd+1] = T[1,2]*( 0.75*calpha[2]*(Q[2,1]*U[2*Nd+1, 0]*w)**3 )
        Fnl_analytical[5*Nd+1] = T[1,2]*( 0.25*calpha[2]*(Q[2,1]*U[2*Nd+1, 0]*w)**3 )
        
        analytical_error = np.linalg.norm(Fnl - Fnl_analytical)
        
        self.assertLess(analytical_error, self.analytical_sol_tol, 
                        'Solution does not match analytical AFT for cubic damping.')
        
        
        # Numerically Verify Gradient
        fun = lambda U: nl_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=self.rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect gradient w.r.t. displacements.')
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: nl_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=self.rtol_grad)

        self.assertFalse(grad_failed, 'Incorrect gradient w.r.t. frequency.')

    def test_mixed_harmonics(self):
        """
        Check the gradients again for a few different cases to make sure they 
        are correct.
        
        Considers mixed combinations of harmonics.

        Returns
        -------
        None.

        """
        
        # np.random.seed(42)
        np.random.seed(1023)
        # np.random.seed(0)
        w = 1.732 # Test for various w, not 1.0 so velocity is clearly different from displacement.


        h = np.array([0, 1, 2, 3, 5, 6, 7]) # Automate Checking with this
        calpha = np.array([3, 5, 7])
        
        # Simple Mapping to spring displacements
        Q = np.array([[1.0, 0], \
                      [-1.0, 1.0], \
                      [0, 1.0]])
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[1.0, 0.25, 0.0], \
                      [0.0, 0.25, 0.5]])
        
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        Nd = Q.shape[1]
        
        U = np.random.rand(Nd*Nhc, 1)
        
        
        nl_force = CubicDamping(Q, T, calpha)
        
        fun = lambda U: nl_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=self.rtol_grad)
        self.assertFalse(grad_failed, 'Incorrect gradient w.r.t. displacements.')
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: nl_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=self.rtol_grad)
        self.assertFalse(grad_failed, 'Incorrect gradient w.r.t. frequency.')
        
        ######################
        # Test without zeroth harmonic + skipping 4th harmonic
        h = np.array([1, 2, 3, 5, 6, 7]) # Automate Checking with this
        
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        Nd = Q.shape[1]
        
        U = np.random.rand(Nd*Nhc, 1)
        
        fun = lambda U: nl_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=self.rtol_grad)
        self.assertFalse(grad_failed, 'Incorrect gradient w.r.t. displacements.')
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: nl_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=self.rtol_grad)
        self.assertFalse(grad_failed, 'Incorrect gradient w.r.t. frequency.')


if __name__ == '__main__':
    unittest.main()