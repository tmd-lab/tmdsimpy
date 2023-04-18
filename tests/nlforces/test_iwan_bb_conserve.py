"""
Test of AFT implementation for the conservative Iwan backbone model

This model uses the loading curve from the 4-parameter Iwan model as a 
conservative stiffness
"""


import sys
import numpy as np
import unittest

# Python Utilities
sys.path.append('..')
import verification_utils as vutils

sys.path.append('../..')
import tmdsimpy.harmonic_utils as hutils
from tmdsimpy.nlforces.iwan_bb_conserve import ConservativeIwanBB


class TestIwanBB(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestIwanBB, self).__init__(*args, **kwargs)      
        
        
        analytical_tol_stuck = 1e-17 # Tolerance against analytical stuck
        analytical_tol_slip = 1e-9 # Fully slipped state tolerance
        
        rtol_grad = 1e-7 # Relative gradient tolerance
        
        high_amp_grad_rtol = 3e-5 # Relative tolerance for a specific case
        
        self.tols = (analytical_tol_stuck, analytical_tol_slip,\
                     rtol_grad, high_amp_grad_rtol)


        #######################################################################
        ###### System 1                                                  ######
        #######################################################################
        
        # Simple Mapping to spring displacements
        Q = np.array([[1.0, 0], \
                      [0, 1.0]])
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[1.0, 0.0], \
                      [0.0, 1.0]])
        
        kt = 2.0
        Fs = 3.0
        chi = 0.0
        beta = 0.0
        
        self.softening_force1 = ConservativeIwanBB(Q, T, kt, Fs, chi, beta)


        #######################################################################
        ###### System 2                                                  ######
        #######################################################################
                
        # Simple Mapping to spring displacements
        Q = np.array([[1.0, 0], \
                      [-1.0, 1.0], \
                      [0, 1.0]])
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[1.0, 0.25, 0.0], \
                      [0.0, 0.25, 1.0]])
        
        kt = 2.0
        Fs = 3.0
        chi = 0.0
        beta = 0.0
        
        self.softening_force2 = ConservativeIwanBB(Q, T, kt, Fs, chi, beta)


        #######################################################################
        ###### System 3                                                  ######
        #######################################################################

        kt = 2.0
        Fs = 3.0
        chi = 0.0
        beta = 0
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[1.0, 0.25, 0.0], \
                      [0.0, 0.25, 0.5]])
        
        self.softening_force3 = ConservativeIwanBB(Q, T, kt, Fs, chi, beta)
        
        
        self.parameters = (kt,Fs,chi,beta)
        
        
    def test_force_disp(self):
        """
        Test the force-displacement relationship
        
        Code is commented out for plotting for reference

        Returns
        -------
        None.

        """
        
        kt,Fs,chi,beta = self.parameters
        
        umax = 4*Fs/kt # Fs/kt is not the slip limit since this is Iwan not Jenkins
        
        # uplot = np.linspace(-umax, umax, 1001) # Multiple values for plotting
        
        uplot = np.array([-umax, 0, umax])
        fanalytical = np.array([-Fs, 0, Fs])
        
        fnlplot = self.softening_force1.local_force_history(uplot, 0*uplot)[0]
        
        self.assertLess(np.linalg.norm(fnlplot-fanalytical), 1e-16, 
                        'Iwan BB does not match expected force values.')
        
        """
        # Code for plotting the force displacement relationship
        import matplotlib.pyplot as plt

        plt.plot(uplot, fnlplot/Fs)
        plt.xlabel('Displacement')
        plt.ylabel('Force/Fs')
        plt.show()
        """
        
    def test_stuck_regime(self):
        """
        Test the solutions in the stuck regime against analytical expectations

        Returns
        -------
        None.

        """
        
        analytical_tol_stuck, analytical_tol_slip,\
                rtol_grad, high_amp_grad_rtol = self.tols
                 
        kt,Fs,chi,beta = self.parameters
        softening_force = self.softening_force1
        
        softening_force.kt = 2 
        softening_force.Fs = 1e16
        
        Q = softening_force.Q
        
        h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) 
        Nhc = hutils.Nhc(h)
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        
        Nd = Q.shape[1]
        
        U = np.zeros((Nd*Nhc, 1))
        
        # First DOF, Cosine Term, Fundamental
        U[Nd+0, 0] = 1e-2
        
        # Second DOF, Sine Term, Fundamental
        U[2*Nd+1, 0] = 1e-2
        
        w = 1 # Test for various w
        
        FnlH = softening_force.aft(U, w, h, Nt=1<<17)[0]
        
        FnlH_analytical = np.zeros_like(FnlH)
        
        # Cosine Term
        FnlH_analytical[Nd+0]    = softening_force.kt*U[Nd+0]
        
        # Sine Term
        FnlH_analytical[2*Nd+1]  = softening_force.kt*U[2*Nd+1] # 1st
        
        error =  np.linalg.norm(FnlH - FnlH_analytical)
        
        self.assertLess(error, analytical_tol_stuck, 
                        'Does not match expected stuck solution.')
        
        # Reset Fs and kt since the order of tests is not defined.
        softening_force.kt = kt
        softening_force.Fs = Fs
        
        
    def test_slipped_regime(self):
        """
        Test the solutions in the slipped regime against analytical expectations
        
        This case approaches a square wave.

        Returns
        -------
        None.

        """
        
        analytical_tol_stuck, analytical_tol_slip,\
                rtol_grad, high_amp_grad_rtol = self.tols
                 
        kt,Fs,chi,beta = self.parameters
        softening_force = self.softening_force1
        
        softening_force.kt = 1e16 
        softening_force.Fs = 0.1
        
        h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) 
        Nhc = hutils.Nhc(h)
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        
        Nd = softening_force.Q.shape[1]
        
        U = np.zeros((Nd*Nhc, 1))
        
        # First DOF, Cosine Term, Fundamental
        U[Nd+0, 0] = 1e16
        
        # Second DOF, Sine Term, Fundamental
        U[2*Nd+1, 0] = 1e16
        
        w = 1 # Test for various w
        
        FnlH = softening_force.aft(U, w, h, Nt=1<<17)[0]
        
        FnlH_analytical = np.zeros_like(FnlH)
        
        # Cosine Term
        FnlH_analytical[Nd+0]    = 4*softening_force.Fs/np.pi # 1st
        FnlH_analytical[5*Nd+0]  = -4/3*softening_force.Fs/np.pi # 3rd
        FnlH_analytical[9*Nd+0]  = 4/5*softening_force.Fs/np.pi # 5th 
        FnlH_analytical[13*Nd+0] = -4/7*softening_force.Fs/np.pi # 7th 
        
        # Sine Term
        FnlH_analytical[2*Nd+1]  = 4*softening_force.Fs/np.pi # 1st
        FnlH_analytical[6*Nd+1]  = 4/3*softening_force.Fs/np.pi # 3rd
        FnlH_analytical[10*Nd+1] = 4/5*softening_force.Fs/np.pi # 5th 
        FnlH_analytical[14*Nd+1] = 4/7*softening_force.Fs/np.pi # 7th 
        
        
        error =  np.linalg.norm(FnlH - FnlH_analytical)

        self.assertLess(error, analytical_tol_slip, 
                        'Does not match expected slipped solution.')
        
        # Reset Fs and kt since the order of tests is not defined.
        softening_force.kt = kt
        softening_force.Fs = Fs
        
    def test_grads_mdof1(self):
        """
        Test gradients with simple harmonic motion
        
        Multiple DOFs are used, but the center is fixed.

        Returns
        -------
        None.

        """
        
        # Saved Data
        analytical_tol_stuck, analytical_tol_slip,\
                rtol_grad, high_amp_grad_rtol = self.tols
                
        softening_force = self.softening_force2
        kt,Fs,chi,beta = self.parameters
                
        
        # h = np.array([0, 1, 2, 3]) # Manual Checking expansion / debugging
        h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) # Automate Checking with this
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        
        Nd = softening_force.Q.shape[1]
        
        U = np.zeros((Nd*Nhc, 1))
        
        # First DOF, Cosine Term, Fundamental
        U[Nd+0, 0] = 4
        
        # Second DOF, Sine Term, Fundamental
        U[2*Nd+1, 0] = 3
        
        w = 1 # Test for various w
        
        #######################
        # Mid Amplitude
        
        # Numerically Verify Gradient
        fun = lambda U: softening_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect mid amplitude displacement gradient.')
                
        # Numerically Verify Frequency Gradient
        fun = lambda w: softening_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=rtol_grad)

        self.assertFalse(grad_failed, 'Incorrect mid amplitude frequency gradient.')
        
        
        #######################
        # High Amplitude
        
        softening_force.Fs = 1e-2*Fs
        
        # Numerically Verify Gradient
        fun = lambda U: softening_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=high_amp_grad_rtol)
        
        self.assertFalse(grad_failed, 'Incorrect high amplitude displacement gradient.')
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: softening_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=rtol_grad)

        self.assertFalse(grad_failed, 'Incorrect high amplitude frequency gradient.')
        
        
        softening_force.Fs = Fs
        
        
        #######################
        # Low Amplitude
        
        softening_force.Fs = 1e5*Fs
        
        # Numerically Verify Gradient
        fun = lambda U: softening_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect low amplitude displacement gradient.')
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: softening_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, rtol=rtol_grad)

        self.assertFalse(grad_failed, 'Incorrect low amplitude frequency gradient.')
        
        softening_force.Fs = Fs


    def test_grads_mdof2(self):
        """
        Test gradients with more complex motion and interacting DOFs
        
        Also test skipping some harmonics

        Returns
        -------
        None.

        """
        
        # Saved Data
        analytical_tol_stuck, analytical_tol_slip,\
                rtol_grad, high_amp_grad_rtol = self.tols
                
        softening_force = self.softening_force3
        kt,Fs,chi,beta = self.parameters
                
        
        np.random.seed(1023)
        
        h = np.array([0, 1, 2, 3, 5, 6, 7]) # Automate Checking with this
        w = 1.375 # Test for various w
        
        
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        Nd = softening_force.Q.shape[1]
        
        U = np.random.rand(Nd*Nhc, 1)
        
        
        fun = lambda U: softening_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect displacement gradient.')
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: softening_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, 
                                        rtol=rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect frequency gradient.')
        
        
        ######################
        # Test without zeroth harmonic + skip 4th harmonic
        
        h = np.array([1, 2, 3, 5, 6, 7]) # Automate Checking with this
        
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        Nd = softening_force.Q.shape[1]
        
        U = np.random.rand(Nd*Nhc, 1)
        
        fun = lambda U: softening_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect displacement gradient.')
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: softening_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False,
                                        rtol=rtol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect frequency gradient.')


if __name__ == '__main__':
    unittest.main()