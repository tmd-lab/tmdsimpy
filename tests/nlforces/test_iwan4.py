"""
Test and Verification of the hysteretic 4-parameter Iwan model


Items to verify
  1. Correct loading force displacement relationship 
      (against conservative implementation)
  2. The correct hysteretic forces (compare with Masing assumptions)
  2a. Consistent dissipation across discretization / range of N to choose?
  3. Correct derivatives of force at a given time instant with respect to 
      those displacements
  4. Correct harmonic derivaitves
  5. Test for a range of values of parameters

"""


import sys
import numpy as np
import unittest

# Python Utilities
sys.path.append('..')
import verification_utils as vutils

sys.path.append('../..')
import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.nlforces.iwan4_element import Iwan4Force
from tmdsimpy.nlforces.iwan_bb_conserve import ConservativeIwanBB


###############################################################################
#### Function for testing time series                                      ####
###############################################################################

def time_series_forces(Unl, h, Nt, w, iwan_force):
    
    # Unl = np.reshape(Unl, ((-1,1)))

    # Nonlinear displacements, velocities in time
    unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
    unltdot = w*hutils.time_series_deriv(Nt, h, Unl, 1) # Nt x Ndnl
    
    Nhc = hutils.Nhc(h)
    cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
    
    unlth0 = Unl[0]
    
    fnl, dfduh, dfdudh = iwan_force.local_force_history(unlt, unltdot, h, cst, unlth0)
    
    fnl = np.einsum('ij -> i', fnl)
    dfduh = np.einsum('ijkl -> il', dfduh)
    
    return fnl, dfduh


###############################################################################
#### Test Class      ####
###############################################################################

class TestIwan4(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestIwan4, self).__init__(*args, **kwargs)      
        
        
        analytical_tol = 1e-5 # Comparison to analytical backbones (v. this is discrete)
        
        analytical_tol_slip = 1e-4 # Fully slipped state tolerance
        
        atol_grad = 1e-9 # Absolute gradient tolerance
        
        self.tols = (analytical_tol, analytical_tol_slip, atol_grad)

        ##############################
        # Systems
                
        # Simple Mapping to spring displacements
        Q = np.array([[1.0]])
        T = np.array([[1.0]])
        
        kt = 2.0
        Fs = 3.0
        chi = -0.1
        beta = 0.1
        
        Nsliders = 500
        
        self.conservative_force = ConservativeIwanBB(Q, T, kt, Fs, chi, beta)
        self.hysteretic_force   = Iwan4Force(Q, T, kt, Fs, chi, beta, 
                                             Nsliders=Nsliders, alphasliders=1.0)

    def test_backbone_force(self):
        """
        Test that the backbone matches the analytical implementation

        Returns
        -------
        None.

        """
        analytical_tol, analytical_tol_slip, atol_grad = self.tols
        conservative_force = self.conservative_force
        hysteretic_force = self.hysteretic_force
        
        #######################################################################
        ###### Backbone Load-Displacement Relationship                   ######
        #######################################################################

        phimax = hysteretic_force.phimax

        amp = 1.01*hysteretic_force.phimax

        u_test = np.linspace(-amp, amp, 301)

        f_hyst = np.zeros_like(u_test)
        f_conv = np.zeros_like(u_test)

        hysteretic_force.init_history_harmonic(0)

        for i in range(len(u_test)):
            
            f_hyst[i] = hysteretic_force.instant_force(u_test[i], 0, update_prev=False)[0]
            
            f_conv[i] = conservative_force.local_force_history(u_test[i], 0)[0]


        error = np.max(np.abs(f_hyst - f_conv))
        
        self.assertLess(error, analytical_tol, 
                        'Iwan backbone does not match analytical solution.')

    def test_masing_force(self):
        """
        Test that analytically constructed hysteresis loops via Masing match 
        the implementation

        Also has commented out code for plotting at the end

        Returns
        -------
        None.

        """
        
        analytical_tol, analytical_tol_slip, atol_grad = self.tols
        conservative_force = self.conservative_force
        hysteretic_force = self.hysteretic_force
                
        #######################################################################
        ###### Masing Load-Displacement Relationship                     ######
        #######################################################################
        
        # Generate Time Series
        amp = 1.2*hysteretic_force.phimax
        
        Unl = amp*np.array([0, 1, 0])
        h = np.array([0, 1])
        Nt = 1 << 8
        
        w = 1.0
        Nhc = hutils.Nhc(h)
        
        # Nonlinear displacements, velocities in time
        unlt = hutils.time_series_deriv(Nt, h, Unl.reshape(-1, 1), 0) # Nt x Ndnl
        unltdot = w*hutils.time_series_deriv(Nt, h, Unl.reshape(-1, 1), 1) # Nt x Ndnl
        cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
        unlth0 = unlt[0, :]
        
        f_hyst = hysteretic_force.local_force_history(unlt, unltdot, h, cst, unlth0)[0]
        
        # Construct the hysteretic response from the backbone and the Masing conditions
        f_masing = np.zeros_like(f_hyst)
        
        u_unload = (amp - unlt[0:129])/2
        f_masing_unload = conservative_force.local_force_history(u_unload, 0*u_unload)[0]
        f_masing[0:128] = f_masing_unload[128] - f_masing_unload[0:128]*2
        
        u_reload = (amp + unlt[128:])/2
        f_masing_reload = conservative_force.local_force_history(u_reload, 0*u_reload)[0]
        f_masing[128:] = -f_masing_unload[128] + f_masing_reload*2
        
        
        error = np.max(np.abs(f_hyst - f_masing))
        
        self.assertLess(error, analytical_tol, 
                        'Iwan hysteresis does not match analytical solution (Masing).')

        # Plotting Code
        """
        
        import matplotlib.pyplot as plt

        # Quick Manual Check That the Force Makes Sense, can be commented out etc for 
        # automated testing. 
        plt.plot(unlt/hysteretic_force.phimax, f_hyst/hysteretic_force.Fs, label='Iwan Force')
        plt.ylabel('Displacement/phi max')
        plt.xlabel('Iwan Force/Fs')
        plt.xlim((-1.1*amp/hysteretic_force.phimax, 1.1*amp/hysteretic_force.phimax))
        plt.ylim((-1.1, 1.1))
        # plt.legend()
        plt.show()

        """

    def test_time_series_deriv(self):
        """
        Tests of the time series derivatives for the Iwan model

        Returns
        -------
        None.

        """
        
        
        analytical_tol, analytical_tol_slip, atol_grad = self.tols
        hysteretic_force = self.hysteretic_force
        
        phimax = hysteretic_force.phimax
        
        Nt = 1 << 7
        
        h = np.array([0, 1, 2, 3])
        Unl = phimax*np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
        
        w = 1.0
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, hysteretic_force)
        
        
        # Basic with some slipping: 
        h = np.array([0, 1])
        Unl = phimax*np.array([[0.75, 0.2, 1.3]]).T
        fun = lambda Unl : time_series_forces(Unl, h, Nt, w, hysteretic_force)
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, atol=atol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect time gradient for displacement')
        
        # Lots of harmonics and slipping check
        h = np.array([0, 1, 2, 3])
        Unl = phimax*np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
        fun = lambda Unl : time_series_forces(Unl, h, Nt, w, hysteretic_force)
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, atol=atol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect time gradient for displacement')
        
        # Stuck Check
        h = np.array([0, 1, 2, 3])
        Unl = phimax*np.array([[0.1, -0.1, 0.3, 0.1, 0.05, -0.1, 0.1]]).T
        fun = lambda Unl : time_series_forces(Unl, h, Nt, w, hysteretic_force)
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, atol=atol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect time gradient for displacement')
        
        
    def test_aft_deriv(self):
        """
        Tests of the derivatives from AFT procedure for the Iwan model

        Returns
        -------
        None.

        """
        
        analytical_tol, analytical_tol_slip, atol_grad = self.tols
        hysteretic_force = self.hysteretic_force
        
                
        w = 2.7
        
        #######################
        # Basic with some slipping: 
        h = np.array([0, 1])
        U = np.array([[0.75, 0.2, 1.3]]).T
        fun = lambda U : hysteretic_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad)

        self.assertFalse(grad_failed, 'Incorrect displacement gradient.')

        Fnl, dFnldU = fun(U)
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: hysteretic_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, atol=atol_grad)

        self.assertFalse(grad_failed, 'Incorrect frequency gradient.')
        
        #######################
        # Lots of harmonics and slipping check
        h = np.array([0, 1, 2, 3])
        U = np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
        fun = lambda U : hysteretic_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad)

        self.assertFalse(grad_failed, 'Incorrect displacement gradient.')
        
        Fnl, dFnldU = fun(U)
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: hysteretic_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, atol=atol_grad)

        self.assertFalse(grad_failed, 'Incorrect frequency gradient.')
        
        #######################
        # Stuck Check
        h = np.array([0, 1, 2, 3])
        U = np.array([[0.1, -0.1, 0.3, 0.1, 0.05, -0.1, 0.1]]).T
        fun = lambda U : hysteretic_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad)

        self.assertFalse(grad_failed, 'Incorrect displacement gradient.')
        
        Fnl, dFnldU = fun(U)
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: hysteretic_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, atol=atol_grad)

        self.assertFalse(grad_failed, 'Incorrect frequency gradient.')
        
        #######################
        # Lots of harmonics and slipping check
        h = np.array([0, 1, 2, 3])
        U = np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        fun = lambda U : hysteretic_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad)

        self.assertFalse(grad_failed, 'Incorrect displacement gradient.')
        
        Fnl, dFnldU = fun(U)
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: hysteretic_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, atol=atol_grad)

        self.assertFalse(grad_failed, 'Incorrect frequency gradient.')
        
        #######################
        # Limit of Full Slip Analytical Check
        h = np.array([0, 1, 2, 3])
        U = np.array([[0.0, 1e30, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        fun = lambda U : hysteretic_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, atol=atol_grad)

        self.assertFalse(grad_failed, 'Incorrect displacement gradient.')
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: hysteretic_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, atol=atol_grad)

        self.assertFalse(grad_failed, 'Incorrect frequency gradient.')
        
                
    def test_fully_slipped_force(self):
        """
        Compare AFT to the analytical solution from a fully slipped case
        Fully slipped approaches square wave.

        Returns
        -------
        None.

        """
               
        analytical_tol, analytical_tol_slip, atol_grad = self.tols
        hysteretic_force = self.hysteretic_force
        Fs = hysteretic_force.Fs
        
        
        w = 2.7
        h = np.array([0, 1, 2, 3])
        U = np.array([[0.0, 1e30, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        
        # Need lots of AFT Points to accurately converge slipping state:
        fun = lambda U : hysteretic_force.aft(U, w, h, Nt=1<<17)[0:2]
        Fnl, dFnldU = fun(U)
        
        force_error = np.abs(Fnl - np.array([0, 0.0, -4*Fs/np.pi, 0.0, 0.0, 0.0, -4*Fs/np.pi/3])).max()
                
        self.assertLess(force_error, analytical_tol_slip, 
                        'Iwan does not match analytical fully slipping force.')
        
        
    def test_static_force(self):
        """
        The global static force against expectation values and check 
        derivatives

        Returns
        -------
        None.

        """
        
        static_tol = self.tols[0]
        atol_grad = self.tols[2]
        
        # Simple Mapping to spring displacements
        Q = np.array([[1.0, -1.0]])
        T = np.array([[1.0], [-1.0]])
        
        kt = 2.0
        Fs = 3.0
        chi = -0.1
        beta = 0.1
        
        Nsliders = 500
        
        conservative_force = ConservativeIwanBB(Q, T, kt, Fs, chi, beta)
        hysteretic_force   = Iwan4Force(Q, T, kt, Fs, chi, beta, 
                                             Nsliders=Nsliders, alphasliders=1.0)
        
        phimax = hysteretic_force.phimax
        
        x0 = np.array([0.5*phimax, 0.0]) # initial loading
        x1 = np.array([1.5*phimax, 0.0]) # Maximum loading
        x2 = np.array([0.0, -0.5*phimax]) # Alternative to initial that should give same result
        
        hysteretic_force.init_history()
        
        f0_ref = conservative_force.force(x0)[0]
        f1_ref = conservative_force.force(x1)[0]
        f2_ref = f1_ref - 2*f0_ref # Masing Assumption (Tangent displacements strategically chosen)
        
        ##########################
        # Check at maximum displacement
        f1_test = hysteretic_force.force(x1, update_hist=False)[0]
        
        self.assertLess(np.linalg.norm(f1_test - f1_ref), static_tol,
                        'Second Static Iwan Force is Wrong.')
        self.assertLess(np.linalg.norm(f1_test - np.array([Fs, -Fs])), static_tol,
                        'Second Static Iwan Force is Wrong.')
        
        
        fun = lambda X: hysteretic_force.force(X, update_hist=False)
        grad_failed = vutils.check_grad(fun, x1, verbose=False, atol=atol_grad)
        self.assertFalse(grad_failed, 'Incorrect first static gradient.')
        
        #########################
        # First Point, should match without history update 
        f0_test = hysteretic_force.force(x0, update_hist=False)[0]
        
        self.assertLess(np.linalg.norm(f0_test - f0_ref), static_tol,
                        'Initial Static Iwan Force is Wrong.')
        
        fun = lambda X: hysteretic_force.force(X, update_hist=False)
        grad_failed = vutils.check_grad(fun, x0, verbose=False, atol=atol_grad)
        self.assertFalse(grad_failed, 'Incorrect second static gradient.')
        
        ########################
        # Update history to maximum displacement
        f1_test = hysteretic_force.force(x1, update_hist=True)[0]
        f2_test = hysteretic_force.force(x2, update_hist=False)[0]
        
        self.assertLess(np.linalg.norm(f2_test - f2_ref), static_tol,
                        'Unloading Static Iwan Force is Wrong.')

        fun = lambda X: hysteretic_force.force(X, update_hist=False)
        grad_failed = vutils.check_grad(fun, x2, verbose=False, atol=atol_grad)
        self.assertFalse(grad_failed, 'Incorrect unloading static gradient.')
        
        # Only update history after checking gradient
        f2_test = hysteretic_force.force(x2, update_hist=True)[0]
        
        #######################
        # Should Reset to the initial loading point
        hysteretic_force.init_history()
        f0_test = hysteretic_force.force(x0, update_hist=False)[0]
        
        self.assertLess(np.linalg.norm(f0_test - f0_ref), static_tol,
                        'Initial Static Iwan Force is wrong after using reset to history.')

        fun = lambda X: hysteretic_force.force(X, update_hist=False)
        grad_failed = vutils.check_grad(fun, x0, verbose=False, atol=atol_grad)
        self.assertFalse(grad_failed, 'Incorrect final static gradient.')
        
        #######################
        # Reset history for other tests
        hysteretic_force.init_history()

        return


if __name__ == '__main__':
    unittest.main()