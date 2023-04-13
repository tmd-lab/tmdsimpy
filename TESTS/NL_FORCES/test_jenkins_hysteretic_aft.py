"""
Verification of hysteretic implementation of AFT and of the implementation of 
the Jenkins element
"""


import sys
import numpy as np
import unittest

sys.path.append('../')
import verification_utils as vutils

# Python Utilities
sys.path.append('../../ROUTINES/')
sys.path.append('../../ROUTINES/NL_FORCES')
import harmonic_utils as hutils

from jenkins_element import JenkinsForce




###############################################################################
#### Function To Wrap History                                              ####
###############################################################################

def modify_history_fun(jenkins_force, u, udot, up, fp):
    
    jenkins_force.up = up
    jenkins_force.fp = fp
    
    fnl,dfnldunl,dfnldup,dfnldfp = jenkins_force.instant_force(u, udot, update_prev=False)
    
    return fnl,dfnldunl,dfnldup,dfnldfp


###############################################################################
#### Verification of Force Time Serives Derivatives w.r.t. Harmonics       ####
###############################################################################

def time_series_forces(Unl, h, Nt, w, jenkins_force):
    
    # Unl = np.reshape(Unl, ((-1,1)))

    # Nonlinear displacements, velocities in time
    unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
    unltdot = w*hutils.time_series_deriv(Nt, h, Unl, 1) # Nt x Ndnl
    
    Nhc = hutils.Nhc(h)
    cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
    
    unlth0 = Unl[0]
    
    fnl, dfduh, dfdudh = jenkins_force.local_force_history(unlt, unltdot, h, cst, unlth0)
    
    fnl = np.einsum('ij -> i', fnl)
    dfduh = np.einsum('ijkl -> il', dfduh)
    
    return fnl, dfduh

###############################################################################
#### Test Class                                                            ####
###############################################################################
class TestJenkins(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestJenkins, self).__init__(*args, **kwargs)      
        
        
        self.analytical_sol_tol_stick = 1e-14 # Tolerance comparing to analytical solution
        self.analytical_sol_tol_slip  = 1e-4 # Tolerance comparing to analytical solution
        
        self.atol_grad = 1e-10 # Absolute gradient tolerance, a few places use 10* this
        
        self.atol_hyst = 1e-15

        #########################
        ### Initialize the Jenkins force and other parameters
        
        # Simple Mapping to displacements
        Q = np.array([[1.0]])
        T = np.array([[1.0]])
        
        kt = 2.0
        Fs = 3.0
        umax = 5.0
        freq = 1.4 # rad/s
        Nt = 1<<8
        
        self.jenkins_force = JenkinsForce(Q, T, kt, Fs)
        
        self.parameters = (umax, freq, Nt, Fs, kt, Q, T)
        
    def test_force_history(self):
        """
        Test the hysteresis loop against expectations of the values.

        Returns
        -------
        None.

        """
        
        umax, freq, Nt, Fs, kt, Q, T = self.parameters
            
        self.jenkins_force.init_history(unlth0=0)
        
        # # For an example of the full hysteresis loop, use these displacements:
        # t = np.linspace(0, 2*np.pi, Nt+1)
        # t = t[:-1]
        # u = umax*np.sin(t)
        # udot = umax*freq*np.cos(t)
        
        
        u = np.array([0, umax, umax-Fs/kt, umax-2*Fs/kt, -umax, 
                      -umax+Fs/kt, -umax+2*Fs/kt, 0])
        f_analytical = np.array([0, Fs, 0, -Fs, -Fs, 0, Fs, Fs])
        
        udot = np.zeros_like(u)
        
        fhist = np.zeros_like(u)
        
        for indext in range(u.shape[0]):
            fhist[indext],_,_,_ = self.jenkins_force.instant_force(u[indext], udot[indext], update_prev=True)
            
        self.assertLess(np.linalg.norm(fhist-f_analytical), self.atol_hyst, 
                        'Jenkins is not reproducing the expected hysteresis loop.')
                    
        ####################
        #### Extra code for plotting the Jenkins force history for understanding
        #### and to serve as an example.
        
        # import matplotlib.pyplot as plt
        # plt.plot(u, fhist, label='Jenkins Force')
        # plt.ylabel('Jenkins Displacement [m]')
        # plt.xlabel('Jenkins Force [N]')
        # plt.xlim((-1.1*umax, 1.1*umax))
        # plt.ylim((-1.1*Fs, 1.1*Fs))
        # # plt.legend()
        # plt.show()

        
    def test_time_series_derivatives(self):
        """
        Test derivatives of Jenkins at time instants for a complete cycle

        Returns
        -------
        None.

        """
        
        umax, freq, Nt, Fs, kt, Q, T = self.parameters
        
        jenkins_force = self.jenkins_force 
                        
        #######################################################################
        #### Verification of Derivatives of Jenkins Force                  ####
        #######################################################################
        
        # jenkins_force.init_history(unlth0=0)
        Nt = 1<<5
        t = np.linspace(0, 2*np.pi, Nt+1)
        t = t[:-1]
        
        u = umax*np.sin(t)
        udot = umax*freq*np.cos(t)
        
        fhist = np.zeros_like(u)
        
        
        jenkins_force.init_history(unlth0=0)
        
        for indext in range(t.shape[0]):
            fnl,dfnldunl,dfnldup,dfnldfp = jenkins_force.instant_force(u[indext], udot[indext], update_prev=False)
            
            up = jenkins_force.up
            fp = jenkins_force.fp
            
            # Check U derivatives
            fun = lambda U: jenkins_force.instant_force(U, udot[indext], update_prev=False)[0:2]
            
            grad_failed = vutils.check_grad(fun, np.array([u[indext]]), 
                                            verbose=False, atol=self.atol_grad)
            
            self.assertFalse(grad_failed, 
                             'Jenkins gradient w.r.t. displacement failed.')
            
            # Check U previous Derivative
            jenkins_force.up = up
            jenkins_force.fp = fp
            fun = lambda Up: modify_history_fun(jenkins_force, u[indext], udot[indext], Up, fp)[0:3:2]
            
            grad_failed = vutils.check_grad(fun, np.array([up]), 
                                            verbose=False, atol=self.atol_grad)
            
            # print('Index: {}, up={}, fp={}'.format(indext, up, fp))
            self.assertFalse(grad_failed, 
                             'Jenkins gradient w.r.t. previous displacement failed.')
            
            # Check F previous derivative
            jenkins_force.up = up
            jenkins_force.fp = fp
            fun = lambda Fp: modify_history_fun(jenkins_force, u[indext], udot[indext], up, Fp)[0:4:3]
            
            grad_failed = vutils.check_grad(fun, np.array([fp]), 
                                            verbose=False, atol=self.atol_grad)
            
            self.assertFalse(grad_failed, 
                             'Jenkins gradient w.r.t. previous force failed.')
            
            # Update History so derivatives can be checked at the next state.
            jenkins_force.up = up
            jenkins_force.fp = fp
            fhist[indext],_,_,_ = jenkins_force.instant_force(u[indext], udot[indext], update_prev=True)

    def test_harmonic_derivs(self):
        """
        Test of the derivatives w.r.t. harmonic coefficients for some simple 
        cases

        Returns
        -------
        None.

        """
                
        # Unpack parameters from before
        umax, freq, Nt, Fs, kt, Q, T = self.parameters
        jenkins_force = self.jenkins_force 
        
        Nt = 1 << 7
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
        
        w = freq
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
        
        
        # Basic with some slipping: 
        h = np.array([0, 1])
        Unl = np.array([[0.75, 0.2, 1.3]]).T
        fun = lambda Unl : time_series_forces(Unl, h, Nt, w, jenkins_force)
        grad_failed = vutils.check_grad(fun, Unl, verbose=False,
                                        atol=self.atol_grad)
        
        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. displacement failed.')
        
        
        # Lots of harmonics and slipping check
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
        fun = lambda Unl : time_series_forces(Unl, h, Nt, w, jenkins_force)
        grad_failed = vutils.check_grad(fun, Unl, verbose=False,
                                        atol=self.atol_grad*10)
        
        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. displacement failed.')
        
        # Stuck Check
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.1, -0.1, 0.3, 0.1, 0.05, -0.1, 0.1]]).T
        fun = lambda Unl : time_series_forces(Unl, h, Nt, w, jenkins_force)
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, 
                                        atol=self.atol_grad)
        
        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. displacement failed.')
        
        
    def test_full_aft_cases(self):
        """
        Test several different cases for derivatives using full AFT
        
        Also tests some different regimes where analytical aft results are 
        possible.

        Returns
        -------
        None.

        """      
        
        # Unpack parameters from before
        umax, freq, Nt, Fs, kt, Q, T = self.parameters
        jenkins_force = self.jenkins_force 
        
        w = 2.7
        
        #############
        # Basic with some slipping: 
        h = np.array([0, 1])
        U = np.array([[0.75, 0.2, 1.3]]).T
        fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                        atol=self.atol_grad)
        
        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. displacement failed.')
        
        Fnl, dFnldU = fun(U)
        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: jenkins_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, 
                                        atol=self.atol_grad)


        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. frequency failed.')
        
        #############
        # Lots of harmonics and slipping check
        h = np.array([0, 1, 2, 3])
        U = np.array([[0.75, 0.2, 1.3, 2, 3, 4, 5]]).T
        fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                        atol=self.atol_grad)


        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. displacement failed.')

        Fnl, dFnldU = fun(U)
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: jenkins_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, 
                                        atol=self.atol_grad)


        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. frequency failed.')
        
        #############
        # Stuck Check
        h = np.array([0, 1, 2, 3])
        U = np.array([[0.1, -0.1, 0.3, 0.1, 0.05, -0.1, 0.1]]).T
        fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                        atol=self.atol_grad)

        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. displacement failed.')        
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: jenkins_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, 
                                        atol=self.atol_grad)

        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. frequency failed.')        
        
        fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
        Fnl, dFnldU = fun(U)
        
        stiffness_error = np.abs(dFnldU - np.diag(np.array([0., 1., 1., 1., 1., 1., 1.])*kt)).max()
        force_error = np.abs(Fnl / U.T / kt - np.array([0, 1, 1, 1, 1, 1, 1])).max()
        
        
        self.assertLess(stiffness_error, self.analytical_sol_tol_stick, 
                        'Jenkins does not match stick stiffness.')
        
        self.assertLess(force_error, self.analytical_sol_tol_stick, 
                        'Jenkins does not match stick force.')
                
        # Lots of harmonics and slipping check
        h = np.array([0, 1, 2, 3])
        U = np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False, 
                                        atol=self.atol_grad)


        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. displacement failed.')
        Fnl, dFnldU = fun(U)
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: jenkins_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, 
                                        atol=self.atol_grad)


        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. frequency failed.')
        
        
        # Limit of Full Slip Analytical Check
        h = np.array([0, 1, 2, 3])
        U = np.array([[0.0, 1e30, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        fun = lambda U : jenkins_force.aft(U, w, h)[0:2]
        grad_failed = vutils.check_grad(fun, U, verbose=False,
                                        atol=self.atol_grad*10)

        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. displacement failed.')
        
        # Numerically Verify Frequency Gradient
        fun = lambda w: jenkins_force.aft(U, w[0], h)[0::2]
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, 
                                        atol=self.atol_grad)

        self.assertFalse(grad_failed, 
                         'Jenkins gradient w.r.t. frequency failed.')       
        
        # Need lots of AFT Points to accurately converge Jenkins:
        fun = lambda U : jenkins_force.aft(U, w, h, Nt=1<<17)[0:2]
        Fnl, dFnldU = fun(U)
        
        force_error = np.abs(Fnl - np.array([0, 0.0, -4*Fs/np.pi, 0.0, 0.0, 0.0, -4*Fs/np.pi/3])).max()
        
        self.assertLess(force_error, self.analytical_sol_tol_slip, 
                        'Jenkins does not match slip force.')
                
        

if __name__ == '__main__':
    unittest.main()