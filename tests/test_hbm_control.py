"""
Test for verifying the correctness of Harmonic Balance Method with amplitude 
control constraint

""" 

import sys
import numpy as np
from scipy import io as sio
import unittest

import verification_utils as vutils

sys.path.append('..')
from tmdsimpy.nlforces.cubic_stiffness import CubicForce

from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy import harmonic_utils as hutils


class TestHarmonicBalanceControl(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Check if MATLAB/Python integration is available and import MATLAB if 
        needed
        
        Also initialize the tolerances all here at the beginning

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        super(TestHarmonicBalanceControl, self).__init__(*args, **kwargs)       
        
        # Tolerances
        self.grad_rtol = 5e-10
        self.nearlin_tol = 1e-13 # Tolerance for linear analytical v. HBM check
        
        
        #######################################################################
        ###### Setup Nonlinear System                                    ######
        #######################################################################
        
        M = np.array([[6.12, 3.33, 4.14],
                      [3.33, 4.69, 3.42],
                      [4.14, 3.42, 3.7 ]])
        
        K = np.array([[3.0, 0.77, 1.8 ],
                        [0.77, 2.48, 1.71],
                        [1.8 , 1.71, 2.51]])
        
        ab_damp = [0.0001, 0.0003]
        
        vib_sys_lin = VibrationSystem(M, K, ab=ab_damp)
        
        self.vib_sys_lin = vib_sys_lin
        
        ###########################
        # Setup Nonlinear Force
        
        # Simple Mapping to spring displacements
        Q = np.array([[-1.0, 1.0, 0.0]])
        
        # Weighted / integrated mapping back for testing purposes
        # MATLAB implementation only supported T = Q.T for instantaneous forcing.
        T = np.array([[-1.0], \
                      [1.0], \
                      [0.0] ])
        
        kalpha = np.array([3.2])
        
        duff_force = CubicForce(Q, T, kalpha)
        
        vib_sys_nl = VibrationSystem(M, K, ab=ab_damp)
        
        vib_sys_nl.add_nl_force(duff_force)
        
        self.vib_sys_nl = vib_sys_nl

    def test_disp_control(self):
        """
        Steps:
            1. Linear FRF to get a point
            2. Try amplitude control residual with linear system and verify all
            zero residuals
            3. Check Gradients of linear system
            4. Try solution with nonlinear system and verify last residual is 
                zero
            5. Verify other NL residuals match HBM
            6. Check Gradients with NL system
        """
        
        ####### Initialization of Parameters for this test
        
        h = np.array(range(6))
        recov = np.array([2.0, 0.0, 0.0])
        
        w = 0.75
        control_order = 0 # displacement = zeroth derivative
        
        Flcos1 = np.array([1.0, 2.0, 3.0])
        Fmag = 1.5
        
        ####### Test Cases Initialization Starts here
        solver = NonlinearSolver()
        vib_sys_lin = self.vib_sys_lin
        vib_sys_nl  = self.vib_sys_nl
        
        Ndof = vib_sys_lin.M.shape[0]
        Nhc = hutils.Nhc(h)
        h0 = h[0] == 0
        
        # Full External Force vector
        Fl = np.zeros(Ndof*Nhc)
        Fl[h0*Ndof:(1+h0)*Ndof] = Flcos1
        
        ####### 1. Linear Solution from FRF
        Xw_linear = vib_sys_lin.linear_frf(w, Fmag*Flcos1, solver, neigs=3)
        
        amp_sq = (recov @ Xw_linear[0, :Ndof])**2 \
                    + (recov @ Xw_linear[0, Ndof:2*Ndof])**2
        
        ####### 2. Amplitude Control Residual of Linear System
        
        UFw = np.zeros(Ndof*Nhc+2)
        UFw[h0*Ndof:(2+h0)*Ndof] = Xw_linear[0, :-1]
        UFw[-2] = Fmag
        UFw[-1] = w
        
        R,_,_ = vib_sys_lin.hbm_amp_control_res(UFw, Fl, h, recov, 
                                                (w**control_order)*np.sqrt(amp_sq), 
                                                control_order)
        
        self.assertLess(np.linalg.norm(R), self.nearlin_tol, 
                        'Linear FRF solution is not satisfying linear HBM residual')
        
        ####### 3. Gradient Checks on Linear System
        
        # Displacement/Force Gradient
        fun = lambda UF : vib_sys_lin.hbm_amp_control_res(
                                np.hstack((UF, UFw[-1])), Fl, h, recov, 
                                (w**control_order)*np.sqrt(amp_sq), 
                                control_order)[0:2]
        
        grad_failed = vutils.check_grad(fun, UFw[:-1], rtol=self.grad_rtol, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for UF.')        
        
        # Frequency Gradient 
        fun = lambda w : vib_sys_lin.hbm_amp_control_res(
                                            np.hstack((UFw[:-1], w)), Fl, h, 
                                            recov, 
                                            (w**control_order)*np.sqrt(amp_sq), 
                                            control_order)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(UFw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')   
        
        ####### 4+5. Check Residuals for the Nonlinear System

        Uw = np.zeros(Ndof*Nhc+1)
        Uw[h0*Ndof:(2+h0)*Ndof] = Xw_linear[0, :-1]
        Uw[-1] = w
        
        R,_,_ = vib_sys_nl.hbm_amp_control_res(UFw, Fl, h, recov, 
                                                (w**control_order)*np.sqrt(amp_sq), 
                                                control_order)
        
        Rhbm,_,_ = vib_sys_nl.hbm_res(Uw, Fmag*Fl, h)
        
        self.assertLess(np.linalg.norm(R[:-1] - Rhbm), 1e-12, 
                        'Nonlinear residual is not matching HBM residual')
        
        self.assertLess(np.abs(R[-1]), 1e-12, 
                        'Nonlinear system is not satisfying amplitude constraint.')
        
        ####### 6. Check Gradients for the Nonlinear System

        # Displacement/Force Gradient
        fun = lambda UF : vib_sys_nl.hbm_amp_control_res(
                                np.hstack((UF, UFw[-1])), Fl, h, recov, 
                                (w**control_order)*np.sqrt(amp_sq), 
                                control_order)[0:2]
        
        grad_failed = vutils.check_grad(fun, UFw[:-1], rtol=self.grad_rtol, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for UF.')        
        
        # Frequency Gradient 
        fun = lambda w : vib_sys_nl.hbm_amp_control_res(
                                            np.hstack((UFw[:-1], w)), Fl, h, 
                                            recov, 
                                            (w**control_order)*np.sqrt(amp_sq), 
                                            control_order)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(UFw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')   
        
        return
    
    
    def test_accel_control(self):
        """
        Repeat displacement control, but applying control to acceleration 
        amplitude instead
        
        Choose a different point than displacement control to get more coverage
        
        Also use a different recovery matrix + different set of harmonics
        
        Steps:
            1. Linear FRF to get a point
            2. Try amplitude control residual with linear system and verify all
            zero residuals
            3. Check Gradients of linear system
            4. Try solution with nonlinear system and verify last residual is 
                zero
            5. Verify other NL residuals match HBM
            6. Check Gradients with NL system
        """
        
        h = np.array(range(1, 6))
        recov = np.array([0.0, -1.5, 2.0])
        
        w = 0.5
        control_order = 2 # acceleration = second derivative
        
        Flcos1 = np.array([0.0, 2.0, -1.0])
        Fmag = 2.5
        
        ####### Test Cases Initialization Starts here
        solver = NonlinearSolver()
        vib_sys_lin = self.vib_sys_lin
        vib_sys_nl  = self.vib_sys_nl
        
        Ndof = vib_sys_lin.M.shape[0]
        Nhc = hutils.Nhc(h)
        h0 = h[0] == 0
        
        # Full External Force vector
        Fl = np.zeros(Ndof*Nhc)
        Fl[h0*Ndof:(1+h0)*Ndof] = Flcos1
        
        ####### 1. Linear Solution from FRF
        Xw_linear = vib_sys_lin.linear_frf(w, Fmag*Flcos1, solver, neigs=3)
        
        amp_sq = (recov @ Xw_linear[0, :Ndof])**2 \
                    + (recov @ Xw_linear[0, Ndof:2*Ndof])**2
        
        ####### 2. Amplitude Control Residual of Linear System
        
        UFw = np.zeros(Ndof*Nhc+2)
        UFw[h0*Ndof:(2+h0)*Ndof] = Xw_linear[0, :-1]
        UFw[-2] = Fmag
        UFw[-1] = w
        
        R,_,_ = vib_sys_lin.hbm_amp_control_res(UFw, Fl, h, recov, 
                                                (w**control_order)*np.sqrt(amp_sq), 
                                                control_order)
        
        self.assertLess(np.linalg.norm(R), self.nearlin_tol, 
                        'Linear FRF solution is not satisfying linear HBM residual')
        
        ####### 3. Gradient Checks on Linear System
        
        # Displacement/Force Gradient
        fun = lambda UF : vib_sys_lin.hbm_amp_control_res(
                                np.hstack((UF, UFw[-1])), Fl, h, recov, 
                                (w**control_order)*np.sqrt(amp_sq), 
                                control_order)[0:2]
        
        grad_failed = vutils.check_grad(fun, UFw[:-1], rtol=self.grad_rtol, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for UF.')        
        
        # Frequency Gradient 
        fun = lambda w_input : vib_sys_lin.hbm_amp_control_res(
                                            np.hstack((UFw[:-1], w_input)), Fl, 
                                            h, recov, 
                                            (w**control_order)*np.sqrt(amp_sq), 
                                            control_order)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(UFw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')   
        
        ####### 4+5. Check Residuals for the Nonlinear System

        Uw = np.zeros(Ndof*Nhc+1)
        Uw[h0*Ndof:(2+h0)*Ndof] = Xw_linear[0, :-1]
        Uw[-1] = w
        
        R,_,_ = vib_sys_nl.hbm_amp_control_res(UFw, Fl, h, recov, 
                                                (w**control_order)*np.sqrt(amp_sq), 
                                                control_order)
        
        Rhbm,_,_ = vib_sys_nl.hbm_res(Uw, Fmag*Fl, h)
        
        self.assertLess(np.linalg.norm(R[:-1] - Rhbm), 1e-12, 
                        'Nonlinear residual is not matching HBM residual')
        
        self.assertLess(np.abs(R[-1]), 1e-12, 
                        'Nonlinear system is not satisfying amplitude constraint.')
        
        ####### 6. Check Gradients for the Nonlinear System

        # Displacement/Force Gradient
        fun = lambda UF : vib_sys_nl.hbm_amp_control_res(
                                np.hstack((UF, UFw[-1])), Fl, h, recov, 
                                (w**control_order)*np.sqrt(amp_sq), 
                                control_order)[0:2]
        
        grad_failed = vutils.check_grad(fun, UFw[:-1], rtol=self.grad_rtol, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for UF.')        
        
        # Frequency Gradient 
        fun = lambda w_input : vib_sys_nl.hbm_amp_control_res(
                                            np.hstack((UFw[:-1], w_input)), Fl, 
                                            h, recov, 
                                            (w**control_order)*np.sqrt(amp_sq), 
                                            control_order)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(UFw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')   
        
        return
    
    def test_static_force(self):
        """
        Test that force scaling gets appropriately applied when there are 
        static forces
        
        Steps:
            1. Compare R with two different HBM residuals
            2. Check Gradients, specifically w.r.t. Fmag

        Returns
        -------
        None.

        """
        
        Ndof = self.vib_sys_nl.M.shape[0]
        h = np.arange(4)
        Nhc = hutils.Nhc(h)
        
        Fl = np.zeros(Nhc*Ndof)
        Fl[:Ndof] = 1.345
        Fl[Ndof:3*Ndof] = 3.756
        
        Uw = np.array([0.14367379, 0.36867602, 0.39050928, 0.98579746, 0.41899098,
                       0.81947508, 0.97431731, 0.07685235, 0.92784135, 0.50964209,
                       0.21553171, 0.34303005, 0.78459336, 0.71293194, 0.24523768,
                       0.69202978, 0.93502781, 0.98957847, 0.85421524, 0.27126554,
                       0.64560423, 1.3151])
        
        UFw = np.hstack((Uw[:-1], 2.0, Uw[-1]))
        
        #############
        # Check Scaling on Static Force
        
        Fstat = np.copy(Fl)
        Fstat[Ndof:] = 0.0
        
        Fdyn = np.copy(Fl)
        Fdyn[:Ndof] = 0.0
        
        Rhbm = self.vib_sys_nl.hbm_res(Uw, Fstat + Fdyn*UFw[-2], h)[0]
        
        order = 0
        amp = 1.0
        Recov = np.ones(3)
        
        Rtest = self.vib_sys_nl.hbm_amp_control_res(UFw, Fl, h, Recov, amp, order)[0]
        
        self.assertLess(np.linalg.norm(Rhbm - Rtest[:-1]), 1e-12,
                        'Static force is inappropriately scaled.')
        
        #############
        # Check Gradients with Static Forces
        
        # UF gradient (special focus on the last column w.r.t. F)
        fun = lambda UF : self.vib_sys_nl.hbm_amp_control_res(
                                                np.hstack((UF, UFw[-1])), 
                                                Fl, h, Recov, amp, order)[0:2]
        
        grad_failed = vutils.check_grad(fun, UFw[:-1], rtol=self.grad_rtol, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for UF.') 
        
        
        # Frequency Gradient
        fun = lambda w : self.vib_sys_nl.hbm_amp_control_res(
                                            np.hstack((UFw[:-1], w)), 
                                            Fl, h, Recov, amp, order)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(UFw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')   
        
        
    
if __name__ == '__main__':
    unittest.main()
    
        