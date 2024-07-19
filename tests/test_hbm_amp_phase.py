"""
Test for verifying the correctness of Harmonic Balance Method (HBM)
with amplitude and phase control constraints
""" 

import sys
import numpy as np
import unittest

import verification_utils as vutils

sys.path.append('..')
from tmdsimpy.nlforces.cubic_stiffness import CubicForce

from tmdsimpy.vibration_system import VibrationSystem
import tmdsimpy.utils.harmonic as hutils


class TestHBMPhaseAmp(unittest.TestCase):
    
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a linear and nonlinear system for later tests to use.

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
        
        super(TestHBMPhaseAmp, self).__init__(*args, **kwargs)       
        
        # Tolerances
        self.grad_rtol = 5e-10
        
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
        Test that displacement control gives appropriate results
        """
        
        # Size of the problem
        vib_sys = self.vib_sys_nl
        h = np.array([0, 1, 2, 3, 5])
        Ndof = vib_sys.M.shape[0]
        Nhc = hutils.Nhc(h)
        
        # Generate needed quantities
        rng = np.random.default_rng(seed=1023)
        
        UFcFsw = rng.random(Ndof*Nhc+3) - 0.5
        UFcFsw[-1] = 1.453 # Fix positive frequency
        
        Fl = rng.random(Ndof*Nhc) - 0.5
        recov = np.array([1.0, -2.0, 0.567])
        amp = 1.9
        order = 0
        
        # Baseline solution
        Rbase = vib_sys.hbm_amp_phase_control_res(UFcFsw, Fl, h, recov, amp, 
                                                  order)[0]
        
        # Check residual values for amplitude constraints here.
        amp_cos = recov @ UFcFsw[Ndof:2*Ndof]
        amp_sin = recov @ UFcFsw[2*Ndof:3*Ndof]
        
        self.assertLess(np.linalg.norm((amp_cos-amp) - Rbase[-2]), 1e-11, 
                        'Cosine amplitude constraint looks wrong.')
        
        self.assertLess(np.linalg.norm(amp_sin - Rbase[-1]), 1e-11, 
                        'Sine amplitude constraint looks wrong.')
        
        # Check gradients
        
        # UF gradient (special focus on the last column w.r.t. F)
        fun = lambda UFcFs : self.vib_sys_nl.hbm_amp_phase_control_res(
                                                np.hstack((UFcFs, UFcFsw[-1])), 
                                                Fl, h, recov, amp, order)[0:2]
        
        grad_failed = vutils.check_grad(fun, UFcFsw[:-1], rtol=self.grad_rtol, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for UFcFs.') 
        
        
        # Frequency Gradient
        
        fun = lambda w : self.vib_sys_nl.hbm_amp_phase_control_res(
                                            np.hstack((UFcFsw[:-1], w)), 
                                            Fl, h, recov, amp, order)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(UFcFsw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')   
        
        
    def test_accel_control(self):
        """
        Test that acceleration amplitude control gives appropriate results
        """
        
        # Size of the problem
        vib_sys = self.vib_sys_nl
        h = np.array([0, 1, 2, 3, 5])
        Ndof = vib_sys.M.shape[0]
        Nhc = hutils.Nhc(h)
        
        # Generate needed quantities
        rng = np.random.default_rng(seed=1023)
        
        UFcFsw = rng.random(Ndof*Nhc+3) - 0.5
        UFcFsw[-1] = 1.453 # Fix positive frequency
        
        Fl = rng.random(Ndof*Nhc) - 0.5
        recov = np.array([1.0, -2.0, 0.567])
        amp = 1.9
        order = 2
        
        # Baseline solution
        Rbase = vib_sys.hbm_amp_phase_control_res(UFcFsw, Fl, h, recov, amp, 
                                                  order)[0]
        
        # Check residual values for amplitude constraints here.
        amp_cos = (UFcFsw[-1]**order) * recov @ UFcFsw[Ndof:2*Ndof]
        amp_sin = (UFcFsw[-1]**order) * recov @ UFcFsw[2*Ndof:3*Ndof]
        
        self.assertLess(np.linalg.norm((amp_cos-amp) - Rbase[-2]), 1e-11, 
                        'Cosine amplitude constraint looks wrong.')
        
        self.assertLess(np.linalg.norm(amp_sin - Rbase[-1]), 1e-11, 
                        'Sine amplitude constraint looks wrong.')
        
        # Check gradients
        fun = lambda UFcFs : self.vib_sys_nl.hbm_amp_phase_control_res(
                                                np.hstack((UFcFs, UFcFsw[-1])), 
                                                Fl, h, recov, amp, order)[0:2]
        
        grad_failed = vutils.check_grad(fun, UFcFsw[:-1], rtol=self.grad_rtol, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for UFcFs.') 
        
        
        # Frequency Gradient
        
        fun = lambda w : self.vib_sys_nl.hbm_amp_phase_control_res(
                                            np.hstack((UFcFsw[:-1], w)), 
                                            Fl, h, recov, amp, order)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(UFcFsw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')   
        
        
    def test_scale_force(self):
        """
        Test that amplitude control gives appropriate results
        """
        
        # Size of the problem
        vib_sys = self.vib_sys_nl
        h = np.array([0, 1, 2, 3, 5])
        Ndof = vib_sys.M.shape[0]
        Nhc = hutils.Nhc(h)
        
        # Generate needed quantities
        rng = np.random.default_rng(seed=1023)
        
        UFcFsw = rng.random(Ndof*Nhc+3) - 0.5
        UFcFsw[-1] = 1.453 # Fix positive frequency
        
        Fl = rng.random(Ndof*Nhc) - 0.5
        recov = np.array([1.0, -2.0, 0.567])
        amp = 1.9
        order = 1
        
        # Baseline solution
        Rbase = vib_sys.hbm_amp_phase_control_res(UFcFsw, Fl, h, recov, amp, 
                                                  order)[0]
        
        # Check baseline is preserved for new Fl
        Fl_alt = np.copy(Fl)
        Fl_alt[2*Ndof:] = rng.random(Ndof*(Nhc-2))+0.5
        
        Ralt = vib_sys.hbm_amp_phase_control_res(UFcFsw, Fl_alt, h, recov, amp, 
                                                  order)[0]
        
        self.assertEqual(np.linalg.norm(Rbase-Ralt), 0.0,
             'HBM Amp/Phase Control depends on some wrong quantities in Force')
        
        # Check with varied unknowns -> expected variation in Rbase
        UFcFsw_alt = np.copy(UFcFsw)
        
        UFcFsw_alt[-3] += 2.5
        UFcFsw_alt[-2] -= 1.7
        
        Ralt = vib_sys.hbm_amp_phase_control_res(UFcFsw_alt, Fl, h, recov, amp, 
                                                  order)[0]
        
        self.assertEqual(np.linalg.norm(Ralt[:Ndof] - Rbase[:Ndof]), 0.0, 
                         'Zeroth harmonic should not change with force scaling')
        
        Rexpect = Rbase[Ndof:2*Ndof] - 2.5*Fl[Ndof:2*Ndof]
        
        self.assertLess(np.linalg.norm(Ralt[Ndof:2*Ndof] - Rexpect), 1e-11,
                        'Force for cosine does not appear correctly scaled.')
        
        Rexpect = Rbase[2*Ndof:3*Ndof] + 1.7*Fl[Ndof:2*Ndof]
        
        self.assertLess(np.linalg.norm(Ralt[2*Ndof:3*Ndof] - Rexpect), 1e-11,
                        'Force for sine does not appear correctly scaled.')
        
    def test_calc_grad(self):
        """
        Test that the calc_grad flag is used appropriately.
        """
        
        # Size of the problem
        vib_sys = self.vib_sys_nl
        h = np.array([0, 1, 2, 3, 5])
        Ndof = vib_sys.M.shape[0]
        Nhc = hutils.Nhc(h)
        
        # Generate needed quantities
        rng = np.random.default_rng(seed=1023)
        
        UFcFsw = rng.random(Ndof*Nhc+3) - 0.5
        UFcFsw[-1] = 1.453 # Fix positive frequency
        
        Fl = rng.random(Ndof*Nhc) - 0.5
        recov = np.array([1.0, -2.0, 0.567])
        amp = 1.9
        order = 2
        
        # Baseline solution
        Rdefault = vib_sys.hbm_amp_phase_control_res(UFcFsw, Fl, h, recov, amp, 
                                                     order)
        
        # With calc_grad
        Rtrue = vib_sys.hbm_amp_phase_control_res(UFcFsw, Fl, h, recov, amp, 
                                                     order, calc_grad=True)
        
        # Without calc_grad
        Rfalse = vib_sys.hbm_amp_phase_control_res(UFcFsw, Fl, h, recov, amp, 
                                                     order, calc_grad=False)
        
        # Check correct number of outputs
        self.assertEqual(len(Rdefault), 3, 
                         'Default calc_grad should have 3 ouputs')
        
        self.assertEqual(len(Rtrue), 3, 
                         'calc_grad=True should have 3 ouputs')
        
        self.assertEqual(len(Rfalse), 1, 
                         'calc_grad=False should have 3 ouputs')
        
        # Check outputs are the same for all three
        self.assertEqual(np.linalg.norm(Rdefault[0] - Rtrue[0]), 0.0,
                         'calc_grad changed residual value.')
        
        self.assertEqual(np.linalg.norm(Rdefault[0] - Rfalse[0]), 0.0,
                         'calc_grad changed residual value.')
        
        self.assertEqual(np.linalg.norm(Rdefault[1] - Rtrue[1]), 0.0,
                         'calc_grad changed gradient value.')
        
        self.assertEqual(np.linalg.norm(Rdefault[2] - Rtrue[2]), 0.0,
                         'calc_grad changed gradient value.')
        
    def test_dA_fun(self):
        """
        Test the HBM amplitude/phase constrained residual function for 
        continuation w.r.t. amplitude
        """
        
        # Size of the problem
        vib_sys = self.vib_sys_nl
        h = np.array([0, 1, 2, 3, 5])
        Ndof = vib_sys.M.shape[0]
        Nhc = hutils.Nhc(h)
        
        # Generate needed quantities
        rng = np.random.default_rng(seed=1023)
        
        UFcFsw = rng.random(Ndof*Nhc+3) - 0.5
        UFcFsw[-1] = 1.453 # Fix positive frequency
        
        Fl = rng.random(Ndof*Nhc) - 0.5
        recov = np.array([1.0, -2.0, 0.567])
        amp = 1.9
        order = 1
        
        # Baseline solution
        Rbase, dRdUbase, _ = vib_sys.hbm_amp_phase_control_res(UFcFsw, Fl, h, 
                                                               recov, amp, 
                                                               order)
        
        # Check that the residual / gradients are consistent where identical
        UFcFsA = np.hstack((UFcFsw[:-1], amp))
        
        Ramp, dRdUamp, _ = vib_sys.hbm_amp_phase_control_dA_res(UFcFsA, 
                                                      Fl, h, recov, UFcFsw[-1], 
                                                      order)
        
        self.assertEqual(np.linalg.norm(Rbase-Ramp), 0.0,
                         'Amplitude continuation function changes residual.')
        
        self.assertEqual(np.linalg.norm(dRdUbase-dRdUamp), 0.0,
                         'Amplitude continuation function changes gradient.')
        
        # Numerically check gradients
        fun = lambda UFcFs : vib_sys.hbm_amp_phase_control_dA_res( 
                                                      np.hstack((UFcFs, amp)),
                                                      Fl, h, recov, UFcFsw[-1], 
                                                      order)[0:2]
        
        grad_failed = vutils.check_grad(fun, UFcFsw[:-1], rtol=self.grad_rtol, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for UFcFs.') 
        
        
        # Frequency Gradient
        
        fun = lambda amp : vib_sys.hbm_amp_phase_control_dA_res( 
                                                np.hstack((UFcFsA[:-1], amp)),
                                                Fl, h, recov, UFcFsw[-1], 
                                                order)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(UFcFsw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for amplitude.')   
        
        
    def test_dA_calc_grad(self):
        """
        Test the calc_grad flag for the HBM amplitude constrained solution
        for continuation w.r.t. amplitude
        """
        
        # Size of the problem
        vib_sys = self.vib_sys_nl
        h = np.array([0, 1, 2, 3, 5])
        Ndof = vib_sys.M.shape[0]
        Nhc = hutils.Nhc(h)
        
        # Generate needed quantities
        rng = np.random.default_rng(seed=1023)
        
        UFcFsA = rng.random(Ndof*Nhc+3) - 0.5
        UFcFsA[-1] = 1.453 # Fix positive frequency
        
        Fl = rng.random(Ndof*Nhc) - 0.5
        recov = np.array([1.0, -2.0, 0.567])
        freq = 1.9
        order = 2
        
        # Baseline solution
        Rdefault = vib_sys.hbm_amp_phase_control_dA_res(UFcFsA, Fl, h, recov, 
                                                        freq, order)
        
        # With calc_grad
        Rtrue = vib_sys.hbm_amp_phase_control_dA_res(UFcFsA, Fl, h, recov, 
                                                 freq, order, calc_grad=True)
        
        # Without calc_grad
        Rfalse = vib_sys.hbm_amp_phase_control_dA_res(UFcFsA, Fl, h, recov, 
                                                freq, order, calc_grad=False)
        
        # Check correct number of outputs
        self.assertEqual(len(Rdefault), 3, 
                         'Default calc_grad should have 3 ouputs')
        
        self.assertEqual(len(Rtrue), 3, 
                         'calc_grad=True should have 3 ouputs')
        
        self.assertEqual(len(Rfalse), 1, 
                         'calc_grad=False should have 3 ouputs')
        
        # Check outputs are the same for all three
        self.assertEqual(np.linalg.norm(Rdefault[0] - Rtrue[0]), 0.0,
                         'calc_grad changed residual value.')
        
        self.assertEqual(np.linalg.norm(Rdefault[0] - Rfalse[0]), 0.0,
                         'calc_grad changed residual value.')
        
        self.assertEqual(np.linalg.norm(Rdefault[1] - Rtrue[1]), 0.0,
                         'calc_grad changed gradient value.')
        
        self.assertEqual(np.linalg.norm(Rdefault[2] - Rtrue[2]), 0.0,
                         'calc_grad changed gradient value.')
        
    
if __name__ == '__main__':
    unittest.main()