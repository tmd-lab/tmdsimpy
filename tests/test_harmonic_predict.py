"""
Test the initial guesses to HBM and HBM type equations to verify that 
they are reasonable linear approximations of the solutions.
"""

import sys
import numpy as np
import unittest

# Python Utilities
sys.path.append('..')
import tmdsimpy.harmonic_utils as hutils
from tmdsimpy.nlforces.cubic_stiffness import CubicForce # Just for VPRNM
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver


class TestHarmonicGuess(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a linear and a nonlinear system to use to test the 
        harmonic prediction guesses

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
        
        super(TestHarmonicGuess, self).__init__(*args, **kwargs)     
        
        ############################################
        # Generate Vibration Systems to Test
        
        M = np.array([[ 1.08, -0.41, -0.28],
                   [-0.41,  0.21,  0.07],
                   [-0.28,  0.07,  0.11]])
        
        K = np.array([[ 2.48, -0.97, -0.57],
                       [-0.97,  0.44,  0.17],
                       [-0.57,  0.17,  0.19]])
        
        vib_sys = VibrationSystem(M, K, ab=[0.01, 0.01])
        vib_sys_nl = VibrationSystem(M, K, ab=[0.01, 0.01])
        
        
        Q = np.array([[1, -1, 0.0]])
        T = Q.T
        kalpha = 0.05
        
        nl_force = CubicForce(Q, T, kalpha)
        
        vib_sys_nl.add_nl_force(nl_force)
        
        ############################################
        # Generate some harmonic information
        
        h_set1 = np.arange(1, 5)
        h_set2 = np.arange(0, 5)
        
        Ndof = M.shape[0]
        
        rng = np.random.default_rng(seed=42)
        
        # Forcing vectors - '_cos' = only harmonic 1 cosine
        # '_full' = includes sine
        Fl_set1_cos = np.zeros(Ndof*hutils.Nhc(h_set1))
        Fl_set1_full = np.zeros(Ndof*hutils.Nhc(h_set1))
        
        Fl_set1_cos[:Ndof] = rng.random(Ndof)-0.5
        Fl_set1_full[:2*Ndof] = rng.random(2*Ndof)-0.5
        
        Fl_set2_cos = np.zeros(Ndof*hutils.Nhc(h_set2))
        Fl_set2_full = np.zeros(Ndof*hutils.Nhc(h_set2))
        
        Fl_set2_cos[Ndof:2*Ndof] = rng.random(Ndof)-0.5
        Fl_set2_full[Ndof:3*Ndof] = rng.random(2*Ndof)-0.5
        
        ############################################
        # Store Data for all Tests
        self.data = (vib_sys, vib_sys_nl, h_set1, h_set2,
                     Fl_set1_cos, Fl_set1_full, Fl_set2_cos, Fl_set2_full)
        
    def test_hbm_predict(self):
        """
        Test the HBM initial guess
        """
        
        ###### Extract Data
        vib_sys, vib_sys_nl, h_set1, h_set2, \
            Fl_set1_cos, Fl_set1_full, Fl_set2_cos, Fl_set2_full = self.data
        
        solver = NonlinearSolver()
        
        w = 1.756
        fmag = 3.5
        
        ###### 1. Exact Linear Prediction w/ h0
        Ulam0 = hutils.predict_harmonic_solution(vib_sys, w, Fl_set1_full, 
                                                 h_set1, solver, 'HBM',
                                                 fmag=fmag)
        
        R = vib_sys.hbm_res(Ulam0, fmag*Fl_set1_full, h_set1)[0]
        
        self.assertLess(np.linalg.norm(R), 1e-13, 
                        'HBM prediction should be exact for linear system')

        ###### 2. Exact Linear Prediction w/o h0
        
        Ulam0 = hutils.predict_harmonic_solution(vib_sys, w, Fl_set2_full, 
                                                 h_set2, solver, 'HBM',
                                                 fmag=fmag)
        
        R = vib_sys.hbm_res(Ulam0, fmag*Fl_set2_full, h_set2)[0]
        
        self.assertLess(np.linalg.norm(R), 1e-13, 
                        'HBM prediction should be exact for linear system')
        
        
        ###### 3. Inexact, nonlinear Prediction w/ h0 and static displacements
        
        Xstat = np.array([1.0, 5.0, -3.7])
        
        Ulam0_stat = hutils.predict_harmonic_solution(vib_sys_nl, w, 
                                                      Fl_set2_full, 
                                                 h_set2, solver, 'HBM',
                                                 Xstat=Xstat,
                                                 fmag=fmag)
        
        Ndof = Xstat.shape[0]
        
        self.assertLess(np.linalg.norm(Ulam0[Ndof:] - Ulam0_stat[Ndof:]), 1e-12, 
                        'HBM Prediction - Xstat should not change h1')
        
        self.assertLess(np.linalg.norm(Xstat - Ulam0_stat[:Ndof]), 1e-12, 
                        'HBM Prediction - Xstat should be preserved')
        
    def test_hbm_amp_predict(self):
        """
        Test the HBM initial guess for amplitude controlled response
        """
        
        ###### Extract Data
        vib_sys, vib_sys_nl, h_set1, h_set2, \
            Fl_set1_cos, Fl_set1_full, Fl_set2_cos, Fl_set2_full = self.data
        
        solver = NonlinearSolver()
        
        w = 1.75
        
        ###### 1. Exact Linear Prediction w/ h0
        order = 0
        recov = np.array([1.0, 1.7, -1.0])
        amp = 3.4
        
        Ulam0 = hutils.predict_harmonic_solution(vib_sys, w, Fl_set1_full, 
                                                 h_set1, solver, 'HBM_AMP',
                                                 control_amp=amp, 
                                                 control_recov=recov, 
                                                 control_order=order)
        
        R = vib_sys.hbm_amp_control_res(Ulam0, Fl_set1_full, 
                                        h_set1, recov, amp, order)[0]
        
        self.assertLess(np.linalg.norm(R), 1e-13, 
                        'HBM prediction should be exact for linear system')

        ###### 2. Exact Linear Prediction w/o h0
        order = 1
        recov = np.array([1.0, 0.0, -1.0])
        
        Ulam0 = hutils.predict_harmonic_solution(vib_sys, w, Fl_set2_full, 
                                                 h_set2, solver, 'HBM_AMP',
                                                 control_amp=amp, 
                                                 control_recov=recov, 
                                                 control_order=order)
        
        R = vib_sys.hbm_amp_control_res(Ulam0, Fl_set2_full, 
                                        h_set2, recov, amp, order)[0]
        
        self.assertLess(np.linalg.norm(R), 1e-13, 
                        'HBM prediction should be exact for linear system')
        
        
        ###### 3. Inexact, nonlinear Prediction w/ h0 and static displacements
        
        Xstat = np.array([1.0, 5.0, -3.7])
        
        Ulam0_stat = hutils.predict_harmonic_solution(vib_sys_nl, w, Fl_set2_full, 
                                                 h_set2, solver, 'HBM_AMP',
                                                 control_amp=amp, 
                                                 control_recov=recov, 
                                                 control_order=order,
                                                 Xstat=Xstat)
        
        Ndof = Xstat.shape[0]
        
        self.assertLess(np.linalg.norm(Ulam0[Ndof:] - Ulam0_stat[Ndof:]), 1e-12, 
                        'HBM Prediction - Xstat should not change h1')
        
        self.assertLess(np.linalg.norm(Xstat - Ulam0_stat[:Ndof]), 1e-12, 
                        'HBM Prediction - Xstat should be preserved')
        
        
    def test_hbm_amp_phase_predict(self):
        """
        Test the HBM initial guess for amplitude + phase controlled response
        """
        
        ###### Extract Data
        vib_sys, vib_sys_nl, h_set1, h_set2, \
            Fl_set1_cos, Fl_set1_full, Fl_set2_cos, Fl_set2_full = self.data
        
        solver = NonlinearSolver()
        
        w = 1.75
        
        ###### 1. Exact Linear Prediction w/ h0
        order = 0
        recov = np.array([1.0, 1.7, -1.0])
        amp = 3.4
        
        Ulam0 = hutils.predict_harmonic_solution(vib_sys, w, Fl_set1_cos, 
                                             h_set1, solver, 'HBM_AMP_PHASE',
                                             control_amp=amp, 
                                             control_recov=recov, 
                                             control_order=order)
       
        R = vib_sys.hbm_amp_phase_control_res(Ulam0, Fl_set1_cos, 
                                        h_set1, recov, amp, order)[0]
        
        self.assertLess(np.linalg.norm(R), 1e-13, 
                        'HBM prediction should be exact for linear system')

        ###### 2. Exact Linear Prediction w/o h0
        order = 1
        recov = np.array([1.0, 0.0, -1.0])
        
        Ulam0 = hutils.predict_harmonic_solution(vib_sys, w, Fl_set2_cos, 
                                            h_set2, solver, 'HBM_AMP_PHASE',
                                            control_amp=amp, 
                                            control_recov=recov, 
                                            control_order=order)
        
        R = vib_sys.hbm_amp_phase_control_res(Ulam0, Fl_set2_cos, 
                                        h_set2, recov, amp, order)[0]
        
        self.assertLess(np.linalg.norm(R), 1e-13, 
                        'HBM prediction should be exact for linear system')
        
        
        ###### 3. Inexact, nonlinear Prediction w/ h0 and static displacements
        
        Xstat = np.array([1.0, 5.0, -3.7])
        
        Ulam0_stat = hutils.predict_harmonic_solution(vib_sys_nl, w, Fl_set2_cos, 
                                            h_set2, solver, 'HBM_AMP_PHASE',
                                            control_amp=amp, 
                                            control_recov=recov, 
                                            control_order=order,
                                            Xstat=Xstat)
        
        Ndof = Xstat.shape[0]
        
        self.assertLess(np.linalg.norm(Ulam0[Ndof:] - Ulam0_stat[Ndof:]), 1e-12, 
                        'HBM Prediction - Xstat should not change h1')
        
        self.assertLess(np.linalg.norm(Xstat - Ulam0_stat[:Ndof]), 1e-12, 
                        'HBM Prediction - Xstat should be preserved')
        
    def test_vprnm_amp_phase_predict(self):
        """
        Test the VPRNM initial guess for amplitude + phase controlled response
        """
        
        ###### Extract Data
        vib_sys, vib_sys_nl, h_set1, h_set2, \
            Fl_set1_cos, Fl_set1_full, Fl_set2_cos, Fl_set2_full = self.data
        
        solver = NonlinearSolver()
        
        w = 1.00
        order = 2
        recov = np.array([1.0, 0.0, -1.0])
        amp = 3.4
        rhi = 3
        
        Xstat = np.array([1.0, 5.0, -3.7])
        Ndof = Xstat.shape[0]
        
        ###### 1. Full VPRNM Prediction
        
        Ulam0 = hutils.predict_harmonic_solution(vib_sys, w, Fl_set2_cos, 
                                            h_set2, solver, 'VPRNM_AMP_PHASE',
                                            control_amp=amp, 
                                            control_recov=recov, 
                                            control_order=order,
                                            vib_sys_nl=vib_sys_nl,
                                            rhi=rhi,
                                            Xstat=Xstat)
        
        
        
        ###### 2. Check What is Possible
        
        Ulam0_hbm = hutils.predict_harmonic_solution(vib_sys_nl, w, Fl_set2_cos, 
                                            h_set2, solver, 'HBM_AMP_PHASE',
                                            control_amp=amp, 
                                            control_recov=recov, 
                                            control_order=order,
                                            Xstat=Xstat)
        
        self.assertLess(np.linalg.norm(Xstat - Ulam0[:Ndof]), 1e-12, 
                        'VPRNM Prediction - Xstat should be preserved')
        
        self.assertLess(np.linalg.norm(Ulam0[:3*Ndof] - Ulam0_hbm[:3*Ndof]), 
                        1e-12, 
                        'VPRNM Prediction - harmonics 0,1 should match HBM prediction')
        
        self.assertLess(np.linalg.norm(Ulam0[-4:] \
                                       - np.hstack((Ulam0_hbm[-3:], amp))), 
                        1e-12, 
                        'VPRNM Prediction - should match HBM for ending items')
        
        self.assertEqual(np.linalg.norm(Ulam0[3*Ndof:5*Ndof]), 0, 
                         'VPRNM should have no harmonic 2')
            
        
        self.assertEqual(np.linalg.norm(Ulam0[7*Ndof:9*Ndof]), 0, 
                         'VPRNM should have no harmonic 4')
        
        U_h3_expect = np.array([1.73363747,  3.72074871,  2.08535952, 
                                -0.92147125, -1.98943523, -1.10384826])
        
        # NOTE: This test does not say this prediction is correct, just that
        # it hasn't changed since it was first written.
        self.assertLess(np.linalg.norm(Ulam0[5*Ndof:7*Ndof] - U_h3_expect), 
                        1e-8, 
                        'VPRNM - superharmonic prediction has changed.')
        
if __name__ == '__main__':
    unittest.main()