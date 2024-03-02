"""
Unit test of Phase Resonance Nonlinear Modes residual function (SDOF)

Steps:
    1. Create Systems (Duffing + Jenkins)
    2. Verify Gradients
    3. Verify Residual accuracy for phase condition
    4. Test at a few known solution points (residual = 0)
    5. Test for cubic damping for freq. gradient
    6. Test MDOF for Duffing and Cubic Damping
"""

import sys
import numpy as np
import unittest


sys.path.append('..')

from tmdsimpy.vibration_system import VibrationSystem

from tmdsimpy.nlforces.cubic_stiffness import CubicForce
from tmdsimpy.nlforces.vector_jenkins import VectorJenkins
from tmdsimpy.nlforces.cubic_damping import CubicDamping

import tmdsimpy.harmonic_utils as hutils

sys.path.append('../DEPENDENCIES/tmdsimpy/tests/')
import verification_utils as vutils


class TestVPRNM(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        
        super(TestVPRNM, self).__init__(*args, **kwargs)
        
        #######################################################################
        # Duffing Initialization
        
        m = 1 # kg
        c = 0.01 # kg/s
        k = 1 # N/m
        knl = 1 # N/m^3
        
        # Nonlinear Force
        Q = np.array([[1.0]])
        T = np.array([[1.0]])
        
        kalpha = np.array([knl])
        
        duff_force = CubicForce(Q, T, kalpha)
        
        
        # Setup Vibration System
        M = np.array([[m]])
        
        K = np.array([[k]])
        
        ab_damp = [c/m, 0]
        
        vib_sys_duffing = VibrationSystem(M, K, ab=ab_damp)
        
        vib_sys_duffing.add_nl_force(duff_force)
        
        self.vib_sys_duffing = vib_sys_duffing
        
        #######################################################################
        # Jenkins Initialization
        
        m = 1 # kg
        c = 0.01 # kg/s
        k = 0.75 # N/m
        kt = 0.25 # N/m
        Fs = 0.2 # N
        
        vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)
        
        
        # Setup Vibration System
        M = np.array([[m]])
        
        K = np.array([[k]])
        
        ab_damp = [c/m, 0]
        
        vib_sys_jenkins = VibrationSystem(M, K, ab=ab_damp)
        
        vib_sys_jenkins.add_nl_force(vector_jenkins_force)

        self.vib_sys_jenkins = vib_sys_jenkins
        
        
        #######################################################################
        # Cubic Damping Initialization
            
        m = 1 # kg
        c = 0.01 # kg/s
        k = 1 # N/m
        cnl = 0.03 # N/(m/s)^3 = N s^3 / m^3 = kg s / m^2
        
        calpha = np.array([cnl])
        
        nl_damping = CubicDamping(Q, T, calpha)
        
        # Setup Vibration System
        M = np.array([[m]])
        
        K = np.array([[k]])
        
        ab_damp = [c/m, 0]
        
        vib_sys_cubic_damp = VibrationSystem(M, K, ab=ab_damp)
        
        vib_sys_cubic_damp.add_nl_force(nl_damping)

        self.vib_sys_cubic_damp = vib_sys_cubic_damp
        
        #######################################################################
        # 2 DOF Duffing Initialization
        
        M = np.array([[2, 1], [1, 3]])
        K = np.array([[5, -1], [-1, 2]])
        
        knl = 1 # N/m^3
        
        # Nonlinear Force
        Q = np.array([[1.0, -0.5]])
        T = np.array([[1.0], [-0.5]])
        
        kalpha = np.array([knl])
        
        duff_force = CubicForce(Q, T, kalpha)
        
        
        # Setup Vibration System
        
        ab_damp = [0.01, 0]
        
        vib_sys_duffing = VibrationSystem(M, K, ab=ab_damp)
        
        vib_sys_duffing.add_nl_force(duff_force)
        
        self.vib_sys_duffing_2dof = vib_sys_duffing
        
        #######################################################################
        # 3 DOF Cubic Damping Initialization
        
        M = np.array([[2, 1, 0], [1, 3, 1.5], [0, 1.5, 4.2]])
        K = np.array([[5, -1, 0], [-1, 2, -1.7], [0, -1.7, 4.2]])
        
        calpha = 1 # N/m^3
        
        # Nonlinear Force
        Q = np.array([[1.0, -0.5, 0.0]])
        T = np.array([[1.0], [-0.5], [0.0]])
        
        kalpha = np.array([knl])
        
        nl_damping = CubicDamping(Q, T, calpha)
        
        
        # Setup Vibration System
        
        ab_damp = [0.01, 0]
        
        vib_sys_cubic_damp = VibrationSystem(M, K, ab=ab_damp)
        
        vib_sys_cubic_damp.add_nl_force(nl_damping)

        self.vib_sys_cubic_damp_3dof = vib_sys_cubic_damp
        

    def test_gradient_duffing(self):
        """
        Numerical Gradient Check with higher harmonic phase calculated by AFT
        for duffing
        """
        
        # Inputs for PRNM
        h = np.array(range(6))
        rhi = 3
        UwF0 = np.zeros(hutils.Nhc(h)+2)
        
        UwF0[0] = 0.1 # Static
        UwF0[1] = 0.5 # Cosine Fundamental
        UwF0[2] = -0.3 # Sine Fundamental
        
        UwF0[5] = 0.5 # Cosine 3rd
        UwF0[6] = -0.3 # Sine 3rd
        
        UwF0[-2] = 1.4
        UwF0[-1] = 2.0
        
        Fl = np.zeros_like(UwF0[:-2])
        Fl[1] = 1
        
        # Check of dRdUw
        fun = lambda Uw : self.vib_sys_duffing.vprnm_res(
                                    np.hstack((Uw, UwF0[-1])), h, rhi, Fl)[0:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[:-1], verbose=False, rtol=5e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. Uw')
        
        
        # Check of dRdF
        fun = lambda F : self.vib_sys_duffing.vprnm_res(
                                    np.hstack((UwF0[:-1], F)), h, rhi, Fl)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[-1:], verbose=False, rtol=1e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. force magnitude')
        
    def test_gradient_jenkins(self):
        """
        Numerical Gradient Check with higher harmonic phase calculated by AFT
        for Jenkins
        """
        
        
        # Inputs for PRNM
        h = np.array(range(6))
        rhi = 3
        UwF0 = np.zeros(hutils.Nhc(h)+2)
        
        UwF0[0] = 0.1 # Static
        UwF0[1] = 1.0 # Cosine Fundamental
        UwF0[2] = -0.5 # Sine Fundamental
        
        UwF0[5] = 0.5 # Cosine 3rd
        UwF0[6] = -0.3 # Sine 3rd
        
        UwF0[-2] = 1.4
        UwF0[-1] = 2.0
        
        Fl = np.zeros_like(UwF0[:-2])
        Fl[1] = 1
        
        # Check of dRdUw
        fun = lambda Uw : self.vib_sys_jenkins.vprnm_res(
                                    np.hstack((Uw, UwF0[-1])), h, rhi, Fl)[0:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[:-1], verbose=False, atol=2e-9, rtol=5e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. Uw')
        
        
        # Check of dRdF
        fun = lambda F : self.vib_sys_jenkins.vprnm_res(
                                    np.hstack((UwF0[:-1], F)), h, rhi, Fl)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[-1:], verbose=False, atol=1e-10, rtol=1e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. force magnitude')
        
    def test_gradient_cubic_damp(self):
        """
        Numerical Gradient Check with higher harmonic phase calculated by AFT
        for cubic damping
        """
        
        
        # Inputs for PRNM
        h = np.array(range(6))
        rhi = 3
        UwF0 = np.zeros(hutils.Nhc(h)+2)
        
        UwF0[0] = 0.1 # Static
        UwF0[1] = 1.0 # Cosine Fundamental
        UwF0[2] = -0.5 # Sine Fundamental
        
        UwF0[5] = 0.5 # Cosine 3rd
        UwF0[6] = -0.3 # Sine 3rd
        
        UwF0[-2] = 0.33
        UwF0[-1] = 2.0
        
        Fl = np.zeros_like(UwF0[:-2])
        Fl[1] = 1
        
        # Check of dRdUw
        fun = lambda Uw : self.vib_sys_cubic_damp.vprnm_res(
                                    np.hstack((Uw, UwF0[-1])), h, rhi, Fl)[0:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[:-1], verbose=False, atol=2e-9, rtol=5e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. Uw')
        
        
        # Check of dRdF
        fun = lambda F : self.vib_sys_cubic_damp.vprnm_res(
                                    np.hstack((UwF0[:-1], F)), h, rhi, Fl)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[-1:], verbose=False, atol=1e-10, rtol=1e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. force magnitude')
        
        
    def test_known_solution(self):
        """
        Verify residual vector is near zero for known solutions to system of 
        equations
        1 each for jenkins and duffing
        """
        
        h = np.array(range(4))
        rhi = 3
        
        
        # Jenkins solution at low/mid amplitude
        UwF0 = np.array([0.        , 1.28497308, 0.2842021 , 0.        , 0.,
                         0.50030017, 0.22542694, 0.3115958, 1.0])
        
        Fl = np.zeros_like(UwF0[:-2])
        Fl[1] = 1
        
        R = self.vib_sys_jenkins.vprnm_res(UwF0, h, rhi, Fl)[0]
        
        self.assertLess(np.linalg.norm(R[:-1]), 6e-5, 'Previous solution fails residual.')
        
        # Jenkins solution at high amplitude
        UwF0 = np.array([0.        , 150.01064345,   1.0314443 ,   0.        ,
                 0.        ,  -9.20507502,  -2.65110885,   0.28892166, 100.0])
        
        R = self.vib_sys_jenkins.vprnm_res(UwF0, h, rhi, Fl)[0]
        
        self.assertLess(np.linalg.norm(R[:-1]), 4e-4, 'Previous solution fails residual.')
         
 
    def test_duffing_5to1(self):
        """
        Numerical Gradient Check with higher harmonic phase calculated by AFT
        for duffing
        """
        
        # Inputs for PRNM
        h = np.array(range(10))
        rhi = 5
        UwF0 = np.zeros(hutils.Nhc(h)+2)
        
        UwF0[0] = 0.0 # Static
        UwF0[1] = 1.0 # Cosine Fundamental
        UwF0[2] = 0.0 # Sine Fundamental
        
        UwF0[5] = -0.1 # Cosine 3rd
        UwF0[6] = 0.0 # Sine 3rd
        
        # Should give 1.0 for phase constraint since parallel to desired direction
        UwF0[9] = 1.0 # Cosine 5th 
        UwF0[10] = 0.0 # Sine 5th
        
        UwF0[-2] = 1.4
        UwF0[-1] = 2.0
        
        Fl = np.zeros_like(UwF0[:-2])
        Fl[1] = 1
        
        # Check of dRdUw
        fun = lambda Uw : self.vib_sys_duffing.vprnm_res(
                                    np.hstack((Uw, UwF0[-1])), h, rhi, Fl)[0:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[:-1], verbose=False, rtol=5e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. Uw')
        
        R = fun(UwF0[:-1])[0]
        
        self.assertLess(np.abs(R[-1] - 1.0), 1e-12)
        
        
        # Check of dRdF
        fun = lambda F : self.vib_sys_duffing.vprnm_res(
                                    np.hstack((UwF0[:-1], F)), h, rhi, Fl)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[-1:], verbose=False, rtol=1e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. force magnitude')
        
        
    def test_gradient_duffing_2dof(self):
        """
        Numerical Gradient Check with higher harmonic phase calculated by AFT
        for duffing
        """
        
        Ndof = 2
        
        # Inputs for PRNM
        h = np.array(range(6))
        rhi = 3
        UwF0 = np.zeros(Ndof*hutils.Nhc(h)+2)
        
        UwF0[0] = 0.1 # Static
        UwF0[1] = 0.3 # Static
        
        UwF0[Ndof + 0] = 0.5 # Cosine Fundamental
        UwF0[Ndof + 1] = 0.7 # Cosine Fundamental
        
        UwF0[2*Ndof + 0] = -0.3 # Sine Fundamental
        UwF0[2*Ndof + 1] = -0.1 # Sine Fundamental
        
        UwF0[5*Ndof + 0] = 0.6 # Cosine 3rd
        UwF0[5*Ndof + 1] = 6.0 # Cosine 3rd
        
        UwF0[6*Ndof + 0] = -0.5 # Sine 3rd
        UwF0[6*Ndof + 1] = -0.3 # Sine 3rd
        
        UwF0[-2] = 1.4 # Frequency
        UwF0[-1] = 2.0 # Force Level
        
        Fl = np.zeros_like(UwF0[:-2])
        Fl[Ndof + 0] = 1
        Fl[Ndof + 1] = 0.25
        
        # Check of dRdUw
        fun = lambda Uw : self.vib_sys_duffing_2dof.vprnm_res(
                                    np.hstack((Uw, UwF0[-1])), h, rhi, Fl)[0:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[:-1], verbose=False, rtol=5e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. Uw')
        
        
        # Check of dRdF
        fun = lambda F : self.vib_sys_duffing_2dof.vprnm_res(
                                    np.hstack((UwF0[:-1], F)), h, rhi, Fl)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[-1:], verbose=False, rtol=1e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. force magnitude')
        
        
    def test_gradient_damping_3dof(self):
        """
        Numerical Gradient Check with higher harmonic phase calculated by AFT
        for duffing
        """
        
        Ndof = 3
        
        # Inputs for PRNM
        h = np.array(range(6))
        rhi = 3
        UwF0 = np.zeros(Ndof*hutils.Nhc(h)+2)
        
        UwF0[0] = 0.1 # Static
        UwF0[1] = 0.3 # Static
        UwF0[2] = 0.3 # Static
        
        UwF0[Ndof + 0] = 0.5 # Cosine Fundamental
        UwF0[Ndof + 1] = 0.7 # Cosine Fundamental
        UwF0[Ndof + 2] = 0.5 # Cosine Fundamental
        
        UwF0[2*Ndof + 0] = -0.3 # Sine Fundamental
        UwF0[2*Ndof + 1] = -0.1 # Sine Fundamental
        UwF0[2*Ndof + 3] = -0.3 # Sine Fundamental
        
        UwF0[5*Ndof + 0] = 0.6 # Cosine 3rd
        UwF0[5*Ndof + 1] = 6.0 # Cosine 3rd
        UwF0[5*Ndof + 2] = -.5 # Cosine 3rd
        
        UwF0[6*Ndof + 0] = -0.5 # Sine 3rd
        UwF0[6*Ndof + 1] = -0.3 # Sine 3rd
        UwF0[6*Ndof + 2] = 1.2 # Sine 3rd
        
        UwF0[-2] = 1.4 # Frequency
        UwF0[-1] = 2.0 # Force Level
        
        Fl = np.zeros_like(UwF0[:-2])
        Fl[Ndof + 0] = 1
        Fl[Ndof + 1] = 0.25
        Fl[Ndof + 1] = 5.0
        
        # Check of dRdUw
        fun = lambda Uw : self.vib_sys_cubic_damp_3dof.vprnm_res(
                                    np.hstack((Uw, UwF0[-1])), h, rhi, Fl)[0:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[:-1], verbose=False, rtol=5e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. Uw')
        
        
        # Check of dRdF
        fun = lambda F : self.vib_sys_cubic_damp_3dof.vprnm_res(
                                    np.hstack((UwF0[:-1], F)), h, rhi, Fl)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[-1:], verbose=False, rtol=1e-1)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. force magnitude')
        
    def test_static_force(self):
        """
        copies 'test_gradient_damping_3dof', but adds a non-zero
        static force that should not be scaled by Fl

        Returns
        -------
        None.

        """
        
        Ndof = 3
        
        # Inputs for PRNM
        h = np.array(range(6))
        rhi = 3
        UwF0 = np.zeros(Ndof*hutils.Nhc(h)+2)
        
        UwF0[0] = 0.1 # Static
        UwF0[1] = 0.3 # Static
        UwF0[2] = 0.3 # Static
        
        UwF0[Ndof + 0] = 0.5 # Cosine Fundamental
        UwF0[Ndof + 1] = 0.7 # Cosine Fundamental
        UwF0[Ndof + 2] = 0.5 # Cosine Fundamental
        
        UwF0[2*Ndof + 0] = -0.3 # Sine Fundamental
        UwF0[2*Ndof + 1] = -0.1 # Sine Fundamental
        UwF0[2*Ndof + 3] = -0.3 # Sine Fundamental
        
        UwF0[5*Ndof + 0] = 0.6 # Cosine 3rd
        UwF0[5*Ndof + 1] = 6.0 # Cosine 3rd
        UwF0[5*Ndof + 2] = -.5 # Cosine 3rd
        
        UwF0[6*Ndof + 0] = -0.5 # Sine 3rd
        UwF0[6*Ndof + 1] = -0.3 # Sine 3rd
        UwF0[6*Ndof + 2] = 1.2 # Sine 3rd
        
        UwF0[-2] = 1.4 # Frequency
        UwF0[-1] = 2.0 # Force Level
        
        Fl = np.zeros_like(UwF0[:-2])
        Fl[:Ndof] = 0.352
        Fl[Ndof + 0] = 1
        Fl[Ndof + 1] = 0.25
        Fl[Ndof + 1] = 5.0
        
        Fl_half = np.copy(Fl)
        Fl_half[Ndof:] *= 0.5
        
        ########## Correctness of scaling force by Fl_half by 2 gives same R
        # as when using just Fl

        R_ref = self.vib_sys_cubic_damp_3dof.vprnm_res(
                            np.hstack((UwF0[:-1], UwF0[-1])), h, rhi, Fl)[0]
        
        
        R_half = self.vib_sys_cubic_damp_3dof.vprnm_res(
                    np.hstack((UwF0[:-1], 2.0*UwF0[-1])), h, rhi, Fl_half)[0]
        
        self.assertLess(np.linalg.norm(R_ref - R_half), 1e-12, 
                        'Scaling likely changes static force.')
        
        ########## Gradients with Static Force
        # Check of dRdUw
        fun = lambda Uw : self.vib_sys_cubic_damp_3dof.vprnm_res(
                                    np.hstack((Uw, UwF0[-1])), h, rhi, Fl)[0:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[:-1], verbose=False, rtol=5e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. Uw')
        
        
        # Check of dRdF
        fun = lambda F : self.vib_sys_cubic_damp_3dof.vprnm_res(
                                    np.hstack((UwF0[:-1], F)), h, rhi, Fl)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[-1:], verbose=False, rtol=1e-1)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. force magnitude')
                
    def test_skipped_harmonics(self):
        """
        Verify that VPRNM still works when some harmonics are skipped in h
        Should not throw an indexing error (this was an original bug). 
        
        Test is similar to 5:1 duffing, but uses Jenkins so that the third
        harmonic can be dropped and still excite the fifth harmonic

        Returns
        -------
        None.

        """
        
        # Inputs for PRNM
        h = np.array([0, 1, 5])
        rhi = 5
        UwF0 = np.zeros(hutils.Nhc(h)+2)
        
        UwF0[0] = 0.0 # Static
        UwF0[1] = 1.0 # Cosine Fundamental
        UwF0[2] = 0.0 # Sine Fundamental
        
        # Should give 1.0 for phase constraint since parallel to desired direction
        UwF0[3] = 1.0 # Cosine 5th 
        UwF0[4] = 0.0 # Sine 5th
        
        UwF0[-2] = 1.4
        UwF0[-1] = 2.0
        
        Fl = np.zeros_like(UwF0[:-2])
        Fl[1] = 1
        
        ####################
        # Accuracy Check - Reference solution with all harmonics
        
        h_ref = np.array(range(10))
        UwF0_ref = np.zeros(hutils.Nhc(h_ref)+2)
        
        UwF0_ref[0] = 0.0 # Static
        UwF0_ref[1] = 1.0 # Cosine Fundamental
        UwF0_ref[2] = 0.0 # Sine Fundamental
        
        # Should give 1.0 for phase constraint since parallel to desired direction
        UwF0_ref[9] = 1.0 # Cosine 5th 
        UwF0_ref[10] = 0.0 # Sine 5th
        
        UwF0_ref[-2] = 1.4
        UwF0_ref[-1] = 2.0
        
        Fl_ref = np.zeros_like(UwF0_ref[:-2])
        Fl_ref[1] = 1
        
        R_ref = self.vib_sys_jenkins.vprnm_res(UwF0_ref, h_ref, rhi, Fl_ref)[0]
        
        R_test = self.vib_sys_jenkins.vprnm_res(UwF0, h, rhi, Fl)[0]
        
        # Zeroth + 1st harmonic residual
        self.assertLess(np.linalg.norm(R_ref[:3] - R_test[:3]), 1e-12)
        
        # Fifth harmonic Residual
        self.assertLess(np.linalg.norm(R_ref[9:11] - R_test[3:5]), 1e-12)
        
        # VPRNM Constraint residual
        self.assertLess(np.abs(R_ref[-1] - R_test[-1]), 1e-12)
        
        ####################
        # Check Residual, then do Grad Checks
        
        # Check of dRdUw
        fun = lambda Uw : self.vib_sys_jenkins.vprnm_res(
                                    np.hstack((Uw, UwF0[-1])), h, rhi, Fl)[0:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[:-1], verbose=False, rtol=5e-9)
        
        self.assertFalse(grad_failed, 
                         'Incorrect Gradient w.r.t. Uw for skipped harmonics')
        
        # Check of dRdF
        fun = lambda F : self.vib_sys_jenkins.vprnm_res(
                                    np.hstack((UwF0[:-1], F)), h, rhi, Fl)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, UwF0[-1:], verbose=False, rtol=1e-11)
        
        self.assertFalse(grad_failed, 
             'Incorrect Gradient w.r.t. force magnitude for skipped harmonics')
        
                
if __name__ == '__main__':
    unittest.main()
