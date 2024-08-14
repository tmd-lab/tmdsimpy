"""
Tests for verifying the accuracy of the ROMs based on EPMC.
""" 

import sys
import numpy as np
import unittest

sys.path.append('../..')

from tmdsimpy.roms import epmc

import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver

class TestEpmcRoms(unittest.TestCase):
    
    def test_constant_force_frf(self):
        """
        Test EPMC at constant force for recreating linear FRF responses.
        """
        
        # Construct a linear system with prescribed mode shapes and frequencies
        Phi = np.array([[1, 2], [2, 1]])
        Phi_inv = np.linalg.inv(Phi)
        
        wn = np.array([2.0, 100.0])
        
        M = Phi_inv.T @ Phi_inv
        K = Phi_inv.T @ np.diag(wn)**2 @ Phi_inv
        Ndof = M.shape[0]
        
        ab_damp = [0.2, 0.0]
        
        vib_sys = VibrationSystem(M, K, ab=ab_damp)
        
        # EPMC Solution (analytically constructed) with some made up zeroth 
        # harmonic contribution that can be checked at the end
        h = np.array([0, 1])
        
        # Mode shape has a bit on sine and a bit on cosine
        U_point = np.array([0.1, 6.2, 1, 2, 0.1, 0.2, wn[0], ab_damp[0], 0.0])
        
        norm = np.sqrt(U_point[Ndof:2*Ndof] @ M @ U_point[Ndof:2*Ndof] \
                       + U_point[2*Ndof:3*Ndof] @ M @ U_point[2*Ndof:3*Ndof])
        
        U_point[Ndof:3*Ndof] = (1/norm) * U_point[Ndof:3*Ndof]
        
        epmc_bb = np.ones((50, 1)) @ np.atleast_2d(U_point)
        epmc_bb[:, -1] = np.linspace(-3, 3, epmc_bb.shape[0])
        
        #############################
        # Check that this constructed solution is done correctly and passes 
        # EPMC and therefore the test should be valid
        
        Nhc = hutils.Nhc(h)
        Fl_epmc = np.zeros(Nhc*Ndof)
        
        Fl_epmc[:Ndof] = K @ U_point[:Ndof]
        
        # ignore EPMC phase constraint on artificial solution
        Fl_epmc[2*Ndof:3*Ndof] = 0*np.ones(Ndof) 
        
        R = vib_sys.epmc_res(epmc_bb[0], Fl_epmc, h)[0]
        
        self.assertLess(np.linalg.norm(R), 1e-12,
                    'Constructed solution does not pass EPMC residual check.')
        
        #############################
        # Test the ROM FRC reconstructions
        Fl = np.zeros(Nhc*Ndof)
        Fl[:Ndof] = K @ U_point[:Ndof]
        Fl[Ndof] = 3.0 # mostly cosine forcing
        Fl[2*Ndof] = 1.0 # a bit of sine forcing
        
        FRC_reconstruct, modal_amp, modal_phase = epmc.constant_force(epmc_bb, 
                                                                    Ndof, h, 
                                                                    Fl=Fl)
        
        self.assertLess(np.linalg.norm(FRC_reconstruct[:, :Ndof] \
                                       - U_point[:Ndof]), 1e-12,
                        'ROM Should take static displacements from EPMC.')
        
        Omega = FRC_reconstruct[:, -1]
        
        # Obtain linear solution
        solver = NonlinearSolver()
        
        # use 1 mode to more closely match the EPMC ROM
        Uw_linear = vib_sys.linear_frf(Omega, Fl[Ndof:2*Ndof], solver, 
                                       neigs=1, 
                                       Flsin=Fl[2*Ndof:3*Ndof])
        
        # Only check points that are far away from the second resonance
        mask = Omega < wn[1] / 10
        
        error = np.abs(FRC_reconstruct[mask, Ndof:3*Ndof] 
                       - Uw_linear[mask, :2*Ndof]) \
                / np.abs(Uw_linear[mask, :2*Ndof]).max()
        
        self.assertLess(error.max(), 1e-10, 
                        'High EPMC reconstruction errors.')

    def test_constant_displacement_frf(self):
        """
        Test EPMC at constant force for recreating linear FRF responses.
        """
        
        # Construct a linear system with prescribed mode shapes and frequencies
        Phi = np.array([[1, 2], [2, 1]])
        Phi_inv = np.linalg.inv(Phi)
        
        wn = np.array([2.0, 100.0])
        
        M = Phi_inv.T @ Phi_inv
        K = Phi_inv.T @ np.diag(wn)**2 @ Phi_inv
        Ndof = M.shape[0]
        
        ab_damp = [0.2, 0.0]
        
        vib_sys = VibrationSystem(M, K, ab=ab_damp)
        
        # EPMC Solution (analytically constructed) with some made up zeroth 
        # harmonic contribution that can be checked at the end
        h = np.array([0, 1])
        
        # Mode shape has a bit on sine and a bit on cosine
        U_point = np.array([0.1, 6.2, 1, 2, 0.1, 0.2, wn[0], ab_damp[0], 0.0])
        
        norm = np.sqrt(U_point[Ndof:2*Ndof] @ M @ U_point[Ndof:2*Ndof] \
                       + U_point[2*Ndof:3*Ndof] @ M @ U_point[2*Ndof:3*Ndof])
        
        U_point[Ndof:3*Ndof] = (1/norm) * U_point[Ndof:3*Ndof]
        
        epmc_bb = np.ones((50, 1)) @ np.atleast_2d(U_point)
        epmc_bb[:, -1] = np.linspace(-3, 3, epmc_bb.shape[0])
        
        #############################
        # Check that this constructed solution is done correctly and passes 
        # EPMC and therefore the test should be valid
        
        Nhc = hutils.Nhc(h)
        Fl_epmc = np.zeros(Nhc*Ndof)
        
        Fl_epmc[:Ndof] = K @ U_point[:Ndof]
        
        # ignore EPMC phase constraint on artificial solution
        Fl_epmc[2*Ndof:3*Ndof] = 0*np.ones(Ndof) 
        
        R = vib_sys.epmc_res(epmc_bb[0], Fl_epmc, h)[0]
        
        self.assertLess(np.linalg.norm(R), 1e-12,
                    'Constructed solution does not pass EPMC residual check.')
        
        #############################
        # Test the ROM FRC reconstructions
        Flcos = np.array([3.0, 1.0])
        control_point = np.array([1.0, 0.0])
        
        control_amplitude = 5.9
        Omega = np.linspace(0.1, 10, 100)
        
        force_magnitude, epmc_point = epmc.constant_displacement(epmc_bb, 
                                             h, Flcos, Omega, control_point, 
                                             control_amplitude)
        
        # Verify that the returned EPMC point satisfies amplitude constraint
        epmc_cos = epmc_point[Ndof:2*Ndof] @ control_point
        epmc_sin = epmc_point[2*Ndof:3*Ndof] @ control_point
        
        epmc_point_amp = np.sqrt(epmc_cos**2 + epmc_sin**2)*(10**epmc_point[-1])
        
        self.assertAlmostEqual(epmc_point_amp, control_amplitude, 10,
                           'Returned EPMC point is not at expected amplitude.')
        
        # Obtain linear solution
        solver = NonlinearSolver()
        
        # use 1 mode to more closely match the EPMC ROM
        Uw_linear = vib_sys.linear_frf(Omega, Flcos, solver, neigs=1)
        
        linear_cos = Uw_linear[:, :Ndof] @ control_point
        linear_sin = Uw_linear[:, Ndof:2*Ndof] @ control_point
        
        linear_force = control_amplitude/np.sqrt(linear_cos**2 + linear_sin**2)
        
        self.assertLess(np.abs(force_magnitude - linear_force).max(), 1e-11,
                'EPMC controlled amplitude gives wrong forces for linear case')
        
if __name__ == '__main__':
    unittest.main()