"""
Tests for verifying the accuracy of the ROMs based on EPMC.
""" 

import sys
import numpy as np
import unittest

sys.path.append('../..')

from tmdsimpy.roms import epmc

from tmdsimpy import harmonic_utils as hutils
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
        Uw_linear = vib_sys.linear_frf(Omega, Fl[Ndof:2*Ndof], solver, 
                                       neigs=M.shape[0], 
                                       Flsin=Fl[2*Ndof:3*Ndof])
        
        # Only check points that are far away from the second resonance
        mask = Omega < wn[1] / 10
        
        error = np.abs(FRC_reconstruct[mask, Ndof:3*Ndof] 
                       - Uw_linear[mask, :2*Ndof]) \
                / Uw_linear[mask, :2*Ndof].max()
                
        error = error.max(axis=0)
        
        self.assertLess(error[0], 1e-3, 'High EPMC reconstruction errors.')
        
        self.assertLess(error[1:].max(), 1e-4, 
                        'High EPMC reconstruction errors.')

if __name__ == '__main__':
    unittest.main()