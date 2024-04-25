"""
Test harmonic post processing utilities with JAX forces cases
"""

import sys
import numpy as np
import unittest

sys.path.append('../..')
import tmdsimpy.postprocess.harmonic as hpost


from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.jax.nlforces.roughcontact.rough_contact import RoughContactFriction


class TestHarmonicPost(unittest.TestCase):
        
    def test_local_harmonic_forces(self):
        """
        Verifies the local nonlinear force post processing against a 
        few nonlinear forces.
        """
        
        ################
        # Setup a nonlinear vibration system
        M = np.eye(3)
        K = np.eye(3)
        
        Ndof = M.shape[0]
        
        Q = np.eye(3)
        T = 3.0*np.eye(3)
        
        ElasticMod = 1.92e+11
        PoissonRatio = 0.3
        Radius = 1.4e-3
        TangentMod = 6.2e8
        YieldStress = 330e6
        mu = 0.4
                
        rc_force = RoughContactFriction(Q, T, ElasticMod, PoissonRatio, 
                                        Radius, TangentMod, YieldStress, mu, 
                                        gaps=np.array([0.0]), 
                                        gap_weights=np.array([1.0]))
        
        vib_sys = VibrationSystem(M, K)
        vib_sys.add_nl_force(rc_force)
        
        ################
        # Alternative vibration system for checking with AFT
        T_eye = np.eye(3)
        
        rc_force_eye = RoughContactFriction(Q, T_eye, ElasticMod, PoissonRatio, 
                                        Radius, TangentMod, YieldStress, mu, 
                                        gaps=np.array([0.0]), 
                                        gap_weights=np.array([1.0]))
        
        
        ################
        # Setup a solution point to check
        Nt = 128
        
        h = np.array([0, 1, 3])
        
        U = np.array([1.0, 0.2, 1.0, # Harmonic 0
                      1e-6, -1e-3, 0.0, 0.0, 0.0, 0.0, # Harmonic 1
                      0.0,  0.0, 0.0, 0.0, 0.0, -0.1]) # Harmonic 3
        
        # Can still access the one in the vibration system since it is 
        # shared with this pointer
        rc_force.set_aft_initialize(U[:Ndof])
        rc_force_eye.set_aft_initialize(U[:Ndof])
        
        w = 1.9
        
        ################
        # Generate results
        
        Uh_ut_Fh_ft = hpost.local_harmonic_forces(vib_sys, U, w, h, Nt=Nt)
        
        ################
        # Check Results
        
        # Harmonic displacements
        Uh = U.reshape((-1, 3))
        
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][0] - Uh), 1e-12,
                        'Harmonic displacements are not as expected.')

        # Time series displacements
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][1][0] \
                                   - np.array([1.000001, 0.199, 1])), 1e-12,
                        'Time series is wrong.')

        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][1][32][:2] - U[:2]), 
                        1e-12, 'Time series is wrong.')
        
        # Harmonic force
        rc_aft = rc_force_eye.aft(U, w, h, Nt=Nt)[0]
        rc_aft = rc_aft.reshape((-1, 3), order= 'C')
        
        # Tangent forces are smaller, so tighter tolerance
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][2][:, :-1] - rc_aft[:, :-1]),
                       1e-9, 'Harmonic forces are not correct.')
        
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][2][:, -1] - rc_aft[:, -1]),
                       1e-7, 'Harmonic forces are not correct.')
        
        # Since the harmonic forces are correct, the time seres should be good
        # since the harmonic forces are based on the time series.
    
        ################
        # Check that AFT initialization is consistent between both approaches
        U[Ndof:] = 0.0
        
        Uh_ut_Fh_ft = hpost.local_harmonic_forces(vib_sys, U, w, h, Nt=Nt)
        
        rc_aft = rc_force_eye.aft(U, w, h, Nt=Nt)[0]
        rc_aft = rc_aft.reshape((-1, 3), order= 'C')
        
        # Tangent forces are smaller, so tighter tolerance
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][2][:, :-1] - rc_aft[:, :-1]),
                       1e-9, 'Harmonic forces are not correct.')
        
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][2][:, -1] - rc_aft[:, -1]),
                       1e-7, 'Harmonic forces are not correct.')
        
        # Should be zero in tangent direction since init is same as the zeroth harmonic
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][2][:, :-1]),
                       1e-9, 'Harmonic forces are not correct.')
        
        ################
        # Check that AFT initialization is consistent between both approaches
        # Now change init to be zeros
        
        # Can still access the one in the vibration system since it is 
        # shared with this pointer
        rc_force.set_aft_initialize(0.0*U[:Ndof])
        rc_force_eye.set_aft_initialize(0.0*U[:Ndof])
        
        U[Ndof:] = 0.0
        
        Uh_ut_Fh_ft = hpost.local_harmonic_forces(vib_sys, U, w, h, Nt=Nt)
        
        rc_aft = rc_force_eye.aft(U, w, h, Nt=Nt)[0]
        rc_aft = rc_aft.reshape((-1, 3), order= 'C')
        
        # Tangent forces are smaller, so tighter tolerance
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][2][:, :-1] - rc_aft[:, :-1]),
                       1e-9, 'Harmonic forces are not correct.')
        
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][2][:, -1] - rc_aft[:, -1]),
                       1e-7, 'Harmonic forces are not correct.')
        
        # Should be nonzero
        self.assertGreater(np.linalg.norm(Uh_ut_Fh_ft[0][2][:, :-1]),
                       10, 'Harmonic forces are not correct.')
        
        
if __name__ == '__main__':
    unittest.main()