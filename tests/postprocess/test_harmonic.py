"""
Test harmonic post processing utilities
"""

import sys
import numpy as np
import unittest

sys.path.append('../..')
import tmdsimpy.postprocess.harmonic as hpost


from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.nlforces.cubic_stiffness import CubicForce
from tmdsimpy.nlforces.vector_jenkins import VectorJenkins


class TestHarmonicPost(unittest.TestCase):
        
    def test_local_harmonic_forces(self):
        """
        Verifies the local nonlinear force post processing against a 
        few nonlinear forces.
        """
        
        ################
        # Setup a nonlinear vibration system
        M = np.eye(2)
        K = np.eye(2)
        
        Q_duff = np.array([[1.0, -1.0], [0.0, 1.0]])
        T_duff = 3.0*np.eye(2)
        
        kalpha = 0.7 # Duffing
        
        Q_jenk = np.array([[1.0, -1.0]])
        T_jenk = np.array([[4.0], [-3.0]])
        kt = 0.8 # Jenkins
        Fs = 1.7 # Jenkins
        
        duff_force = CubicForce(Q_duff, T_duff, kalpha)
        jenk_force = VectorJenkins(Q_jenk, T_jenk, kt, Fs)
        
        vib_sys = VibrationSystem(M, K)
        vib_sys.add_nl_force(duff_force)
        vib_sys.add_nl_force(jenk_force)
        
        ################
        # Alternative vibration system for checking with AFT
        T_eye_duff = np.eye(2)
        T_eye_jenk = np.array([[1.0], [0.0]])
        
        duff_force_eye = CubicForce(Q_duff, T_eye_duff, kalpha)
        jenk_force_eye = VectorJenkins(Q_jenk, T_eye_jenk, kt, Fs)
        
        # vib_sys_eye = VibrationSystem(M, K)
        # vib_sys_eye.add_nl_force(duff_force_eye)
        # vib_sys_eye.add_nl_force(jenk_force_eye)
        
        ################
        # Setup a solution point to check
        Nt = 128
        
        h = np.array([0, 1, 3])
        
        U = np.array([0.0, 0.0, # Harmonic 0
                      5.0, -3.0, 0.0, 0.0, # Harmonic 1
                      0.0,  0.0, 0.0, 0.0]) # Harmonic 3
        
        w = 1.9
        
        ################
        # Generate results
        
        Uh_ut_Fh_ft = hpost.local_harmonic_forces(vib_sys, U, w, h, Nt=Nt)
        
        ################
        # Check Results
        
        # Harmonic displacements
        Uh_duff = np.array([[ 0.,  0.],
                           [ 8., -3.],
                           [ 0.,  0.],
                           [ 0.,  0.],
                           [ 0.,  0.]])
        
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][0] - Uh_duff), 1e-12,
                        'Harmonic displacements are not as expected.')
        
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[1][0] - Uh_duff[:, 0:1]), 
                        1e-12, 'Harmonic displacements are not as expected.')

        # Time series displacements
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][1][0] \
                                       - np.array([8, -3])), 1e-12,
                        'Time series is wrong.')

        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][1][32]), 1e-12,
                        'Time series is wrong.')
        
        # Harmonic force
        duff_aft = duff_force_eye.aft(U, w, h, Nt=Nt)[0]
        duff_aft = duff_aft.reshape((-1, 2), order= 'C')
        
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[0][2] - duff_aft),
                       1e-12, 'Harmonic duffing forces are not correct.')
        
        jenk_aft = jenk_force_eye.aft(U, w, h, Nt=Nt)[0]
        
        self.assertLess(np.linalg.norm(Uh_ut_Fh_ft[1][2][:, 0] - jenk_aft[::2]),
                        1e-12, 'Harmonic Jenkins force are not correct.')
        
        # Since the harmonic forces are correct, the time seres should be good
        # since the harmonic forces are based on the time series.
    
if __name__ == '__main__':
    unittest.main()