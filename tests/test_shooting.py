"""
Test for verifying the correctness of Shooting Method

""" 

import sys
import numpy as np
from scipy import io as sio
import unittest

import verification_utils as vutils

sys.path.append('..')
from tmdsimpy.nlforces.cubic_stiffness import CubicForce

from tmdsimpy.vibration_system import VibrationSystem


class TestShooting(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        
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
        
        super(TestShooting, self).__init__(*args, **kwargs)       
        
        # Tolerances
        self.grad_rtol = 5e-10
        self.nearlin_tol = 1e-12 # Tolerance for linear analytical v. HBM check
        self.dRdw_rtol = 2e-8
        
        #######################################################################
        ###### Setup Nonlinear System                                    ######
        #######################################################################
        
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
        
        ###########################
        # Setup Vibration System
        
        M = np.array([[6.12, 3.33, 4.14],
                      [3.33, 4.69, 3.42],
                      [4.14, 3.42, 3.7 ]])
        
        K = np.array([[2.14, 0.77, 1.8 ],
                      [0.77, 2.15, 1.71],
                      [1.8 , 1.71, 2.12]])
        
        C = 0.01 * M + 0.02*K
        
        vib_sys = VibrationSystem(M, K, C)
        
        vib_sys.add_nl_force(duff_force)
        
        ###########################
        # Store vibration system
        
        self.vib_sys = vib_sys


    def test_gradients(self):
        """
        Check gradients against numerical differentiation

        Returns
        -------
        None.

        """
        
        vib_sys = self.vib_sys
        
        Nt = 1<<7
        
        
        ###########################
        # Displacement Gradient
        w = 1.0
        
        Fl = np.zeros(2*vib_sys.M.shape[0])
        Fl[1] = 1.0
        
        X0 = np.array([1.2, 1.9, 2.5, 0, 0, 0])
        
        fun = lambda X0 : vib_sys.shooting_res(np.hstack((X0, [w])), Fl, Nt=Nt)[0:2]
        
        grad_failed = vutils.check_grad(fun, X0, rtol=self.grad_rtol, 
                                        verbose=False)
        
        
        self.assertFalse(grad_failed, 'Incorrect gradient for displacements.')   
        
        
        ###########################
        # Frequency Gradient
        # w = 0.7913276228169658
        # X0 = np.array([-0.55106882,  0.07415273,  0.91538327, 0, 0, 0])
        # vib_sys.C = 0*vib_sys.C
        
                
        fun = lambda w : vib_sys.shooting_res(np.hstack((X0, w)), Fl, Nt=Nt)[0::2]
        
        grad_failed = vutils.check_grad(fun, np.array([w]), rtol=self.dRdw_rtol, 
                                        verbose=False)
        
        
        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')   
        
        
        
if __name__ == '__main__':
    unittest.main()


