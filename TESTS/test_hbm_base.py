"""
Unit testing of base excitation harmonic balance implementation

Steps:
    1. Create System
    2. Verify Gradients
    3. Static solutions
    4. Comparison to a linear FRF
    
This test does not consider nonlinear forces since the base excitation residual
wraps the normal harmonic balance method and thus the nonlinear force evals are
already tested
"""

import sys
import numpy as np
import unittest


# Path to Harmonic balance / vibration system 
sys.path.append('../ROUTINES/')
sys.path.append('../ROUTINES/NL_FORCES')

from vibration_system import VibrationSystem
from solvers import NonlinearSolver
from continuation import Continuation

import harmonic_utils as hutils
import verification_utils as vutils


class TestHBMBase(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Additional initialization of a system so it does not have to be 
        recreated in each function.
        System:
            
        /|                  k=3
        /|________________/\/\/\/\_________________|
        /|                                         |
        /|___/\/\/\/\___| m1=1 | ___/\/\/\/\___| m2=0.5 |___/\/\/\/\___| m3=2 |
        /|     k=1                    k=2                        k=1
        /|

        Returns
        -------
        None.

        """
        super(TestHBMBase, self).__init__(*args, **kwargs)        
        
        K = 1.0*np.array([[ 4, -1, -3,  0], \
                          [-1,  3, -2,  0], \
                          [-3, -2,  6, -1], \
                          [ 0,  0,  -1, 1]])
        
        M = np.diag(np.array([0.0, 1.0, 0.5, 2.0]))
        
        ab_damp = [0.0001, 0.0003]
        
        
        vib_sys = VibrationSystem(M, K, ab=ab_damp)
        
        self.vib_sys = vib_sys
        self.base_flag = np.array([True, False, False, False])

    def test_gradients(self):
        """
        Numerical Derivative verification
        """
        
        # Sanity check that vibration system has been saved for the test 
        # correctly
        self.assertEqual(self.vib_sys.M[1,1], 1.0, \
                     'Stored vibration system should have this mass entry.')
            
        h = np.array(range(5))
        Nhc = hutils.Nhc(h)
        
        Nbase = self.base_flag.sum()
        
        # Randomly generated vector of Uw
        Uw = np.array([0.34463874,  0.47030463, -0.0239792 , -0.02151482,  0.79866951,
                       -0.95386587, -0.00755133,  0.91014791,  0.96469309, -0.59122385,
                        0.57267981,  0.02136095,  0.21514046,  0.65978107,  0.29403553,
                        0.80076917, -0.93450551, -0.6053289 , -0.80127006, -0.65768091,
                       -0.16408756, -0.71138074, -0.29354861,  0.61332329, -0.94332345,
                        0.12850915,  0.54556297, 1.25])

        Ubase = np.zeros(Nbase*Nhc)
        Ubase[0] = 1.0 # Fixed first harmonic base motion
        
        # Displacement Derivative Checks
        fun = lambda U : self.vib_sys.hbm_base_res(np.hstack((U, Uw[-1])), \
                                                   Ubase, self.base_flag, h)[0:2]
            
        grad_failed = vutils.check_grad(fun, Uw[:-1], verbose=False, rtol=1e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. U')
        
        # Frequency Gradient Check
        fun = lambda w : self.vib_sys.hbm_base_res(np.hstack((Uw[:-1], w)), \
                                               Ubase, self.base_flag, h)[0:3:2]

        grad_failed = vutils.check_grad(fun, Uw[-1:], verbose=False, rtol=1e-11)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. frequency')

    def test_static_deformation(self):
        """
        Static Deformation Checks of structure - full structure should just 
        translate
        """
        
        h = np.array([0])
        Nhc = hutils.Nhc(h)
        
        Ndof = np.logical_not(self.base_flag).sum()
        Nbase = self.base_flag.sum()
        
        Uw = np.ones(Ndof*Nhc+1)
        Ubase = np.ones(Nbase*Nhc)

        R, dRdU, dRdw = self.vib_sys.hbm_base_res(Uw, Ubase, self.base_flag, h)

        self.assertLess(np.abs(R).sum(), 1e-16, 'Static base translation failed.')
        
        self.assertLess(np.abs(dRdU \
                       - self.vib_sys.K[:, np.logical_not(self.base_flag)]\
                           [np.logical_not(self.base_flag), :]).sum(), \
                        1e-16, 'Static base translation failed.')
        
        self.assertLess(np.abs(dRdw).sum(), 1e-16, 'Bad static gradient.')

    def test_frf(self):
        """
        Compares a continuation for base excitation against an analytical 
        solution (linear problem)

        Returns
        -------
        None.

        """
        
        #####################################
        # Setup
        
        solver = NonlinearSolver
        
        h = np.array(range(3+1))
        Nhc = hutils.Nhc(h)
        
        Ndof = np.logical_not(self.base_flag).sum()
        Nbase = self.base_flag.sum()
        
        
        ####################################
        # Continuation Solution
        
        # Uw = np.ones(Ndof*Nhc+1)
        Ubase = np.zeros(Nbase*Nhc)
        Ubase[1] = 1.0 # Cos, fundamental
        Ubase[2] = 0.75*0 # Sin, fundamental
        
        
        # System frequencies are : [0.61900995, 1.49237621, 3.59021447]
        lam0 = 0.01
        lam1 = 5.0
        
        # Uw0_1st = self.vib_sys.linear_frf_base(np.array(lam0), Ubase[1:3], \
        #                                      self.base_flag, solver, neigs=3)
        
        Uw0 = np.zeros(Nhc*Ndof+1)
        # Uw0[Ndof:3*Ndof] = Uw0_1st[:, :2*Ndof]
        Uw0[Ndof:2*Ndof]   = Ubase[1]
        Uw0[2*Ndof:3*Ndof] = Ubase[2]
        Uw0[-1] = lam0
        
                
        continue_config = {'DynamicCtoP': True, 
                           'TargetNfev' : 200,
                           'MaxSteps'   : 2000,
                           'dsmin'      : 0.005,
                           'verbose'    : False,
                           'xtol'       : 5e-8*Uw0.shape[0], 
                           'corrector'  : 'Ortho'}
        
        CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*Ubase[1])
        CtoP[-1] = lam1 # so steps are approximately ds/sqrt(2) of lam1
        
        cont_solver = Continuation(solver, ds0=0.05, CtoP=CtoP, config=continue_config)
        
        fun = lambda Uw : self.vib_sys.hbm_base_res(Uw, Ubase, self.base_flag, h)
        
        XlamP_full = cont_solver.continuation(fun, Uw0, lam0, lam1)

        
        ####################################
        # Analytical Solution
        
        w = XlamP_full[:, -1]
        
        Ub_analytical = Ubase[1:3] # cos, sin
        Xw_analytical = self.vib_sys.linear_frf_base(w, Ub_analytical, \
                                             self.base_flag, solver, neigs=3)
        
        error = XlamP_full[:, Ndof:3*Ndof] - Xw_analytical[:, :-1]
        
        self.assertLess(np.abs(error).max(), 1e-4, 'Peak errors exceed limit.')
        
        away_from_res_mask = np.min(np.abs(w \
                               - np.array([[0.619, 1.492, 3.590]]).T), axis=0) > 0.01
        
        self.assertLess(np.abs(error[away_from_res_mask, :]).max(), 3e-6, 'FRF errors exceed limit.')
        
        
        self.assertLess(XlamP_full[:,:Ndof].max(), 1e-16, \
                        'Should have no 0th harmonic.')
        
        self.assertLess(XlamP_full[:,3*Ndof:-1].max(), 1e-16, \
                        'Should have no higher harmonics.')
        
        
if __name__ == '__main__':
    unittest.main()