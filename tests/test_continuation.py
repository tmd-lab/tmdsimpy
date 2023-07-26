"""
Test for verifying the accuracy of the continuation methods

Outline:
    1. Setup a Nonlinear Model with Harmonic Balance (HBM)
    2. Solve HBM at a point
    3. Verify that the residual is appropriate at that point (e.g., 0 except arc length)
    4. Verify Gradients at the solution point
    5. Verify that the forward stepping solves the arc length residual exactly
    6. Try a full continuation (linear against FRF)
    
Notes:
    1. It would be better to have all the tolerances defined somewhere together
    rather than the current check of having them wherever they are used.
""" 


import sys
import numpy as np
import unittest

import verification_utils as vutils

sys.path.append('..')
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.continuation import Continuation
import tmdsimpy.harmonic_utils as hutils

from tmdsimpy.nlforces.cubic_stiffness import CubicForce


def continuation_test(fmag, Uw, Fl, h, solver, vib_sys, cont_config, test_obj):
    """
    Define one function than can be repeatedly called to verify the important
    aspects of the continuation routine for different test cases. 

    Parameters
    ----------
    fmag : Force Magnitude
    Uw : Displacements and Frequency vector
    Fl : Vector defining where the external force is applied for Harmonic 
         Balance
    h : List of Harmonics
    solver : Reference to the solver to use
    vib_sys : Vibration system description
    cont_config : Continuation configuration settings.
    test_obj : unittest object that is being used to raise errors for the test.

    Returns
    -------
    None

    """
        
    ###############################################################################
    ####### 2. Solve HBM at point                                          #######
    ###############################################################################
    
    fun = lambda U : vib_sys.hbm_res(np.hstack((U, Uw[-1])), \
                                     fmag*Fl, h, Nt=128, aft_tol=1e-7)[0:2]
    
    X, R, dRdX, sol = solver.nsolve(fun, fmag*Uw[:-1], verbose=False)
    
    R_fun, dRdX_fun = fun(X)
        
    if fmag < 1e-6:
        # Only check against linear resonance amplitude for light forcing 
        # (when it is near linear response)
        
        test_obj.assertLess(np.abs(X - fmag*Uw[:-1].reshape((-1))).max(), 
                            1e-10, 
                            'Low forcing amplitude does not match linear response.')
    
    ###############################################################################
    ####### 3. Verify Continuation Residual                                 #######
    ###############################################################################
    
    Uw0 = np.hstack((X, Uw[-1])) # Solution from previous HBM solve.
    
    Ndof = vib_sys.M.shape[0]
    CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*fmag)
        
    ds = 0.01
    
    # Generate Continuation Model
    cont_solver = Continuation(solver, ds0=ds, CtoP=CtoP, config=cont_config)
    
    fun = lambda Uw : vib_sys.hbm_res(Uw, fmag*Fl, h, Nt=128, aft_tol=1e-7)
    
    ########### Continuation Residual at Initial Point:
    XlamC = Uw0 / CtoP
    XlamC0 = Uw0 / CtoP
    
    # Require a predictor: 
    XlamPprev = np.copy(Uw0)
    XlamPprev[-1] = XlamPprev[-1] - 1
    dirC = cont_solver.predict(fun, Uw0, XlamPprev)
    
    Raug, dRaugdXlamC = cont_solver.correct_res(fun, XlamC, XlamC0, ds, dirC)
    
    
    test_obj.assertLess(np.max(np.abs(Raug[:-1])), 1e-5, 
                        'Failed non arc length residual check.')
    
    test_obj.assertLess(np.abs(np.abs(Raug[-1]) - 1), 1e-14, 
                        'Failed arc length residual check [abs(Arc Residual) - 1 (should be 0)].')
    
    ###############################################################################
    ####### 4. Gradient of Augmented Equations                              #######
    ###############################################################################
    
    fun_aug = lambda XlamC : cont_solver.correct_res(fun, XlamC, XlamC0, ds, dirC)
    
    grad_failed = vutils.check_grad(fun_aug, XlamC, atol=0.0, rtol=1e-6, verbose=False)
        
    test_obj.assertFalse(grad_failed, 'Incorrect gradient.')
    
    
    ###############################################################################
    ####### 5. Forward Stepping Satisfyies Length Residual                  #######
    ###############################################################################
    
    dXlamPprev = CtoP*XlamC
    dXlamPprev[-1] -= CtoP[-1]*ds # Set to increasing frequency
    
    dirC = cont_solver.predict(fun, CtoP*XlamC, dXlamPprev)
    
    test_obj.assertTrue(np.sign(dirC[-1]) == 1.0, 
                        'Incorrect Frequency step direction.')
    
    
    Raug, dRaugdXlamC = cont_solver.correct_res(fun, XlamC + ds*dirC, XlamC, ds, dirC)
    
    test_obj.assertLess(np.abs(Raug[-1]), 1e-14, 
                        'Arc Length Residual should be exactly satisifed by prediction.')
    
    
    return



class TestContinuation(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Additional initialization of a system so it does not have to be 
        recreated in each function.
        
        Returns
        -------
        None.

        """
        super(TestContinuation, self).__init__(*args, **kwargs)        
        
                
        #######################################################################
        ####### 1. Setup Nonlinear HBM Model                            #######
        #######################################################################
                
        ###########################
        # Setup Nonlinear Force
        
        # Simple Mapping to spring displacements
        Q = np.array([[-1.0, 1.0, 0.0]])
        
        # Weighted / integrated mapping back for testing purposes
        T = np.array([[-0.5], \
                      [0.5], \
                      [0.0] ])
        
        kalpha = np.array([3.2])
        
        duff_force = CubicForce(Q, T, kalpha)
        
        ###########################
        # Setup Vibration System
        
        M = np.array([[6.12, 3.33, 4.14],
                      [3.33, 4.69, 3.42],
                      [4.14, 3.42, 3.7 ]])
        
        K = np.array([[3.0, 0.77, 1.8 ],
                       [0.77, 2.48, 1.71],
                       [1.8 , 1.71, 2.51]])
        
        
        ab_damp = [0.0001, 0.0003]
        C = ab_damp[0]*M + ab_damp[1]*K
        
        vib_sys = VibrationSystem(M, K, C)
        
        # Verify Mass and Stiffness Matrices are Appropriate
        solver = NonlinearSolver()
        
        # lam,V = solver.eigs(M) # M must be positive definite.
        # lam,V = solver.eigs(K) # K should be at least positive semi-definite.
        lam,V = solver.eigs(K, M)
        
        vib_sys.add_nl_force(duff_force)
        
        
        ###########################
        # Solution Initial Guess
        lam,V = solver.eigs(vib_sys.K, vib_sys.M)
        
        
        h = np.array([0, 1, 2, 3, 4, 5]) 
        
        Nhc = hutils.Nhc(h)
        Ndof = M.shape[0]
        
        mi = 0 # mode of interest
        wn = np.sqrt(lam[mi])
        w = wn # Force at near resonance.
        vi = V[:, mi]
        
        Fl = np.zeros((Nhc*Ndof,))
        Fl[1*Ndof:2*Ndof] = (M @ vi) # First Harmonic Cosine, DOF 1
        
        Uw = np.zeros((Nhc*Ndof+1,))
        Uw[-1] = w 
        
        # Mode 2 proportional damping
        zeta = ab_damp[0]/w/2 +  ab_damp[1]*w/2
        
        qlinear = (vi @ Fl[1*Ndof:2*Ndof]) / np.sqrt( (wn**2 - w**2)**2 + (2*zeta*w*wn)**2)
        
        # 90 deg. phase lag near resonance.
        Uw[2*Ndof:3*Ndof] = qlinear * vi

    
        ###########################
        # Store Everything in self
        self.continuation_data = (Uw, Fl, h, solver, vib_sys)
        
        
        #######################################################################
        ####### Second Linear System Setup for Some Tests               #######
        #######################################################################

        fmag = 1.0 
        lam0 = 0.2
        lam1 = 3

        # Linear system
        vib_sys = VibrationSystem(M, K, ab=ab_damp)

        kalpha = np.array([0.0])
        duff_force = CubicForce(Q, T, kalpha)

        vib_sys.add_nl_force(duff_force)

        # Forcing
        Fl = np.zeros((Nhc*Ndof,))
        Fl[Ndof] = 1

        # Solution at initial point
        fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam0)), fmag*Fl, h)[0:2]

        U0stat = np.linalg.solve(vib_sys.K, Fl[Ndof:2*Ndof])
        U0 = np.zeros_like(Fl)
        U0[Ndof:2*Ndof] = U0stat

        X, R, dRdX, sol = solver.nsolve(fun, fmag*U0, verbose=False)

        Uw0 = np.hstack((U0, lam0))

        CtoP = hutils.harmonic_wise_conditioning(Uw0, Ndof, h, delta=1e-3*fmag)
        CtoP[-1] = lam1 # so steps are approximately ds/sqrt(2) of lam1


        fun = lambda Uw : vib_sys.hbm_res(Uw, fmag*Fl, h)

        
        ###########################
        # Store Everything in self
        self.linear_sys_data = (Uw0,solver,CtoP,fun,lam0,lam1,vib_sys,Fl)
        
    def test_low_amp_pseudo(self):
        """
        Testing a low amplitude FRC, which should be essentially linear with 
        pseudo arc length

        Returns
        -------
        None.

        """
        
        # Unpack data shared between all tests
        Uw, Fl, h, solver, vib_sys = self.continuation_data
        
        # Settings for Specific Test
        fmag = 0.0000001
        psuedo_config = {'corrector': 'Pseudo',
                         'verbose'  : -1}

        # Run test
        continuation_test(fmag, Uw, Fl, h, solver, vib_sys, psuedo_config, self)

    def test_nl_pseudo(self):
        """
        Testing a high amplitude FRC, for continuation with ortho arc length

        Returns
        -------
        None.

        """
        
        # Unpack data shared between all tests
        Uw, Fl, h, solver, vib_sys = self.continuation_data
        
        # Settings for Specific Test
        fmag = 1.0
        psuedo_config = {'corrector': 'Pseudo',
                         'verbose'  : -1}

        # Run test
        continuation_test(fmag, Uw, Fl, h, solver, vib_sys, psuedo_config, self)


    def test_low_amp_ortho(self):
        """
        Testing a low amplitude FRC, which should be essentially linear with 
        ortho arc length

        Returns
        -------
        None.

        """
        
        # Unpack data shared between all tests
        Uw, Fl, h, solver, vib_sys = self.continuation_data
        
        # Settings for Specific Test
        fmag = 0.0000001
        ortho_config = {'corrector': 'ortho',
                         'verbose'  : -1}

        # Run test
        continuation_test(fmag, Uw, Fl, h, solver, vib_sys, ortho_config, self)



    def test_nl_ortho(self):
        """
        Testing a high amplitude FRC, for continuation with pseudo arc length

        Returns
        -------
        None.

        """
        
        # Unpack data shared between all tests
        Uw, Fl, h, solver, vib_sys = self.continuation_data
        
        # Settings for Specific Test
        fmag = 1.0
        ortho_config = {'corrector': 'ortho',
                         'verbose'  : -1}

        # Run test
        continuation_test(fmag, Uw, Fl, h, solver, vib_sys, ortho_config, self)


    def test_full_continuation_pseudo(self):
        """
        Test a full arclength continuation with the pseudo corrector
        Uses a linear system and compares against linear FRF

        Returns
        -------
        None.

        """
        Uw0,solver,CtoP,fun,lam0,lam1,vib_sys,Fl = self.linear_sys_data
        Ndof = vib_sys.M.shape[0]
        
        continue_config = {'DynamicCtoP': True, 
                           'TargetNfev' : 200,
                           'MaxSteps'   : 2000,
                           'dsmin'      : 0.005,
                           'verbose'    : -1,
                           'xtol'       : 5e-8*Uw0.shape[0], 
                           'corrector'  : 'Pseudo'}
        
        # print('Currently have all conditioning turned off.')
        cont_solver = Continuation(solver, ds0=0.05, CtoP=CtoP, config=continue_config)
        
        
        XlamP_full = cont_solver.continuation(fun, Uw0, lam0, lam1)
        
        # Compare Results to the Linear FRF:
        
        Xwlinear = vib_sys.linear_frf(XlamP_full[:, -1], Fl[Ndof:2*Ndof], solver, 3)
        
        # Error including near resonance:
        # error = np.max(np.abs(XlamP_full[:, Ndof:3*Ndof] - Xwlinear[:, :-1]))
        
        away_from_resonance_mask = np.max(np.sqrt(Xwlinear[:, :3]**2 + Xwlinear[:, 3:6]**2), axis=1) < 100
        
        error = np.max(np.abs(XlamP_full[away_from_resonance_mask, Ndof:3*Ndof] \
                              - Xwlinear[away_from_resonance_mask, :-1]))
        
        self.assertLess(error, 1e-5, 
                        'Unexpectedly high error between FRF and continuation.')

    def test_full_continuation_ortho(self):
        """
        Test a full arclength continuation with the pseudo corrector
        Uses a linear system and compares against linear FRF

        Returns
        -------
        None.

        """
        
        Uw0,solver,CtoP,fun,lam0,lam1,vib_sys,Fl = self.linear_sys_data
        
        Ndof = vib_sys.M.shape[0]

        continue_config = {'DynamicCtoP': True, 
                           'TargetNfev' : 200,
                           'MaxSteps'   : 2000,
                           'dsmin'      : 0.005,
                           'verbose'    : -1,
                           'xtol'       : 5e-8*Uw0.shape[0], 
                           'corrector'  : 'Ortho'}
        
        cont_solver = Continuation(solver, ds0=0.05, CtoP=CtoP, config=continue_config)
        
        XlamP_full = cont_solver.continuation(fun, Uw0, lam0, lam1)
        
        Xwlinear = vib_sys.linear_frf(XlamP_full[:, -1], Fl[Ndof:2*Ndof], solver, 3)
        
        error = np.max(np.abs(XlamP_full[:, Ndof:3*Ndof] - Xwlinear[:, :-1]))
        
        self.assertLess(error, 1e-4, 
                        'Unexpectedly high error between FRF and continuation.')



if __name__ == '__main__':
    unittest.main()