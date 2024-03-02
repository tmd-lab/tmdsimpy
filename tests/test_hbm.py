"""
Test for verifying the correctness of Harmonic Balance Method

Automatically checks if MATLAB is present and gives a warning if it is not.     

""" 

import sys
import numpy as np
from scipy import io as sio
import unittest

import verification_utils as vutils

sys.path.append('..')
from tmdsimpy.nlforces.cubic_stiffness import CubicForce
from tmdsimpy.nlforces.cubic_damping import CubicDamping

from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy import harmonic_utils as hutils


class TestHarmonicBalance(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Check if MATLAB/Python integration is available and import MATLAB if 
        needed
        
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
        
        super(TestHarmonicBalance, self).__init__(*args, **kwargs)       
        
        # Tolerances
        self.matlab_tol = 1e-12
        self.grad_rtol = 5e-10
        self.nearlin_tol = 1e-12 # Tolerance for linear analytical v. HBM check

        
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
        
        # Verify Mass and Stiffness Matrices are Appropriate
        solver = NonlinearSolver()
        
        # lam,V = solver.eigs(M) # M must be positive definite.
        # lam,V = solver.eigs(K) # K should be at least positive semi-definite.
        lam,V = solver.eigs(K, M)
        
        vib_sys.add_nl_force(duff_force)
        
        ###########################
        # Store vibration system
        
        self.vib_sys = vib_sys
            
        #######################################################################
        ###### Setup Second Nonlinear System                             ######
        #######################################################################
        
        # Remove the rigid body mode.
        K = np.array([[3.0, 0.77, 1.8 ],
                               [0.77, 2.48, 1.71],
                               [1.8 , 1.71, 2.51]])
        
        ab_damp = [0.0001, 0.0003]
        C = ab_damp[0]*M + ab_damp[1]*K
        
        vib_sys2 = VibrationSystem(M, K, C)
        vib_sys2.add_nl_force(duff_force)
        
        self.vib_sys2 = vib_sys2
        
        
        #######################################################################
        ###### General Data that is reused                               ######
        #######################################################################
        
        h = np.array([0, 1, 2, 3, 4]) 
        
        Uw = np.array([2.79, 2.14, 4.06, 2.61, 1.02, 0.95, 1.25, 3.28, 2.09, 0.97, 4.98,
                       1.48, 1.13, 2.49, 3.34, 4.35, 0.69, 4.84, 3.27, 2.03, 3.82, 2.86,
                       0.99, 3.52, 3.8 , 3.4 , 1.89, 0.75])
        
        
        self.baseline_data = (h, Uw, ab_damp)

    def test_compare_matlab_res(self):
        """
        Compare a solution to a saved MATLAB version of the results.

        Returns
        -------
        None.

        """
        # self.assertTrue(self.check_matlab, 
        #                 'MATLAB integration failed to load, so the test fails.')
        
        ###########################
        # Evaluate Harmonic Balance Residual

        h = np.array([0, 1, 2, 3, 4]) 

        Uw = np.array([2.79, 2.14, 4.06, 2.61, 1.02, 0.95, 1.25, 3.28, 2.09, 0.97, 4.98,
               1.48, 1.13, 2.49, 3.34, 4.35, 0.69, 4.84, 3.27, 2.03, 3.82, 2.86,
               0.99, 3.52, 3.8 , 3.4 , 1.89, 0.75])

        #Uw = np.atleast_2d(Uw).T

        Fl = np.zeros((27,))
        Fl[1*3] = 1.0 # First Harmonic Cosine, DOF 1
        Fl[3*3] = 0.8 # Second Harmonic Cosine, DOF 1

        R, dRdU, dRdw = self.vib_sys.hbm_res(Uw, Fl, h, Nt=128, aft_tol=1e-7)

        ###########################
        # Compare to the MATLAB Solution

        mat_sol = sio.loadmat('./MATLAB_VERSIONS/duffing_3DOF.mat')
        
        # Residual
        error = np.linalg.norm(mat_sol['R'][:, 0] - R)
        self.assertLess(error, self.matlab_tol, 
                        'Calculated residual is different than expected.')
        
        # Gradient
        error = np.linalg.norm(mat_sol['dRdU'] - dRdU)
        self.assertLess(error, self.matlab_tol, 
                        'Calculated gradient is different than expected.')
        
        # Gradient w.r.t. w [frequency]
        error = np.linalg.norm(mat_sol['dRdw'][:, 0] - dRdw)
        self.assertLess(error, self.matlab_tol, 
                        'Calculated gradient w.r.t. w is different than expected.')

    def test_gradients(self):
        """
        Check gradients against numerical differentiation

        Returns
        -------
        None.

        """
            
        h, Uw = self.baseline_data[0:2]
        
        Fl = np.zeros((27,))
        Fl[1*3] = 1.0 # First Harmonic Cosine, DOF 1
        Fl[3*3] = 0.8 # Second Harmonic Cosine, DOF 1

        # Displacement Gradient
        fun = lambda U : self.vib_sys.hbm_res(np.hstack((U, Uw[-1])), Fl, h, 
                                              Nt=128, aft_tol=1e-7)[0:2]
        
        grad_failed = vutils.check_grad(fun, Uw[:-1], rtol=self.grad_rtol, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for displacements.')        
        
        # Frequency Gradient 
        fun = lambda w : self.vib_sys.hbm_res(np.hstack((Uw[:-1], w)), Fl, h, 
                                              Nt=128, aft_tol=1e-7)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(Uw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)

        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')  
        
    def test_gradients_skipped_harmonics(self):
        """
        Check gradients against numerical differentiation when not including 
        all harmonics

        Returns
        -------
        None.

        """
            
        h, Uw = self.baseline_data[0:2]
        
        # Remove data from the third harmonic to make sure code still works 
        # correctly
        h = np.hstack((h[0:3], h[4:]))
        Uw = np.hstack((Uw[:5*3], Uw[-2*3-1:]))
        
        Fl = np.zeros((21,))
        Fl[1*3] = 1.0 # First Harmonic Cosine, DOF 1
        Fl[3*3] = 0.8 # Second Harmonic Cosine, DOF 1
        
        # Displacement Gradient
        fun = lambda U : self.vib_sys.hbm_res(np.hstack((U, Uw[-1])), Fl, h, 
                                              Nt=128, aft_tol=1e-7)[0:2]
        
        grad_failed = vutils.check_grad(fun, Uw[:-1], rtol=self.grad_rtol, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for displacements.')        
        
        # Frequency Gradient 
        fun = lambda w : self.vib_sys.hbm_res(np.hstack((Uw[:-1], w)), Fl, h, 
                                              Nt=128, aft_tol=1e-7)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(Uw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)

        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')   
            
    def test_solution(self):
        """
        Test HBM by solving a near linear problem at a point.

        Returns
        -------
        None.

        """
        
        vib_sys = self.vib_sys2
        h, UwMisc, ab_damp = self.baseline_data

        
        solver = NonlinearSolver()
        lam,V = solver.eigs(vib_sys.K, vib_sys.M)
        
        
        h = np.array([0, 1, 2, 3, 4, 5]) 
        
        Nhc = hutils.Nhc(h)
        Ndof = vib_sys.M.shape[0]
        
        fmag = 0.0000001
        
        mi = 0 # mode of interest
        wn = np.sqrt(lam[mi])
        w = wn # Force at near resonance.
        vi = V[:, mi]
        
        Fl = np.zeros((Nhc*Ndof,))
        Fl[1*Ndof:2*Ndof] = (vib_sys.M @ vi) # First Harmonic Cosine, DOF 1
        
        
        Uw = np.zeros((Nhc*Ndof+1,))
        Uw[-1] = w 
        
        # Mode 2 proportional damping
        zeta = ab_damp[0]/w/2 +  ab_damp[1]*w/2
        
        qlinear = fmag*(vi @ Fl[1*Ndof:2*Ndof]) \
                    / np.sqrt( (wn**2 - w**2)**2 + (2*zeta*w*wn)**2)
        
        # 90 deg. phase lag near resonance.
        Uw[2*Ndof:3*Ndof] = qlinear * vi
        
        fun = lambda U : vib_sys.hbm_res(np.hstack((U, Uw[-1])), \
                                         fmag*Fl, h, Nt=128, aft_tol=1e-7)[0:2]
        
        X, R, dRdX, sol = solver.nsolve(fun, Uw[:-1], verbose=False)
        
        R_fun, dRdX_fun = fun(X)
        
        ########################
        # Verify Solver outputs at the final state, R and dRdX
        
        solver_error_R = np.abs(R-R_fun).max()
        self.assertLess(solver_error_R, 1e-16, 
                        'Solver returned wrong residual at solution.')
        
        solver_error_dRdX = np.abs(dRdX-dRdX_fun).max()
        self.assertLess(solver_error_dRdX, 1e-16, 
                        'Solver returned wrong gradient at solution.')
        
        
        ########################
        # Verify Solver accurately solved the problem
        
        linear_solve_error = np.abs(X - Uw[:-1].reshape((-1))).max()
        
        self.assertLess(linear_solve_error, self.nearlin_tol, 
                        'Failed to converge to solution for near linear HBM.')
        

    def test_nonlinear_solution(self):
        """
        Test HBM by solving a nonlinear problem at a point.

        Returns
        -------
        None.

        """
        
        vib_sys = self.vib_sys2
        h, UwMisc, ab_damp = self.baseline_data
                
        ###################
        # Initial Linear Details
        
        solver = NonlinearSolver()
        lam,V = solver.eigs(vib_sys.K, vib_sys.M)
        
        
        h = np.array([0, 1, 2, 3, 4, 5]) 
        
        Nhc = hutils.Nhc(h)
        Ndof = vib_sys.M.shape[0]
                
        mi = 0 # mode of interest
        wn = np.sqrt(lam[mi])
        w = wn # Force at near resonance.
        vi = V[:, mi]
        
        Fl = np.zeros((Nhc*Ndof,))
        Fl[1*Ndof:2*Ndof] = (vib_sys.M @ vi) # First Harmonic Cosine, DOF 1
        
        ###################
        # Nonlinear Point Solution

        fmag = 1.0

        zeta = ab_damp[0]/w/2 +  ab_damp[1]*w/2
        qlinear = fmag*(vi @ Fl[1*Ndof:2*Ndof]) / np.sqrt( (wn**2 - w**2)**2 + (2*zeta*w*wn)**2)


        Uw = np.zeros((Nhc*Ndof+1,))
        Uw[-1] = w 
        
        # 90 deg. phase lag near resonance.
        Uw[2*Ndof:3*Ndof] = (qlinear * vi)


        fun = lambda U : vib_sys.hbm_res(np.hstack((U, Uw[-1])), \
                                         fmag*Fl, h, Nt=128, aft_tol=1e-7)[0:2]

        X, R, dRdX, sol = solver.nsolve(fun, Uw[:-1], verbose=False)

        R_fun, dRdX_fun = fun(X)

        ########################
        # Verify Solver outputs at the final state, R and dRdX
        
        solver_error_R = np.abs(R-R_fun).max()
        self.assertLess(solver_error_R, 1e-16, 
                        'Solver returned wrong residual at solution.')
        
        solver_error_dRdX = np.abs(dRdX-dRdX_fun).max()
        self.assertLess(solver_error_dRdX, 1e-16, 
                        'Solver returned wrong gradient at solution.')
        

        # Check convergence of solver against an expected tolerance. 
        self.assertLess(np.linalg.norm(R), 3e-3, 
                        'Unexpectedly high residual from nonlinear HBM solution')


        ###########################
        # Verify Gradients
        fun = lambda U : vib_sys.hbm_res(np.hstack((U, Uw[-1])), fmag*Fl, h, 
                                         Nt=128, aft_tol=1e-7)[0:2]
        
        grad_failed = vutils.check_grad(fun, X, rtol=self.grad_rtol, 
                                        verbose=False)

        self.assertFalse(grad_failed, 'Incorrect gradient for displacements.')    

        fun = lambda w : vib_sys.hbm_res(np.hstack((X, w)), fmag*Fl, h,
                                         Nt=128, aft_tol=1e-7)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(Uw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)

        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')    
        
    def test_cubic_damping_gradient(self):
        """
        Test Gradients for the case of cubic damping. Duffing nonlinearity does
        not check if velocity dependent gradients are correct so this is needed
        (e.g., frequency contribution to nonlinear force are 0 for duffing)

        Returns
        -------
        None.

        """
        
        ###########################
        # Verify Gradients of system with cubic damping
        
        cnl = 0.03 # N/(m/s)^3 = N s^3 / m^3 = kg s / m^2
        
        
        # Nonlinear Force
        Q = np.array([[1.0, 0.0, 0.0]])
        T = np.array([[1.0], [0.0], [0.0]])
        
        calpha = np.array([cnl])
        
        nl_damping = CubicDamping(Q, T, calpha)
        
        # Setup Vibration System
        vib_sys_nldamp = VibrationSystem(self.vib_sys.M, 
                                         self.vib_sys.K, 
                                         self.vib_sys.C)
        
        
        vib_sys_nldamp.add_nl_force(nl_damping)
        
        
        ###########################
        # Misc Initialization and Settings
        h, Uw, ab_damp = self.baseline_data

        Fl = np.zeros((27,))
        Fl[1*3] = 1.0 # First Harmonic Cosine, DOF 1
        Fl[3*3] = 0.8 # Second Harmonic Cosine, DOF 1

        fmag = 1.0

        ###########################
        # Verify Gradients for cubic damping
        fun = lambda U : vib_sys_nldamp.hbm_res(np.hstack((U, Uw[-1])), 
                                                fmag*Fl, h, Nt=128, aft_tol=1e-7)[0:2]
        
        grad_failed = vutils.check_grad(fun, Uw[:-1], rtol=self.grad_rtol*10, 
                                        verbose=False)
        
        
        self.assertFalse(grad_failed, 'Incorrect gradient for displacements.')    
        
        fun = lambda w : vib_sys_nldamp.hbm_res(np.hstack((Uw[:-1], w)), fmag*Fl, h, 
                                                Nt=128, aft_tol=1e-7)[0:3:2]
        grad_failed = vutils.check_grad(fun, np.atleast_1d(Uw[-1]), 
                                        rtol=self.grad_rtol, verbose=False)

        self.assertFalse(grad_failed, 'Incorrect gradient for frequency.')    

        
    def test_hbm_Fl_res(self):
        """
        Check gradients and values of the residual function for Fl continuation 
        of HBM.

        Returns
        -------
        None.

        """
            
        h, Uw = self.baseline_data[0:2]
        w = Uw[-1]
        
        Fl = np.zeros((27,))
        Fl[1*3] = 1.0 # First Harmonic Cosine, DOF 1
        Fl[3*3] = 0.8 # Second Harmonic Cosine, DOF 1
        
        Fmag = 3.245
        
        # Normal HBM Residual and Gradient to compare against:
        Rhbm,dRdUhbm = self.vib_sys.hbm_res(Uw, Fmag*Fl, h, 
                                              Nt=128, aft_tol=1e-7)[0:2]
        
        # Fl continuation function call
        R2,dR2dUhbm = self.vib_sys.hbm_res_dFl(np.hstack((Uw[:-1], Fmag)), 
                                               w, Fl, h, 
                                               Nt=128, aft_tol=1e-7)[0:2]
        
        # Should be exact since one function just wraps the other for the 
        # first two outputs
        self.assertLess(np.linalg.norm(Rhbm-R2), 1e-12)
        self.assertLess(np.linalg.norm(dRdUhbm-dR2dUhbm), 1e-12)
        
        
        # Displacement Gradient
        fun = lambda U : self.vib_sys.hbm_res_dFl(np.hstack((U, Fmag)),
                                                  w, Fl, h, 
                                                  Nt=128, aft_tol=1e-7)[0:2]
        
        grad_failed = vutils.check_grad(fun, Uw[:-1], rtol=self.grad_rtol*10, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for displacements.')        
        
        # Frequency Gradient 
        fun = lambda Fmag : self.vib_sys.hbm_res_dFl(np.hstack((Uw[:-1], Fmag)),
                                                     w, Fl, h, 
                                              Nt=128, aft_tol=1e-7)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(Uw[-1]), 
                                        rtol=self.grad_rtol*10, verbose=False)

        self.assertFalse(grad_failed, 'Incorrect gradient for force magnitude scaling.')   

    def test_hbm_Fl_h0(self):
        """
        Check gradients and values of the residual function for Fl continuation 
        of HBM when there is a static force. 
        
        Static force should not be scaled by Fmag

        Returns
        -------
        None.

        """
            
        h, Uw = self.baseline_data[0:2]
        w = Uw[-1]
        
        Fl = np.zeros((27,))
        Fl[0] = 3.14
        Fl[1] = 2.15
        Fl[1*3] = 1.0 # First Harmonic Cosine, DOF 1
        Fl[3*3] = 0.8 # Second Harmonic Cosine, DOF 1
        
        Fmag = 3.245
        
        Fl_scaled = np.copy(Fl)
        Fl_scaled[3:] *= Fmag
        
        # Normal HBM Residual and Gradient to compare against:
        Rhbm,dRdUhbm = self.vib_sys.hbm_res(Uw, Fl_scaled, h, 
                                              Nt=128, aft_tol=1e-7)[0:2]
        
        # Fl continuation function call
        R2,dR2dUhbm = self.vib_sys.hbm_res_dFl(np.hstack((Uw[:-1], Fmag)), 
                                               w, Fl, h, 
                                               Nt=128, aft_tol=1e-7)[0:2]
        
        # Should be exact since one function just wraps the other for the 
        # first two outputs
        self.assertLess(np.linalg.norm(Rhbm-R2), 1e-12)
        self.assertLess(np.linalg.norm(dRdUhbm-dR2dUhbm), 1e-12)
        
        
        # Displacement Gradient
        fun = lambda U : self.vib_sys.hbm_res_dFl(np.hstack((U, Fmag)),
                                                  w, Fl, h, 
                                                  Nt=128, aft_tol=1e-7)[0:2]
        
        grad_failed = vutils.check_grad(fun, Uw[:-1], rtol=self.grad_rtol*10, 
                                        verbose=False)
        
        self.assertFalse(grad_failed, 'Incorrect gradient for displacements.')        
        
        # Frequency Gradient 
        fun = lambda Fmag : self.vib_sys.hbm_res_dFl(np.hstack((Uw[:-1], Fmag)),
                                                     w, Fl, h, 
                                              Nt=128, aft_tol=1e-7)[0:3:2]
        
        grad_failed = vutils.check_grad(fun, np.atleast_1d(Uw[-1]), 
                                        rtol=self.grad_rtol*10, verbose=False)

        self.assertFalse(grad_failed, 
                         'Incorrect gradient for force magnitude scaling.')   
        
if __name__ == '__main__':
    unittest.main()