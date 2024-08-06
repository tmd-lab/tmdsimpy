"""

Script for testing the accuracy of the elastic dry friction implementation 
that uses JAX - Compares AFT results to those from vector Jenkins

Tests include:
    1. Low and high normal displacement checks against Jenkins with different
    values of Fs
    2. Verify second harmonic of normal induces second harmonic of
    tangential force when slipping is present (tests 3 and 5)
    3. Verify if separated that all is zero.

"""

# Standard imports
import numpy as np
import sys
import unittest

# Python Utilities
sys.path.append('../..')

# # vectorized (non JAX version)
# from tmdsimpy.jax.nlforces.jenkins_element import JenkinsForce 

# JAX version for elastic dry friction
from tmdsimpy.jax.nlforces import ElasticDryFriction3D 

# JAX version for elastic dry friction
from tmdsimpy.jax.nlforces import ElasticDryFriction2D 

import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.vibration_system import VibrationSystem


sys.path.append('..')
import verification_utils as vutils


###############################################################################
###     Testing Class                                                       ###
###############################################################################

def run_comparison(obj, Unl, w, h, Nt, force_tol, df_tol, eldry2d, eldry3d,
                   delta_grad=1e-5):
    
        dof = Unl.shape[0]
        
        # Create a mask tangent x and normal
        txn = np.zeros(dof, dtype=bool)
        txn[0::3] = True
        txn[2::3] = True

        tyn = np.zeros(dof, dtype=bool)
        tyn[1::3] = True
        tyn[2::3] = True
         
       
        FnlH_vec, dFnldUH_vec, dFnldw_vec \
            = eldry2d.aft(Unl[txn,:], w, h, Nt=Nt)
        
        FnlH, dFnldUH, dFnldw = eldry3d.aft(Unl, w, h, Nt=Nt)
        
        
        FH_error = np.max(np.abs(FnlH[txn]-FnlH_vec))
        
        
        obj.assertLess(FH_error, force_tol, 
                        'Incorrect elastic dry friction force in X direction.')
                
        ###############
        # Tangent - Tangent Gradient in X direction
        dFH_error = np.max(np.abs(dFnldUH[np.ix_(txn,txn)]-dFnldUH_vec))
        
        obj.assertLess(dFH_error, df_tol, 
                        'Incorrect Tangential AFT gradient in X direction.')
        
        
        
        FnlH_vec, dFnldUH_vec, dFnldw_vec \
            = eldry2d.aft(Unl[tyn,:], w, h, Nt=Nt)
        
        FnlH, dFnldUH, dFnldw = eldry3d.aft(Unl, w, h, Nt=Nt)
        
        
        FH_error = np.max(np.abs(FnlH[tyn]-FnlH_vec))
        
        
        obj.assertLess(FH_error, force_tol, 
                        'Incorrect elastic dry friction force in Y direction.')
                
        ###############
        # Tangent - Tangent Gradient in Y direction
        dFH_error = np.max(np.abs(dFnldUH[np.ix_(tyn,tyn)]-dFnldUH_vec))
        
        obj.assertLess(dFH_error, df_tol, 
                        'Incorrect Tangential AFT gradient in Y direction.')
        ###############
        # Numeric Gradient check, should capture dTangent/dNormal terms
        
        # Check gradient - Unl
        fun = lambda U : eldry3d.aft(U, w, h, Nt=Nt)[0:2]
        
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, 
                                        atol=obj.atol_grad, h=delta_grad)
        
        obj.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. Unl.')
        
        # Gradient w
        fun = lambda w : eldry3d.aft(Unl, w, h, Nt=Nt)[0::2]
        
        grad_failed = vutils.check_grad(fun, np.array([w]), verbose=False, 
                                        atol=obj.atol_grad)

        obj.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. w.')
        
class TestJAXEldry(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestJAXEldry, self).__init__(*args, **kwargs)      
        
        
        force_tol = 5e-15 # All should be exactly equal
        df_tol = 1e-14 # rounding error on the derivatives
        dfdw_tol = 1e-16 # everything should have zero derivative w.r.t. frequency
        self.atol_grad = 1e-9

        self.tols = (force_tol, df_tol, dfdw_tol)


        # Simple Mapping to displacements - eldry
        Q_3D = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
        T_3D = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
        
        # Mapping for 2D elastic dry x direction
        Q_2D = np.array([[1.0, 0.0], [ 0.0, 1.0]])
        T_2D = np.array([[1.0, 0.0], [0.0, 1.0]])
        

        
        kt = 2.0
        kn = 2.5
        mu = 0.75
        
        self.un_low = 1.5
        self.un_high = 5.7
        

        
        self.eldry2D_force = ElasticDryFriction2D(Q_2D, T_2D, kt, kn, mu, 
                                              u0=np.array([0.0]))
        
        self.eldry3D_force = ElasticDryFriction3D(Q_3D, T_3D, kt, kn, mu, 
                                                u0=np.array([0.0]))
        
        
        # Create Two eldry Force options with different u0 and verify in the 
        # stuck regime that they give the correct forces for harmonic 0
        self.eldry_force_3D = ElasticDryFriction3D(Q_3D, T_3D, kt, kn, mu, u0=np.array([0.0]))
        self.eldry_force2_2D = ElasticDryFriction2D(Q_2D, T_2D, kt, kn, mu, u0=np.array([0.0]))
        

        self.eldry_force3_3D = ElasticDryFriction3D(Q_3D, T_3D, kt, kn, mu, u0=np.array([0.2]))
        self.eldry_force3_2D = ElasticDryFriction2D(Q_2D, T_2D, kt, kn, mu, u0=np.array([0.2]))
        

        
        
        # 1.5 Times Eldry
        Q_15 = np.array([[1.0, 0.0, 0.0], 
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0], 
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])
        
        T_15 = np.array([[1.0, 0.0, 0.0, 0.5, 0.0, 0.0], 
                         [0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0, 0.5]])
        
        self.eldry_force15 = ElasticDryFriction3D(Q_15, T_15, kt, kn, mu, u0=0.0)
        
        # Split Eldry
        Qsplit = np.eye(6)
        Tsplit = np.diag([0.5, 0.5,0.5, 1.5, 1.5,1.5])
        
        self.eldry_force_split = ElasticDryFriction3D(Qsplit, Tsplit, kt, kn, mu, u0=0.0)

    def test_eldry1(self):
        """
        Test some basic / straightforward motion comparisons 
        (highish amplitude)

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        
        
        ###### Test Parameters
        Nt = 1 << 7
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.75, 2.0, 1.3, 0.0, 0.0, 1.0, 0.0],
                        [0.75, 2.0, 1.3, 0.0, 0.0, 1.0, 0.0],
                          [self.un_low, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*5
        Unl[:, 1] = Unl[:, 1]*3
        Unl = Unl.reshape(-1,1)
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.eldry2D_force, self.eldry3D_force)

        
        Unl[2] = self.un_high
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.eldry2D_force, self.eldry3D_force)
        

        
        
        return
        
    def test_eldry2(self):
        """
        Test low amplitude motion

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        
        
        ###### Test Parameters
        Nt = 1 << 7
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0],
                        [0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0],
                          [self.un_low, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        

        Unl[:, 0] = Unl[:, 0]*0.2
        Unl[:, 1] = Unl[:, 1]*0.3
        
        Unl = Unl.reshape(-1,1)
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.eldry2D_force, self.eldry3D_force)
        
        
        Unl[2] = self.un_high
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.eldry2D_force, self.eldry3D_force)

    def test_eldry3(self):
        """
        Test very high amplitude motion

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        
        
        ###### Test Parameters
        Nt = 1 << 7
        w = 1.7
        
        ##### Verification

        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0],
                        [0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0],
                          [self.un_low, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*10000000000.0
        Unl[:, 1] = Unl[:, 1]*20000000000.0
        
        Unl = Unl.reshape(-1,1)
        
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.eldry2D_force, self.eldry3D_force)
        
        
        Unl[2] = self.un_high
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.eldry2D_force, self.eldry3D_force)
        
        # Verify that 2nd harmonic of normal would induce same in tangent
        # [t0, n0, t1c, n1c, t1s, n1s, t2c, n2c, ]
        FnlH, dFnldUH, dFnldw = self.eldry3D_force.aft(Unl, w, h, Nt=Nt)
        
        ## Not clear 
        # self.assertGreater(np.abs(dFnldUH[9,11]), 0.1, 
        #                    'Normal load may not be influencing x tangent force.')
        
        # self.assertGreater(np.abs(dFnldUH[10,11]), 0.1, 
        #                    'Normal load may not be influencing y tangent force.')

    def test_eldry4(self):
        """
        Test moderate amplitude case

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        
        
        ###### Test Parameters
        Nt = 1 << 7
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0],
                        [0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0],
                          [self.un_low, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*5
        Unl[:, 1] = Unl[:, 1]*7
        
        Unl = Unl.reshape(-1,1)
        
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.eldry2D_force, self.eldry3D_force, delta_grad=3e-6)
        

        
        Unl[2] = self.un_high
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.eldry2D_force, self.eldry3D_force)
        
    def test_eldry5(self):
        """
        Test moderate amplitude case

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        
        
        ###### Test Parameters
        Nt = 1 << 7
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[4.29653115, 4.29165565, 2.8307871 , 4.17186848, 3.37441948,\
                           0.80543152, 3.55638299],
                        [4.29653115, 4.29165565, 2.8307871 , 4.17186848, 3.37441948,\
                                           0.80543152, 3.55638299],
                          [self.un_low, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*5
        Unl[:, 1] = Unl[:, 1]*7
        
        Unl = Unl.reshape(-1,1)
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.eldry2D_force, self.eldry3D_force)
        
        
        Unl[2] = self.un_high
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.eldry2D_force, self.eldry3D_force)
        
        # Verify that 2nd harmonic of normal would induce same in tangent
        # [t0, n0, t1c, n1c, t1s, n1s, t2c, n2c, ]
        FnlH, dFnldUH, dFnldw = self.eldry3D_force.aft(Unl, w, h, Nt=Nt)
        
        # self.assertGreater(np.abs(dFnldUH[6,7]), 0.01, 
        #                    'Normal load may not be influencing tangent force.')
        
    def test_h0_force_opts(self):
        """
        Test moderate amplitude case

        Returns
        -------
        None.

        """
        
        h = np.array([0, 1, 2, 3])
        Unl = np.zeros((7*3,1))
        Unl[0] = 0.15
        Unl[1] = 0.30
        Unl[2] = 100.0
        
        w = 1.75
        Nt = 1 << 7
        
        dof = Unl.shape[0]
        
        # Create a mask tangent x and normal
        txn = np.zeros(dof, dtype=bool)
        txn[0::3] = True
        txn[2::3] = True


        # Create a mask tangent y and normal
        tyn = np.zeros(dof, dtype=bool)
        tyn[1::3] = True
        tyn[2::3] = True
                
        # Force Values for harmonic zero

        #################  tangent x direction
        # U0 < Unl[0]
        FnlH2, dFnldUH, dFnldw = self.eldry_force2_3D.aft(Unl, w, h, Nt=Nt)
        
        FnlH2_2D, dFnldUH_2D, dFnldw_2D = self.eldry_force2_2D.aft(Unl[txn,:], w, h, Nt=Nt)
        

        error1 = np.max(np.abs(FnlH2[txn] - FnlH2_2D))
        error2 = np.max(np.abs(dFnldUH[np.ix_(txn,txn)] - dFnldUH_2D))
        error3 = np.max(np.abs(dFnldw[txn] - dFnldw_2D))


        self.assertLess(error1, 1e-18, 
                        'Static force from prestressed state is incorrect in tangent X.')
        
        self.assertLess(error2, 1e-18, 
                        'gradiaent wrt displacement from prestressed state is incorrect in tangent X.')
        
        self.assertLess(error3, 1e-18, 
                        'gradient wrt freq from prestressed state is incorrect in tangent X.')
        
        
        ###################  tangent y direction
        
        FnlH2_2D, dFnldUH_2D, dFnldw_2D = self.eldry_force2_2D.aft(Unl[tyn,:], w, h, Nt=Nt)

        error1 = np.max(np.abs(FnlH2[tyn] - FnlH2_2D))
        error2 = np.max(np.abs(dFnldUH[np.ix_(tyn,tyn)] - dFnldUH_2D))
        error3 = np.max(np.abs(dFnldw[tyn] - dFnldw_2D))


        self.assertLess(error1, 1e-18, 
                        'Static force from prestressed state is incorrect in tangent Y.')
        
        self.assertLess(error2, 1e-18, 
                        'gradiaent wrt displacement from prestressed state is incorrectin tangent Y.')
        
        self.assertLess(error3, 1e-18, 
                        'gradient wrt freq from prestressed state is incorrect in tangent Y.')
        
        #######################
        
        # Force Values for harmonic zero

        #################  tangent x direction
        # U0 < Unl[0]
        FnlH2, dFnldUH, dFnldw = self.eldry_force3_3D.aft(Unl, w, h, Nt=Nt)
        
        FnlH2_2D, dFnldUH_2D, dFnldw_2D = self.eldry_force3_2D.aft(Unl[txn,:], w, h, Nt=Nt)
        

        error1 = np.max(np.abs(FnlH2[txn] - FnlH2_2D))
        error2 = np.max(np.abs(dFnldUH[np.ix_(txn,txn)] - dFnldUH_2D))
        error3 = np.max(np.abs(dFnldw[txn] - dFnldw_2D))


        self.assertLess(error1, 1e-18, 
                        'Static force from initial displacement is incorrect in tangent X.')
        
        self.assertLess(error2, 1e-18, 
                        'gradiaent wrt displacement from initial displacement is incorrect in tangent X.')
        
        self.assertLess(error3, 1e-18, 
                        'gradient wrt freq from initial displacement is incorrect in tangent X.')
        
        
        ###################  tangent y direction
        
        FnlH2_2D, dFnldUH_2D, dFnldw_2D = self.eldry_force3_2D.aft(Unl[tyn,:], w, h, Nt=Nt)
        

        error1 = np.max(np.abs(FnlH2[tyn] - FnlH2_2D))
        error2 = np.max(np.abs(dFnldUH[np.ix_(tyn,tyn)] - dFnldUH_2D))
        error3 = np.max(np.abs(dFnldw[tyn] - dFnldw_2D))


        self.assertLess(error1, 1e-18, 
                        'Static force from initial displacement is incorrect in tangent Y.')
        
        self.assertLess(error2, 1e-18, 
                        'gradiaent wrt displacement from initial displacement is incorrectin tangent Y.')
        
        self.assertLess(error3, 1e-18, 
                        'gradient wrt freq from initial displacement is incorrect in tangent Y.')
        ############
        
        
        
        # Check gradient
        fun = lambda U : self.eldry_force2_3D.aft(U, w, h)[0:2]
        
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, 
                                        atol=self.atol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient from u0 setting.')
        
        #################
        # U0 > Unl[0]
        FnlH3, dFnldUH, dFnldw = self.eldry_force3_3D.aft(Unl, w, h, Nt=Nt)
        
        
        error3 = FnlH3[0] - (Unl[0] - self.eldry_force3_3D.u0)*self.eldry_force3_3D.kt
        
        self.assertLess(error3, 1e-18, 
                        'Static force from prestressed state is incorrect.')
        
        
        # Check gradient
        fun = lambda U : self.eldry_force3_3D.aft(U, w, h)[0:2]
        
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, 
                                        atol=self.atol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient from u0 setting.')
        # Gradients are correct?
    
    def test_separation(self):
        """
        Test separated contact forces

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        
        
        ###### Test Parameters
        Nt = 1 << 7
        w = 1.7
        h = np.array([0, 1, 2, 3])
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[4.29653115, 4.29165565, 2.8307871 , 4.17186848, 3.37441948,\
                           0.80543152, 3.55638299],
                        [4.29653115, 4.29165565, 2.8307871 , 4.17186848, 3.37441948,\
                                           0.80543152, 3.55638299],
                          [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*5
        Unl[:, 1] = Unl[:, 1]*3
        
        Unl = Unl.reshape(-1,1)
        
        
        FnlH, dFnldUH, dFnldw = self.eldry3D_force.aft(Unl, w, h, Nt=Nt)
        
        self.assertEqual(FnlH.sum(), 0, 
                         'Contact forces should be zero when out of contact')
        
        self.assertEqual(dFnldUH.sum(), 0, 
                         'Contact gradient should be zero when out of contact')
        
        # Verify at zero in contact that there are not NaN's in gradient etc.
        Unl[2] = 0
        FnlH, dFnldUH, dFnldw = self.eldry3D_force.aft(Unl, w, h, Nt=Nt)

        self.assertEqual(FnlH.sum(), 0, 
                         'NaNs in contact force when barely in contact')
        
        self.assertFalse(np.isnan(dFnldUH.sum()), 
                         'NaNs in gradient when just barely in contact')
        
    def test_mdof_eldry(self):
        """
        Test an elastic dry friction version with multiple degrees of freedom
        and multiple independent 

        Returns
        -------
        None.

        """
        
        compare_tol = 2e-14
        
        # Friction Versions
        eldry_sdof = self.eldry3D_force
        eldry_times_1p5 = self.eldry_force15
        eldry_split = self.eldry_force_split
        
        Tsplit = eldry_split.T
        
        # Misc Parameters to use
        Nt = 1 << 7
        w = 1.7
        h = np.array([0, 1, 2, 3])
        Nhc = hutils.Nhc(h)
        
        # Generate some baseline vectors for the first and second DOFs
        np.random.seed(1023)
        U1 = np.random.rand(Nhc*3)
        U2 = np.random.rand(Nhc*3)
        # All random values are positive, so will be in contact
        
        # Allow tangential displacements to be positive and negative
        U1[::3] = U1[::3] - 0.5
        U2[::3] = U2[::3] - 0.5
        
        
        U1[1::3] = U1[1::3] - 0.3
        U2[1::3] = U2[1::3] - 0.3
        
        # Scale the tangential displacements by different values
        scale_tan = [0.1, 1.0, 10.0, 100.0]

        for scale in scale_tan:
            
            # Baseline from U1
            U1scale = np.copy(U1)
            U1scale[::3] = scale*U1scale[::3]
            U1scale[1::3] = scale*U1scale[1::3]
            
            F1 = eldry_sdof.aft(U1scale, w, h, Nt=Nt)[0]
            
            # Baseline from U2
            U2scale = np.copy(U2)
            U2scale[::3] = scale*U2scale[::3]
            U2scale[1::3] = scale*U2scale[1::3]
            
            F2 = eldry_sdof.aft(U2scale, w, h, Nt=Nt)[0]
            
            ############################
            # Combined Solution for 1.5 times
            F_15 = eldry_times_1p5.aft(U1scale, w, h, Nt=Nt)[0]
            
            self.assertLess(np.abs(F_15 - 1.5*F1).max(), compare_tol, 
                            'Using 2 DOFs to scale force failed.')
            
            ############ Gradient
            
            fun = lambda U : eldry_times_1p5.aft(U, w, h)[0:2]
            
            grad_failed = vutils.check_grad(fun, U1scale, verbose=False, 
                                            atol=self.atol_grad)
            
            self.assertFalse(grad_failed, 
                             'Incorrect Gradient for 1.5 times scaling force.')
        
            #######################
            # Split Solution
            Ucombo = np.zeros(Nhc*6)
            Ucombo[::6] = U1[::3]*scale # Tanget
            Ucombo[1::6] = U1[1::3]*scale # Tangent
            Ucombo[2::6] = U1[2::3] # Normal
            Ucombo[3::6] = U2[::3]*scale # Tangent 
            Ucombo[4::6] = U2[1::3]*scale # Tanegnt
            Ucombo[5::6] = U2[2::3] #Normal
            
            Fcombo = eldry_split.aft(Ucombo, w, h, Nt=Nt)[0]
            
            Fcombo_ref = np.zeros_like(Fcombo)
            Fcombo_ref[::6]  = Tsplit[0,0]*F1[::3] # Tanget
            Fcombo_ref[1::6] = Tsplit[1,1]*F1[1::3] # Tangent
            Fcombo_ref[2::6] = Tsplit[2,2]*F1[2::3] # Normal 
            Fcombo_ref[3::6]  = Tsplit[3,3]*F2[::3] # Tanget
            Fcombo_ref[4::6] = Tsplit[4,4]*F2[1::3] # Tangent
            Fcombo_ref[5::6] = Tsplit[5,5]*F2[2::3] # Normal
            
            self.assertLess(np.abs(Fcombo - Fcombo_ref).max(), compare_tol, 
                            'Using 2 DOFs to scale force failed.')
            
            
            ############ Gradient
            fun = lambda U : eldry_split.aft(U, w, h)[0:2]
            
            grad_failed = vutils.check_grad(fun, Ucombo, verbose=False, 
                                            atol=self.atol_grad)
            
            self.assertFalse(grad_failed, 
                             'Incorrect Gradient for combined two force test.')
        
    def test_static_eldry(self):
        """
        Test static forces and gradients in three regimes:
            1. Out of contact
            2. In Contact and stuck
            3. In Contact and Slipping
        """
        
        rtol = 1e-12
        valtol = 1e-12
        
        eldry3D_force = self.eldry3D_force
        
        eldry3D_force.init_history()
        
        fun = lambda X : eldry3D_force.force(X)
        
        ###########
        
        # kt = 2.0
        # kn = 2.5
        # mu = 0.75
        
        # Test Cases
        Uout = np.array([0.0, 0.0, -1.0])
        Ustuck = np.array([0.1, -0.1, 1.0])
        Ustuck2 = np.array([-0.1, 0.1, 1.0])
        Uslip = np.array([15.0, -8.0, 1.0])
        Uslip2 = np.array([-15.0, 8.0, 1.0])
        
        Fout = np.array([0.0, 0.0, 0.0])
        Fstuck = np.array([0.2, -0.2, 2.5])
        Fstuck2 = np.array([-0.2, 0.2, 2.5])
        Fslip = np.array([1.875, -1.875, 2.5])
        Fslip2 = np.array([-1.875, 1.875, 2.5])
        
        U_list = [Uout, Ustuck, Ustuck2, Uslip, Uslip2]
        F_list = [Fout, Fstuck, Fstuck2, Fslip, Fslip2]
        
        ###########
        
        for i in range(len(U_list)):
            
            U = U_list[i]
            F = F_list[i]

            fnl = fun(U)[0]
            
            self.assertLess(np.linalg.norm(F - fnl), valtol, 
                            'Static force is incorrect for index {}'.format(i))
            
            grad_failed = vutils.check_grad(fun, U, verbose=False, rtol=rtol)
            
            self.assertFalse(grad_failed, 
                         'Incorrect Gradient for static index {}'.format(i))
        pass

        
    def test_eldry_test_7elem(self):
        """
        check elastic dry friction code with nl forces added individually vs
        added together
        """
        #check force and jacobian
        rng = np.random.default_rng(12345)
        Q_3D = rng.random((21,10))
        T_3D=Q_3D.T
        M = rng.random((10,10))
        K = rng.random((10,10))
        bdof=3
        u1=rng.random(Q_3D.shape[1])
        Fs=np.zeros_like(u1)
        
        inputpars_kt = rng.random(Q_3D.shape[0]//3)
        inputpars_kn = rng.random(Q_3D.shape[0]//3)
        inputpars_mu = np.abs(rng.random(Q_3D.shape[0]//3))
        
        
        vib_sys_together = VibrationSystem(M, K)
        
        fnl_force1 = ElasticDryFriction3D(Q_3D, T_3D, inputpars_kt, inputpars_kn,\
                                          inputpars_mu, u0=0)

        vib_sys_together.add_nl_force(fnl_force1)
        
        Fnl_together, dFnldU_together  = vib_sys_together.static_res(u1,Fs)
            
        vib_sys_individual = VibrationSystem(M, K)
        for quad in range(Q_3D.shape[0]//3):
            start_b = quad*bdof
            end_b = quad* bdof + bdof
            fnl_force2 = ElasticDryFriction3D(Q_3D[start_b:end_b,:],T_3D[:,start_b:end_b],
                        inputpars_kt[quad], inputpars_kn[quad], inputpars_mu[quad], u0=0)
            vib_sys_individual.add_nl_force(fnl_force2)
            
        Fnl_individual, dFnldU_individual  = vib_sys_individual.static_res(u1, Fs)
        
        error = np.linalg.norm(Fnl_together - Fnl_individual)
        
        self.assertLess(error, self.atol_grad, 
                        'Nonlinear force with individual friction element does\
                            not match with combined elements.')
        
        error = np.linalg.norm(dFnldU_together - dFnldU_individual )
        
        self.assertLess(error, self.atol_grad, 
                        'Nonlinear force Jacobian with individual friction element\
                            does not match with combined elements')
        
        h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) # Automate Checking with this
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        
        Nd = Q_3D.shape[1]
        
        U = rng.random((Nd*Nhc, 1))
        
        
        w = 1 # Test for various w
        
        # Testing Simple First Harmonic Motion        
        Fnl_together, dFnldU_together = vib_sys_together.total_aft(U, w, h)[0:2] 
        Fnl_individual, dFnldU_individual = vib_sys_individual.total_aft(U, w, h)[0:2]
        
        error = np.linalg.norm(Fnl_together - Fnl_individual)
        
        self.assertLess(error, self.atol_grad, 
                        'Aft of Nonlinear force with individual friction element does'\
                           +' not match with combined elements.')
        
        error = np.linalg.norm(dFnldU_together - dFnldU_individual )
        
        self.assertLess(error, self.atol_grad, 
                        'Aft of Nonlinear force Jacobian with individual friction element'\
                            +'does not match with combined elements')                
        
        # Update history variables after static so sliders reset
        # If you do not do this, then the residual traction field will be different.
        vib_sys_together.update_force_history(2*u1)
        vib_sys_individual.update_force_history(2*u1)
                      
        error = np.linalg.norm(vib_sys_together.static_res(u1, Fs)[0] \
                               - vib_sys_individual.static_res(u1, Fs)[0] )
        
        self.assertLess(error, self.atol_grad, 
                        'Nonlinear force with individual friction element does\
                            not match with combined elements after update force history.')  
        

        vib_sys_together.set_aft_initialize(2*u1)
        vib_sys_individual.set_aft_initialize(2*u1)
        
        # Testing Simple First Harmonic Motion        
        Fnl_together, dFnldU_together = vib_sys_together.total_aft(U, w, h)[0:2] 
        Fnl_individual, dFnldU_individual = vib_sys_individual.total_aft(U, w, h)[0:2]
        
        error = np.linalg.norm(Fnl_together - Fnl_individual)
        
        self.assertLess(error, self.atol_grad, 
                        'Aft of Nonlinear force with individual friction element does'\
                           +' not match with combined elements.')
        
        error = np.linalg.norm(dFnldU_together - dFnldU_individual )
        
        self.assertLess(error, self.atol_grad, 
                        'Aft of Nonlinear force Jacobian with individual friction element'\
                            +'does not match with combined elements')  
            
            
    def test_meso_gap(self):
        """
        check if the function works correctly for meso gap with multiple 
        friction elements. Here, 2 test cases are generated where with gap 
        test case has stat displacement delta and without gap
        """
        #check force and jacobian
        Ndofs = 21
        rng = np.random.default_rng(12345)
        Q_3D = np.eye(Ndofs)
        T_3D=Q_3D.T
        M = rng.random((Ndofs,Ndofs))
        K = rng.random((Ndofs,Ndofs))
      
        inputpars_kt = rng.random(Q_3D.shape[0]//3)
        inputpars_kn = rng.random(Q_3D.shape[0]//3)
        inputpars_mu = np.abs(rng.random(Q_3D.shape[0]//3))
        
        meso_gap =  np.array([0.05, 0.04, 0.03, 0.01, 0.03, 0.04, 0.05])
        # meso_gap1 =  np.array([0, 0, 0, 0, 0, 0, 0])
        u0wogap = rng.random(Q_3D.shape[1])
        
        # Modify specific elements of u0wogap
        u0wogap[2::3] = np.array([-0.01, 0.06, 0.05, 0.0, 0.03, 0.02, 0.10])
        
        # Create an independent copy of u0wogap
        u0wgap = u0wogap.copy()
        
        # Add meso_gap to every second element of the copy
        u0wgap[2::3] = u0wgap[2::3] + meso_gap
        
        Fs=np.zeros_like(u0wgap)
        
        vib_sys_wgap = VibrationSystem(M, K)
    
        fnl_force1 = ElasticDryFriction3D(Q_3D, T_3D, inputpars_kt, inputpars_kn,\
                                          inputpars_mu, u0=0, meso_gap=meso_gap)

        vib_sys_wgap.add_nl_force(fnl_force1)
        
        #system without gap
        vib_sys_wogap = VibrationSystem(M, K)
        
        fnl_force2 = ElasticDryFriction3D(Q_3D, T_3D, inputpars_kt, inputpars_kn,\
                                          inputpars_mu, u0=0, meso_gap=0)

        vib_sys_wogap.add_nl_force(fnl_force2)
        
        
        Fnl_wgap, dFnldU_wgap  = vib_sys_wgap.static_res(u0wgap,Fs + K@ (u0wgap-u0wogap) )
        Fnl_wogap, dFnldU_wogap  = vib_sys_wogap.static_res(u0wogap,Fs)
        
        
        error = np.linalg.norm(Fnl_wgap - Fnl_wogap)

        self.assertLess(error, self.atol_grad, 
                        'Nonlinear force with gap '\
                            +'does not match without gap')   
        
        error = np.linalg.norm(dFnldU_wgap - dFnldU_wogap )
        
        self.assertLess(error, self.atol_grad, 
                        'Nonlinear force Jacobian with gap'\
                            +'does not match without gap')                

        #AFT check
        h = np.array([0, 1, 2, 3, 4, 5, 6, 7]) # Automate Checking with this
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        
        Nd = Q_3D.shape[1]
        
        Ugap = rng.random((Nd*Nhc, 1))
        U_nogap = Ugap.copy() 
        
        Ugap[:Ndofs] = u0wgap.reshape(-1,1)
        U_nogap[:Ndofs] = u0wogap.reshape(-1,1)
        
        Ugap[Ndofs:2*Ndofs]=1 # to get oscillations 
        U_nogap[Ndofs:2*Ndofs]=1
        
        w = 1.7 # Test for various w
        
        # Testing Simple First Harmonic Motion        
        Fnl_wgap, dFnldU_wgap  = vib_sys_wgap.total_aft(Ugap, w, h)[0:2] 
        Fnl_wogap, dFnldU_wogap = vib_sys_wogap.total_aft(U_nogap, w, h)[0:2]
        
        error = np.linalg.norm(Fnl_wgap - Fnl_wogap)
        
        self.assertLess(error, self.atol_grad, 
                        'Aft of Nonlinear force with gap'\
                           +' not match without gap')
        
        error = np.linalg.norm(dFnldU_wgap - dFnldU_wogap )
        
        self.assertLess(error, self.atol_grad, 
                        'Aft of Nonlinear force with gap'\
                            +'does not match without')       
            
    def test_aft_no_grad(self): 
        """
        check if the function works correctly for meso gap with multiple 
        friction elements. Here, 2 test cases are generated where with gap 
        test case has stat displacement delta and without gap
        """

        # Simple Mapping to displacements - eldry
        Q_3D = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
        T_3D = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
        
        inputpars_kt = 2.0
        inputpars_kn = 2.5
        inputpars_mu = 0.75
            
        fnl_force = ElasticDryFriction3D(Q_3D, T_3D, inputpars_kt, inputpars_kn,\
                                          inputpars_mu, u0=0)
        
        ###### Test Parameters
        Nt = 1 << 7
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        

        U = np.array([[4.29653115, 4.29165565, 2.8307871 , 4.17186848, 3.37441948,\
                           0.80543152, 3.55638299],
                        [2*4.29653115, 2*4.29165565, 2*2.8307871 , 3*4.17186848, -1*3.37441948,\
                                           -1*0.80543152, 2*3.55638299],
                          [-0.2, 0.2, 0.4, 0.0, -0.03, 1, -1]]).T     
        U = U.reshape(-1,1)
            
        
        res_default = fnl_force.aft(U, w, h, Nt=Nt)
        res_true_grad = fnl_force.aft(U, w, h, Nt=Nt, calc_grad=True)
        res_no_grad = fnl_force.aft(U, w, h, Nt=Nt, calc_grad=False)
        
        self.assertEqual(len(res_default), 3, 'Default AFT returns wrong number of outputs')
        self.assertEqual(len(res_no_grad), 1, 'No Grad AFT returns wrong number of outputs')
        self.assertEqual(len(res_true_grad), 3, 'True Grad AFT returns wrong number of outputs')
        
        # Should be exact since the calculation is the same, just 
        # returning different outputs
        # self.assertEqual(np.linalg.norm(res_default[0] - res_no_grad[0]), 0.0,
        #                  'No grad option on AFT is returning wrong force.')  
        
        self.assertNotEqual(np.linalg.norm(res_default[0]), 0.0,
                          'Bad test of nonlinear force, is all zeros.')   
        

        
            
if __name__ == '__main__':
    unittest.main()
