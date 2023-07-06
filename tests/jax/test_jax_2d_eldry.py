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

# vectorized (non JAX version)
from tmdsimpy.jax.nlforces.jenkins_element import JenkinsForce 

# JAX version for elastic dry friction
from tmdsimpy.jax.nlforces.elastic_dry_fric_2d import ElasticDryFriction2D 

import tmdsimpy.harmonic_utils as hutils


sys.path.append('..')
import verification_utils as vutils


###############################################################################
###     Testing Class                                                       ###
###############################################################################

def run_comparison(obj, Unl, w, h, Nt, force_tol, df_tol, jenkins, 
                   delta_grad=1e-5):
       
        FnlH_vec, dFnldUH_vec, dFnldw_vec \
            = jenkins.aft(Unl[::2, :], w, h, Nt=Nt)
        
        FnlH, dFnldUH, dFnldw = obj.eldry_force.aft(Unl, w, h, Nt=Nt)
        
        kn = obj.eldry_force.kn
                
        FH_error = np.max(np.abs(FnlH[::2]-FnlH_vec))
        
        if Unl[3::2].sum() == 0:
            FH_error = FH_error + (FnlH[1] - max(kn*Unl[1], 0.0)) \
                            + np.max(np.abs(FnlH[3::2]))
        
        obj.assertLess(FH_error, force_tol, 
                        'Incorrect elastic dry friction force.')
        
        ###############
        # Tangent - Tangent Gradient
        dFH_error = np.max(np.abs(dFnldUH[::2, ::2]-dFnldUH_vec))
        
        obj.assertLess(dFH_error, df_tol, 
                        'Incorrect Tangential AFT gradient.')
        
        
        ###############
        # Normal - Normal Gradient
        Un_grad = dFnldUH[1::2, 1::2]
        Un_grad = Un_grad - np.eye(Un_grad.shape[0])*kn
        
        dFH_error = np.max(np.abs(Un_grad))
        
        obj.assertLess(dFH_error, df_tol, 
                    'Incorrect Normal U/Normal F AFT gradient.')
        
        ###############
        # dNormal / dTangent Gradient
        
        dFndUt = dFnldUH[1::2, 0::2]
        
        obj.assertLess(np.max(np.abs(dFndUt)), df_tol, 
                    'Incorrect dFn/dUtan AFT gradient.')
        
        ###############
        # Numeric Gradient check, should capture dTangent/dNormal terms
        
        # Check gradient - Unl
        fun = lambda U : obj.eldry_force.aft(U, w, h, Nt=Nt)[0:2]
        
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, 
                                        atol=obj.atol_grad, h=delta_grad)
        
        obj.assertFalse(grad_failed, 'Incorrect Gradient w.r.t. Unl.')
        
        # Gradient w
        fun = lambda w : obj.eldry_force.aft(Unl, w, h, Nt=Nt)[0::2]
        
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
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        T = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        # Mapping for Jenkings
        Q_jenk = np.array([[1.0]])
        T_jenk = np.array([[1.0]])
        
        kt = 2.0
        kn = 2.5
        mu = 0.75
        
        self.un_low = 1.5
        self.un_high = 5.7
        
        Fs_low = self.un_low * kn * mu
        Fs_high = self.un_high * kn * mu
        
        self.jenkins_force_low = JenkinsForce(Q_jenk, T_jenk, kt, Fs_low, 
                                              u0=np.array([0.0]))
        
        self.jenkins_force_high = JenkinsForce(Q_jenk, T_jenk, kt, Fs_high, 
                                               u0=np.array([0.0]))
        
        self.eldry_force = ElasticDryFriction2D(Q, T, kt, kn, mu, 
                                                u0=np.array([0.0]))
        
        
        # Create Two eldry Force options with different u0 and verify in the 
        # stuck regime that they give the correct forces for harmonic 0
        self.eldry_force2 = ElasticDryFriction2D(Q, T, kt, kn, mu, u0=np.array([0.0]))
        self.eldry_force3 = ElasticDryFriction2D(Q, T, kt, kn, mu, u0=np.array([0.2]))
        
        
        # 1.5 Times Eldry
        Q_15 = np.array([[1.0, 0.0], 
                         [0.0, 1.0],
                         [1.0, 0.0], 
                         [0.0, 1.0]])
        
        T_15 = np.array([[1.0, 0.0, 0.5, 0.0], 
                         [0.0, 1.0, 0.0, 0.5]])
        
        self.eldry_force15 = ElasticDryFriction2D(Q_15, T_15, kt, kn, mu, u0=0.0)
        
        # Split Eldry
        Qsplit = np.eye(4)
        Tsplit = np.diag([0.5, 0.5, 1.5, 1.5])
        
        self.eldry_force_split = ElasticDryFriction2D(Qsplit, Tsplit, kt, kn, mu, u0=0.0)

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
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.75, 2.0, 1.3, 0.0, 0.0, 1.0, 0.0],
                          [self.un_low, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*5
        Unl = Unl.reshape(-1,1)
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.jenkins_force_low)
        
        Unl[1] = self.un_high
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.jenkins_force_high)
        
        
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
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0],
                          [self.un_low, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*0.2
        Unl = Unl.reshape(-1,1)
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.jenkins_force_low)
        
        
        Unl[1] = self.un_high
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.jenkins_force_high)

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
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0],
                          [self.un_low, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*10000000000.0
        Unl = Unl.reshape(-1,1)
        
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.jenkins_force_low)
        
        
        Unl[1] = self.un_high
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.jenkins_force_high)
        
        # Verify that 2nd harmonic of normal would induce same in tangent
        # [t0, n0, t1c, n1c, t1s, n1s, t2c, n2c, ]
        FnlH, dFnldUH, dFnldw = self.eldry_force.aft(Unl, w, h, Nt=Nt)
        
        self.assertGreater(np.abs(dFnldUH[6,7]), 0.1, 
                           'Normal load may not be influencing tangent force.')
        

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
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0],
                          [self.un_low, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*5
        Unl = Unl.reshape(-1,1)
        
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.jenkins_force_low, delta_grad=3e-6)
        

        
        Unl[1] = self.un_high
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.jenkins_force_high)
        
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
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[4.29653115, 4.29165565, 2.8307871 , 4.17186848, 3.37441948,\
                           0.80543152, 3.55638299],
                          [self.un_low, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*5
        Unl = Unl.reshape(-1,1)
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.jenkins_force_low)
        
        
        Unl[1] = self.un_high
        
        run_comparison(self, Unl, w, h, Nt, force_tol, df_tol, 
                       self.jenkins_force_high)
        
        # Verify that 2nd harmonic of normal would induce same in tangent
        # [t0, n0, t1c, n1c, t1s, n1s, t2c, n2c, ]
        FnlH, dFnldUH, dFnldw = self.eldry_force.aft(Unl, w, h, Nt=Nt)
        
        self.assertGreater(np.abs(dFnldUH[6,7]), 0.01, 
                           'Normal load may not be influencing tangent force.')
        
    def test_h0_force_opts(self):
        """
        Test moderate amplitude case

        Returns
        -------
        None.

        """
        
        h = np.array([0, 1, 2, 3])
        Unl = np.zeros((7*2,1))
        Unl[0] = 0.15
        Unl[1] = 100.0
        
        w = 1.75
        Nt = 1 << 7
                
        # Force Values for harmonic zero
        
        #################
        # U0 < Unl[0]
        FnlH2, dFnldUH, dFnldw = self.eldry_force2.aft(Unl, w, h, Nt=Nt)

        error2 = FnlH2[0] - (Unl[0] - self.eldry_force2.u0)*self.eldry_force2.kt
        
        self.assertLess(error2, 1e-18, 
                        'Static force from prestressed state is incorrect.')
        
        
        # Check gradient
        fun = lambda U : self.eldry_force2.aft(U, w, h)[0:2]
        
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, 
                                        atol=self.atol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient from u0 setting.')
        
        #################
        # U0 > Unl[0]
        FnlH3, dFnldUH, dFnldw = self.eldry_force3.aft(Unl, w, h, Nt=Nt)
        
        
        error3 = FnlH3[0] - (Unl[0] - self.eldry_force3.u0)*self.eldry_force3.kt
        
        self.assertLess(error3, 1e-18, 
                        'Static force from prestressed state is incorrect.')
        
        
        # Check gradient
        fun = lambda U : self.eldry_force3.aft(U, w, h)[0:2]
        
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
        Nt = 1 << 10
        w = 1.7
        h = np.array([0, 1, 2, 3])
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = np.array([[4.29653115, 4.29165565, 2.8307871 , 4.17186848, 3.37441948,\
                           0.80543152, 3.55638299],
                          [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        
        Unl[:, 0] = Unl[:, 0]*5
        Unl = Unl.reshape(-1,1)
        
        
        FnlH, dFnldUH, dFnldw = self.eldry_force.aft(Unl, w, h, Nt=Nt)
        
        self.assertEqual(FnlH.sum(), 0, 
                         'Contact forces should be zero when out of contact')
        
        self.assertEqual(dFnldUH.sum(), 0, 
                         'Contact gradient should be zero when out of contact')
        
        # Verify at zero in contact that there are not NaN's in gradient etc.
        Unl[1] = 0
        FnlH, dFnldUH, dFnldw = self.eldry_force.aft(Unl, w, h, Nt=Nt)

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
        
        compare_tol = 1e-16
        
        # Friction Versions
        eldry_sdof = self.eldry_force
        eldry_times_1p5 = self.eldry_force15
        eldry_split = self.eldry_force_split
        
        Tsplit = eldry_split.T
        
        # Misc Parameters to use
        Nt = 1 << 10
        w = 1.7
        h = np.array([0, 1, 2, 3])
        Nhc = hutils.Nhc(h)
        
        # Generate some baseline vectors for the first and second DOFs
        np.random.seed(1023)
        U1 = np.random.rand(Nhc*2)
        U2 = np.random.rand(Nhc*2)
        # All random values are positive, so will be in contact
        
        # Allow tangential displacements to be positive and negative
        U1[::2] = U1[::2] - 0.5
        U2[::2] = U2[::2] - 0.5
        
        # Scale the tangential displacements by different values
        scale_tan = [0.1, 1.0, 10.0, 100.0]

        for scale in scale_tan:
            
            # Baseline from U1
            U1scale = np.copy(U1)
            U1scale[::2] = scale*U1scale[::2]
            
            F1 = eldry_sdof.aft(U1scale, w, h, Nt=Nt)[0]
            
            # Baseline from U2
            U2scale = np.copy(U2)
            U2scale[::2] = scale*U2scale[::2]
            
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
            Ucombo = np.zeros(Nhc*4)
            Ucombo[::4] = U1[::2]*scale # Tanget
            Ucombo[1::4] = U1[1::2] # Normal
            Ucombo[2::4] = U2[::2]*scale # Tangent 
            Ucombo[3::4] = U2[1::2] # Normal
            
            Fcombo = eldry_split.aft(Ucombo, w, h, Nt=Nt)[0]
            
            Fcombo_ref = np.zeros_like(Fcombo)
            Fcombo_ref[::4]  = Tsplit[0,0]*F1[::2] # Tanget
            Fcombo_ref[1::4] = Tsplit[1,1]*F1[1::2] # Normal
            Fcombo_ref[2::4] = Tsplit[2,2]*F2[::2] # Tangent 
            Fcombo_ref[3::4] = Tsplit[3,3]*F2[1::2] # Normal
            
            self.assertLess(np.abs(Fcombo - Fcombo_ref).max(), compare_tol, 
                            'Using 2 DOFs to scale force failed.')
            
            
            ############ Gradient
            fun = lambda U : eldry_split.aft(U, w, h)[0:2]
            
            grad_failed = vutils.check_grad(fun, Ucombo, verbose=False, 
                                            atol=self.atol_grad)
            
            self.assertFalse(grad_failed, 
                             'Incorrect Gradient for combined two force test.')
        

if __name__ == '__main__':
    unittest.main()