"""

Script for testing the accuracy of the Jenkins implementation that uses JAX
- Compares AFT results to those from vector Jenkins

A separate example file shows timing comparisons for the two implementations. 

Compilation v. execution time cannot easily be tested with unittest since
the order of the tests is not fixed by writing them therefore JIT may have 
already compiled the function for a different test.

"""

# Standard imports
import numpy as np
import sys
import unittest

# Python Utilities
sys.path.append('../..')

# vectorized (non JAX version)
from tmdsimpy.nlforces.vector_jenkins import VectorJenkins

# JAX version
from tmdsimpy.jax.nlforces.jenkins_element import JenkinsForce 


sys.path.append('..')
import verification_utils as vutils


###############################################################################
###     Testing Class                                                       ###
###############################################################################

class TestJAXJenkins(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestJAXJenkins, self).__init__(*args, **kwargs)      
        
        
        force_tol = 5e-15 # All should be exactly equal
        df_tol = 1e-14 # rounding error on the derivatives
        dfdw_tol = 1e-16 # everything should have zero derivative w.r.t. frequency
        
        self.tols = (force_tol, df_tol, dfdw_tol)


        # Simple Mapping to displacements
        Q = np.array([[1.0]])
        T = np.array([[1.0]])
        
        kt = 2.0
        Fs = 3.0
        
        self.vector_jenkins_force = VectorJenkins(Q, T, kt, Fs)
        self.jenkins_force = JenkinsForce(Q, T, kt, Fs, u0=None)
        
        
        # Create Two Jenkins Force options with different u0 and verify in the 
        # stuck regime that they give the correct forces for harmonic 0
        self.jenkins_force2 = JenkinsForce(Q, T, kt, Fs, u0=np.array([0.0]))
        self.jenkins_force3 = JenkinsForce(Q, T, kt, Fs, u0=np.array([0.2]))
        self.atol_grad = 1e-12


    def test_vec_jenk1(self):
        """
        Test some basic / straightforward motion comparisons 
        (highish amplitude)

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        jenkins_force = self.jenkins_force
        vector_jenkins_force = self.vector_jenkins_force
        
        
        ###### Test Parameters
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = 5*np.array([[0.75, 2.0, 1.3, 0.0, 0.0, 1.0, 0.0]]).T
        # unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
        
        
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        FH_error = np.max(np.abs(FnlH-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Jenkins AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Jenkins AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Jenkins AFT gradient w.r.t. freq.')
        
        
    def test_vec_jenk2(self):
        """
        Test low amplitude motion

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        jenkins_force = self.jenkins_force
        vector_jenkins_force = self.vector_jenkins_force
        
        
        ###### Test Parameters
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = 0.2*np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0]]).T
        
        
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        FH_error = np.max(np.abs(FnlH-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Jenkins AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Jenkins AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Jenkins AFT gradient w.r.t. freq.')
        

    def test_vec_jenk3(self):
        """
        Test very high amplitude motion

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        jenkins_force = self.jenkins_force
        vector_jenkins_force = self.vector_jenkins_force
        
        
        ###### Test Parameters
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = 10000000000.0*np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0]]).T
        
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)
        
       
        FH_error = np.max(np.abs(FnlH-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Jenkins AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Jenkins AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Jenkins AFT gradient w.r.t. freq.')
        

    def test_vec_jenk4(self):
        """
        Test moderate amplitude case

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        jenkins_force = self.jenkins_force
        vector_jenkins_force = self.vector_jenkins_force
        
        
        ###### Test Parameters
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = 5*np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0]]).T
        
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        
        FH_error = np.max(np.abs(FnlH-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Jenkins AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Jenkins AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Jenkins AFT gradient w.r.t. freq.')
        


    def test_vec_jenk5(self):
        """
        Test moderate amplitude case

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        jenkins_force = self.jenkins_force
        vector_jenkins_force = self.vector_jenkins_force
        
        
        ###### Test Parameters
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = 5*np.array([[4.29653115, 4.29165565, 2.8307871 , 4.17186848, 3.37441948,\
                           0.80543152, 3.55638299]]).T
        
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        
        FH_error = np.max(np.abs(FnlH-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Jenkins AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Jenkins AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Jenkins AFT gradient w.r.t. freq.')
        


    def test_h0_force_opts(self):
        """
        Test moderate amplitude case

        Returns
        -------
        None.

        """
        
        h = np.array([0, 1, 2, 3])
        Unl = np.zeros((7,1))
        Unl[0] = 0.15
        
        w = 1.75
        Nt = 1 << 7
        
        
        # Force Values for harmonic zero
        
        #################
        # U0 < Unl[0]
        FnlH2, dFnldUH, dFnldw = self.jenkins_force2.aft(Unl, w, h, Nt=Nt)

        error2 = FnlH2[0] - (Unl[0] - self.jenkins_force2.u0)*self.jenkins_force2.kt
        
        self.assertLess(error2, 1e-18, 
                        'Static force from prestressed state is incorrect.')
        
        
        # Check gradient
        fun = lambda U : self.jenkins_force2.aft(U, w, h)[0:2]
        
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, 
                                        atol=self.atol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient from u0 setting.')
        
        #################
        # U0 > Unl[0]
        FnlH3, dFnldUH, dFnldw = self.jenkins_force3.aft(Unl, w, h, Nt=Nt)
        
        
        error3 = FnlH3[0] - (Unl[0] - self.jenkins_force3.u0)*self.jenkins_force3.kt
        
        self.assertLess(error3, 1e-18, 
                        'Static force from prestressed state is incorrect.')
        
        
        # Check gradient
        fun = lambda U : self.jenkins_force3.aft(U, w, h)[0:2]
        
        grad_failed = vutils.check_grad(fun, Unl, verbose=False, 
                                        atol=self.atol_grad)
        
        self.assertFalse(grad_failed, 'Incorrect Gradient from u0 setting.')
        # Gradients are correct?
        

if __name__ == '__main__':
    unittest.main()
