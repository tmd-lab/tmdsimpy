"""

Verification of vectorized Jenkins algorithm

This script compares outputs to the non-vectorized Jenkins algorithm for 
verification. Thus gradients do not need to be numerically approximated here.

An example script illustrates the computational speedup for the vectorized 
algorithm and is not included here. Convergence of Jenkins is also included in 
the other example. 

"""

import sys
import numpy as np
import unittest

# Python Utilities
sys.path.append('../..')
import tmdsimpy.harmonic_utils as hutils
from tmdsimpy.nlforces.jenkins_element import JenkinsForce
from tmdsimpy.nlforces.vector_jenkins import VectorJenkins



###############################################################################
###     Testing Function                                                    ###
###############################################################################

# Useful functions for testing: 
def time_series_forces(Unl, h, Nt, w, jenkins_force):
    
    # Unl = np.reshape(Unl, ((-1,1)))

    # Nonlinear displacements, velocities in time
    unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
    unltdot = w*hutils.time_series_deriv(Nt, h, Unl, 1) # Nt x Ndnl
    
    Nhc = hutils.Nhc(h)
    cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
    
    unlth0 = Unl[0]
    
    fnl, dfduh, dfdudh = jenkins_force.local_force_history(unlt, unltdot, h, cst, unlth0)
    
    fnl = np.einsum('ij -> i', fnl)
    dfduh = np.einsum('ijkl -> il', dfduh)
    
    return fnl, dfduh

###############################################################################
###     Testing Class                                                       ###
###############################################################################

class TestVectorJenkins(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestVectorJenkins, self).__init__(*args, **kwargs)      
        
        
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
        self.jenkins_force = JenkinsForce(Q, T, kt, Fs)
        
        # vector_jenkins_force.init_history(unlth0=0)


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
        
        
        fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
        FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        force_error = np.max(np.abs(fnl-fnl_vec))
        df_error = np.max(np.abs(dfduh-dfduh_vec))
        
        self.assertLess(force_error, force_tol, 
                        'Incorrect vectorized Jenkins timeseries forces.')
        
        self.assertLess(df_error, df_tol, 
                        'Incorrect vectorized Jenkins timeseries gradient.')
        
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
        
        
        fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
        FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        force_error = np.max(np.abs(fnl-fnl_vec))
        df_error = np.max(np.abs(dfduh-dfduh_vec))
        
        self.assertLess(force_error, force_tol, 
                        'Incorrect vectorized Jenkins timeseries forces.')
        
        self.assertLess(df_error, df_tol, 
                        'Incorrect vectorized Jenkins timeseries gradient.')
        
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
        
        fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
        FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        force_error = np.max(np.abs(fnl-fnl_vec))
        df_error = np.max(np.abs(dfduh-dfduh_vec))
        
        self.assertLess(force_error, force_tol, 
                        'Incorrect vectorized Jenkins timeseries forces.')
        
        self.assertLess(df_error, df_tol, 
                        'Incorrect vectorized Jenkins timeseries gradient.')
        
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
        
        fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
        FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        force_error = np.max(np.abs(fnl-fnl_vec))
        df_error = np.max(np.abs(dfduh-dfduh_vec))
        
        self.assertLess(force_error, force_tol, 
                        'Incorrect vectorized Jenkins timeseries forces.')
        
        self.assertLess(df_error, df_tol, 
                        'Incorrect vectorized Jenkins timeseries gradient.')
        
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
        
        fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_jenkins_force)
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, jenkins_force)
        FnlH, dFnldUH, dFnldw = jenkins_force.aft(Unl, w, h, Nt=Nt)
        
        force_error = np.max(np.abs(fnl-fnl_vec))
        df_error = np.max(np.abs(dfduh-dfduh_vec))
        
        self.assertLess(force_error, force_tol, 
                        'Incorrect vectorized Jenkins timeseries forces.')
        
        self.assertLess(df_error, df_tol, 
                        'Incorrect vectorized Jenkins timeseries gradient.')
        
        FH_error = np.max(np.abs(FnlH-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Jenkins AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Jenkins AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Jenkins AFT gradient w.r.t. freq.')
        






if __name__ == '__main__':
    unittest.main()