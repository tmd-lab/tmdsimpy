"""

Verification of vectorized Iwan 4 algorithm

This script compares outputs to the non-vectorized Iwan 4 algorithm for 
verification. Thus gradients do not need to be numerically approximated here.

An example script illustrates the computational speedup for the vectorized 
algorithm and is not included here. 

"""

import sys
import numpy as np
import unittest

# Python Utilities
sys.path.append('../../ROUTINES/')
sys.path.append('../../ROUTINES/NL_FORCES')
import harmonic_utils as hutils

from iwan4_element import Iwan4Force 
from vector_iwan4 import VectorIwan4



###############################################################################
###     Testing Function                                                    ###
###############################################################################

# Useful functions for testing: 
def time_series_forces(Unl, h, Nt, w, nl_force):
    
    # Unl = np.reshape(Unl, ((-1,1)))

    # Nonlinear displacements, velocities in time
    unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
    unltdot = w*hutils.time_series_deriv(Nt, h, Unl, 1) # Nt x Ndnl
    
    Nhc = hutils.Nhc(h)
    cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
    
    unlth0 = Unl[0]
    
    fnl, dfduh, dfdudh = nl_force.local_force_history(unlt, unltdot, h, cst, unlth0)
    
    fnl = np.einsum('ij -> i', fnl)
    dfduh = np.einsum('ijkl -> il', dfduh)
    
    return fnl, dfduh

###############################################################################
###     Testing Class                                                       ###
###############################################################################

class TestVectorIwan4(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestVectorIwan4, self).__init__(*args, **kwargs)      
        
        
        force_tol = 5e-15 # All should be exactly equal
        df_tol = 1e-14 # rounding error on the derivatives
        dfdw_tol = 1e-16 # everything should have zero derivative w.r.t. frequency
        
        self.tols = (force_tol, df_tol, dfdw_tol)


        # Simple Mapping to displacements
        Q = np.array([[1.0]])
        T = np.array([[1.0]])
        
        
        kt = 2.0
        Fs = 3.0
        chi = -0.1
        beta = 0.1
        
        Nsliders = 100
        
        self.iwan_force   = Iwan4Force(Q, T, kt, Fs, chi, beta, Nsliders=Nsliders, alphasliders=1.0)
        self.vector_force = VectorIwan4(Q, T, kt, Fs, chi, beta, Nsliders=Nsliders, alphasliders=1.0)


    def test_vec_iwan1(self):
        """
        Test some basic / straightforward motion comparisons 
        (highish amplitude)

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        iwan_force = self.iwan_force
        vector_force = self.vector_force
        
        
        ###### Test Parameters
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = 5*np.array([[0.75, 2.0, 1.3, 0.0, 0.0, 1.0, 0.0]]).T
        # unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
        
        
        fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_force)
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_force.aft(Unl, w, h, Nt=Nt)
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, iwan_force)
        FnlH, dFnldUH, dFnldw = iwan_force.aft(Unl, w, h, Nt=Nt)
        
        force_error = np.max(np.abs(fnl-fnl_vec))
        df_error = np.max(np.abs(dfduh-dfduh_vec))
        
        self.assertLess(force_error, force_tol, 
                        'Incorrect vectorized Iwan 4 timeseries forces.')
        
        self.assertLess(df_error, df_tol, 
                        'Incorrect vectorized Iwan 4 timeseries gradient.')
        
        FH_error = np.max(np.abs(FnlH_vec-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Iwan 4 AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Iwan 4 AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Iwan 4 AFT gradient w.r.t. freq.')
        
        
    def test_vec_iwan2(self):
        """
        Test low amplitude motion

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        iwan_force = self.iwan_force
        vector_force = self.vector_force
        
        
        ###### Test Parameters
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = 0.2*np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0]]).T
        
        
        fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_force)
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_force.aft(Unl, w, h, Nt=Nt)
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, iwan_force)
        FnlH, dFnldUH, dFnldw = iwan_force.aft(Unl, w, h, Nt=Nt)
        
        force_error = np.max(np.abs(fnl-fnl_vec))
        df_error = np.max(np.abs(dfduh-dfduh_vec))
        
        self.assertLess(force_error, force_tol, 
                        'Incorrect vectorized Iwan 4 timeseries forces.')
        
        self.assertLess(df_error, df_tol, 
                        'Incorrect vectorized Iwan 4 timeseries gradient.')
        
        FH_error = np.max(np.abs(FnlH_vec-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Iwan 4 AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Iwan 4 AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Iwan 4 AFT gradient w.r.t. freq.')
        

    def test_vec_iwan3(self):
        """
        Test very high amplitude motion

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        iwan_force = self.iwan_force
        vector_force = self.vector_force
        
        
        ###### Test Parameters
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = 10000000000.0*np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0]]).T
        
        fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_force)
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_force.aft(Unl, w, h, Nt=Nt)
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, iwan_force)
        FnlH, dFnldUH, dFnldw = iwan_force.aft(Unl, w, h, Nt=Nt)
        
        force_error = np.max(np.abs(fnl-fnl_vec))
        df_error = np.max(np.abs(dfduh-dfduh_vec))
        
        self.assertLess(force_error, force_tol, 
                        'Incorrect vectorized Iwan 4 timeseries forces.')
        
        self.assertLess(df_error, df_tol, 
                        'Incorrect vectorized Iwan 4 timeseries gradient.')
        
        FH_error = np.max(np.abs(FnlH_vec-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Iwan 4 AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Iwan 4 AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Iwan 4 AFT gradient w.r.t. freq.')
        

    def test_vec_iwan4(self):
        """
        Test moderate amplitude case

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        iwan_force = self.iwan_force
        vector_force = self.vector_force
        
        
        ###### Test Parameters
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = 5*np.array([[0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.0]]).T
        
        fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_force)
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_force.aft(Unl, w, h, Nt=Nt)
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, iwan_force)
        FnlH, dFnldUH, dFnldw = iwan_force.aft(Unl, w, h, Nt=Nt)
        
        force_error = np.max(np.abs(fnl-fnl_vec))
        df_error = np.max(np.abs(dfduh-dfduh_vec))
        
        self.assertLess(force_error, force_tol, 
                        'Incorrect vectorized Iwan 4 timeseries forces.')
        
        self.assertLess(df_error, df_tol, 
                        'Incorrect vectorized Iwan 4 timeseries gradient.')
        
        FH_error = np.max(np.abs(FnlH_vec-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Iwan 4 AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Iwan 4 AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Iwan 4 AFT gradient w.r.t. freq.')
        


    def test_vec_iwan5(self):
        """
        Test moderate amplitude case

        Returns
        -------
        None.

        """
        
        ##### Get models from class
        force_tol, df_tol, dfdw_tol = self.tols
        iwan_force = self.iwan_force
        vector_force = self.vector_force
        
        
        ###### Test Parameters
        Nt = 1 << 10
        w = 1.7
        
        ##### Verification
        
        h = np.array([0, 1, 2, 3])
        Unl = 5*np.array([[4.29653115, 4.29165565, 2.8307871 , 4.17186848, 3.37441948,\
                           0.80543152, 3.55638299]]).T
        
        fnl_vec, dfduh_vec = time_series_forces(Unl, h, Nt, w, vector_force)
        FnlH_vec, dFnldUH_vec, dFnldw_vec = vector_force.aft(Unl, w, h, Nt=Nt)
        
        fnl, dfduh = time_series_forces(Unl, h, Nt, w, iwan_force)
        FnlH, dFnldUH, dFnldw = iwan_force.aft(Unl, w, h, Nt=Nt)
        
        force_error = np.max(np.abs(fnl-fnl_vec))
        df_error = np.max(np.abs(dfduh-dfduh_vec))
        
        self.assertLess(force_error, force_tol, 
                        'Incorrect vectorized Iwan 4 timeseries forces.')
        
        self.assertLess(df_error, df_tol, 
                        'Incorrect vectorized Iwan 4 timeseries gradient.')
        
        FH_error = np.max(np.abs(FnlH_vec-FnlH_vec))
        dFH_error = np.max(np.abs(dFnldUH-dFnldUH_vec))
        
        self.assertLess(FH_error, force_tol, 
                        'Incorrect vectorized Iwan 4 AFT forces.')
        
        self.assertLess(dFH_error, df_tol, 
                        'Incorrect vectorized Iwan 4 AFT gradient.')
        
        
        dfdw_error = np.max(np.abs(dFnldw - dFnldw_vec))
        
        self.assertLess(dfdw_error, dfdw_tol, 
                        'Incorrect vectorized Iwan 4 AFT gradient w.r.t. freq.')
        






if __name__ == '__main__':
    unittest.main()