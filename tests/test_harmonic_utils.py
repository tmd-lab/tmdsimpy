"""
Script for testing the accuracy of the harmonic utilities

To generate the MATLAB files used in this test, run the script 
'MATLAB_VERSIONS/generate_reference.m' in MATLAB. *.mat files are committed to 
the repo, so that is not generally necessary.
    
Notes:
    1. It would be better to have all the tolerances defined somewhere together
    rather than the current check of having them wherever they are used.
""" 

import sys
import numpy as np
import unittest

# Python Utilities
sys.path.append('..')
import tmdsimpy.harmonic_utils as hutils

# to import MATLAB files to compare to old versions
from scipy import io as sio


def verify_hutils(fname, test_obj, tol=1e-12):
    """
    Function that can be repeatedly called to run tests of the harmonic 
    utilities
    
    Values are verified against older MATLAB routines.

    Parameters
    ----------
    nd : Number of degrees of freedom
    h : List of harmonics
    X0 : Value of frequency domain components for the test
    test_obj : unittest class object that is being used and can raise 
                exceptions if the test fails.
    tol : Tolerance for test errors. The default is 1e-12.

    Returns
    -------
    None

    """
    
    
    mat_sol = sio.loadmat(fname)
    M = mat_sol['M']
    C = mat_sol['C']
    K = mat_sol['K']
    
    h = mat_sol['h'].reshape(-1)
    # nd = mat_sol['nd'][0, 0]
    X0 = mat_sol['X0']
    
    w = mat_sol['w'][0, 0]
    
    Nt = mat_sol['x_t0'].shape[0]
    
    ####### Compare Values
    
    E, dEdw = hutils.harmonic_stiffness(M, C, K, w, h)
    
    test_obj.assertLess(np.linalg.norm(E - mat_sol['E']), tol, 
                        'Harmonic stiffness is incorrect.')
    
    test_obj.assertLess(np.linalg.norm(dEdw - mat_sol['dEdw']), tol, 
                        'Harmonic stiffness freq. gradient is incorrect.')
    
    # Verifying GETFOURIERCOEFF / GETFOURIERCOEFF, 
    # looping over derivative order
    for order in range(0, 4):
        
        x_t = hutils.time_series_deriv(Nt, h, X0, order)
        
        error = np.max(np.abs(x_t - mat_sol['x_t'+str(order)]))
        
        test_obj.assertLess(error, tol, 
                'Time series for derivative order {} are incorrect.'.format(order))
        
        v = hutils.get_fourier_coeff(h, x_t)
        
        error = np.max(np.abs(v - mat_sol['v'+str(order)]))
        
        test_obj.assertLess(error, tol, 
                'Fourier coefficients for derivative order {} are incorrect.'.format(order))
        
    return

class TestHarmonicUtils(unittest.TestCase):
    
    def test_with_h0(self):
        """
        Test harmonic utils without harmonic 0
        
        # These should be the parameters saved in this .mat file:
        nd = 5 # DOFs
        h = np.array([0, 1, 2, 3, 6]) # Harmonics
        """
        
        fname = 'MATLAB_VERSIONS/hutils_with_h0.mat'
        
        verify_hutils(fname, self)
        
    def test_without_h0(self):
        """
        Test harmonic utils without harmonic 0
        
        # These should be the parameters saved in this .mat file:
        
        nd = 7 # DOFs
        h = np.array([1, 2, 3, 5, 7, 9]) # Harmonics
        """

        fname = 'MATLAB_VERSIONS/hutils_without_h0.mat'
        
        verify_hutils(fname, self)

if __name__ == '__main__':
    unittest.main()