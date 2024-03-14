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
        
    def test_harmonic_stiffness_opts(self):
        """
        Test options on harmonic stiffness to verify that they still give 
        correct results.
        """
        
        h_sets = [np.array([0, 1, 2, 3, 6]), 
                  np.array([1, 2, 3, 5, 7, 9])]
        
        w = 1.393
        
        E_rtol = 1e-12
        dEdw_rtol = 1e-12
        
        for h in h_sets:
            
            for Ndof in [1, 15]:
            
                rng = np.random.default_rng(seed=1023)
                
                M = rng.random((Ndof, Ndof))
                C = rng.random((Ndof, Ndof))
                K = rng.random((Ndof, Ndof))
                
                ###############
                # Verify the option to not calculate the gradient + verify default
                ref = hutils.harmonic_stiffness(M, C, K, w, h)
                
                yes_grad = hutils.harmonic_stiffness(M, C, K, w, h, 
                                                    calc_grad=True)
                
                no_grad = hutils.harmonic_stiffness(M, C, K, w, h, 
                                                    calc_grad=False)
                
                self.assertEqual(len(ref), len(yes_grad), 
                         'Default does not return correct number of arguments')
                
                self.assertEqual(len(no_grad), 1)
                
                self.assertLess(np.linalg.norm(yes_grad[0] - ref[0]) \
                                / np.linalg.norm(ref[0]), 
                                E_rtol)
                
                self.assertLess(np.linalg.norm(yes_grad[1] - ref[1]) \
                                / np.linalg.norm(ref[1]), 
                                E_rtol)
                    
                self.assertLess(np.linalg.norm(no_grad[0] - ref[0]) \
                                / np.linalg.norm(ref[0]), 
                                dEdw_rtol)
                    
                ###############
                # Verify the EPMC option of only calculating effects of C
                ref_c = hutils.harmonic_stiffness(0.0*M, C, 0.0*K, w, h)
                
                all_mats = hutils.harmonic_stiffness(0.0*M, C, 0.0*K, w, h, 
                                                    only_C=False)
                
                # Pass arbitrary arguments for M and K to verify that they
                # don't influence anything
                # EPMC also does not need the frequency gradient of this
                only_c = hutils.harmonic_stiffness(1.5, C, 
                                                   np.array([1.0, 2.0]), 
                                                   w, h, only_C=True,
                                                   calc_grad=False)
                
                self.assertLess(np.linalg.norm(all_mats[0] - ref_c[0]) \
                                / np.linalg.norm(ref_c[0]), 
                                E_rtol)
                
                self.assertLess(np.linalg.norm(all_mats[1] - ref_c[1]) \
                                / np.linalg.norm(ref_c[1]), 
                                E_rtol)
                    
                self.assertLess(np.linalg.norm(only_c[0] - ref_c[0]) \
                                / np.linalg.norm(ref_c[0]), 
                                dEdw_rtol)
                    
    def test_harmonic_conditioning(self):
        """
        Verify that harmonic conditioning function behaves as expected.
        """
        
        h = np.array([0, 1, 3])
        
        Ndof = 4
        
        # Make something that looks like EPMC Solution
        Uwxa = np.array([0.01, 0.02, 0.1,   0.0001, # harmonic 0
                         1.0,   5.0, 3.0,   2.0, # harmonic 1 cos
                         0.02,  0.05, 0.03, 0.02, # Harmonic 1 sine
                         0.1,   0.2,  0.3,  0.2, # harmonic 3 cos
                         0.0001, 0.0002, 0.00003, 0.00001, # harmonic 3 sine
                         1000.0, 1e-6, -4]) # w, x, a
        
        
        scalar_delta = 1e-4
        CtoP_expected_scalar_delta = \
            np.array([3.252500e-02, 3.252500e-02, 3.252500e-02, 3.252500e-02,
                   1.390000e+00, 1.390000e+00, 1.390000e+00, 1.390000e+00,
                   1.390000e+00, 1.390000e+00, 1.390000e+00, 1.390000e+00,
                   1.000425e-01, 1.000425e-01, 1.000425e-01, 1.000425e-01,
                   1.000425e-01, 1.000425e-01, 1.000425e-01, 1.000425e-01,
                   1.000000e+03, 1.000000e-04, 4.000000e+00])
            
        vec_delta = [1e-3, 2.0, 0.5, 0.0]
        
        CtoP_expected_vec_delta = \
            np.array([3.2525e-02, 3.2525e-02, 3.2525e-02, 3.2525e-02, 2.0000e+00,
                   2.0000e+00, 2.0000e+00, 2.0000e+00, 2.0000e+00, 2.0000e+00,
                   2.0000e+00, 2.0000e+00, 5.0000e-01, 5.0000e-01, 5.0000e-01,
                   5.0000e-01, 5.0000e-01, 5.0000e-01, 5.0000e-01, 5.0000e-01,
                   1.0000e+03, 1.0000e-06, 4.0000e+00])
        
        ##############
        # Baseline case where delta is constant
        CtoP = hutils.harmonic_wise_conditioning(Uwxa, Ndof, h, delta=scalar_delta)
        
        self.assertLess(np.max(np.abs(CtoP_expected_scalar_delta - CtoP) \
                               / CtoP_expected_scalar_delta), 
                        1e-8)
        
        ##############
        # Second Case where delta is a vector
        CtoP = hutils.harmonic_wise_conditioning(Uwxa, Ndof, h, delta=vec_delta)
        
        self.assertLess(np.max(np.abs(CtoP_expected_vec_delta - CtoP) \
                               / CtoP_expected_vec_delta), 
                        1e-8)
            
        ##############
        # Verify No Error for No Extra Terms
        CtoP = hutils.harmonic_wise_conditioning(Uwxa[:-3], Ndof, h, delta=vec_delta[:-1])
        
        self.assertLess(np.max(np.abs(CtoP_expected_vec_delta[:-3] - CtoP) \
                               / CtoP_expected_vec_delta[:-3]), 
                        1e-8)
        

if __name__ == '__main__':
    unittest.main()