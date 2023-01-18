"""
Script for verifying the accuracy of the harmonic utilities
    
failed_flag = False, changes to true if a test fails at any point 

Notes:
    1. It would be better to have all the tolerances defined somewhere together
    rather than the current check of having them wherever they are used.
""" 

import sys
import numpy as np

# Python Utilities
sys.path.append('../ROUTINES/')
import harmonic_utils as hutils
import verification_utils as vutils

# MATLAB Utilities
import os
wdir = os.getcwd()
import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd(wdir + '/MATLAB_VERSIONS/')


def verify_hutils(nd, h, X0, tol=1e-12):
    
    test_failed = False
    
    Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
    
    M = np.random.rand(nd, nd)
    C = np.random.rand(nd, nd)
    K = np.random.rand(nd, nd)
    
    w = np.random.rand()
    
    Nt = 1 << 7
    
    # MATLAB Matrices
    M_mat = matlab.double(M.tolist())
    C_mat = matlab.double(C.tolist())
    K_mat = matlab.double(K.tolist())
    
    h_mat = matlab.double(h.tolist())
    
    X0_mat = matlab.double(X0.tolist())
    
    
    # Verify with zero harmonic
    print('Verifying Harmonic Stiffness:')
    E_mat, dEdw_mat = eng.HARMONICSTIFFNESS(M_mat, C_mat, K_mat, w, h_mat, nargout=2)
    E, dEdw = hutils.harmonic_stiffness(M, C, K, w, h)
    
    error = vutils.compare_mats(E, E_mat)
    test_failed = test_failed or error > tol
    
    error = vutils.compare_mats(dEdw, dEdw_mat)
    test_failed = test_failed or error > tol
    
    print('Verifying GETFOURIERCOEFF / GETFOURIERCOEFF, looping over derivative order:')
    
    for order in range(0, 4):
        x_t_mat = eng.TIMESERIES_DERIV(Nt*1.0, h_mat, X0_mat, order*1.0, nargout=1)
        x_t = hutils.time_series_deriv(Nt, h, X0, order)
        
        error = vutils.compare_mats(x_t, x_t_mat)
        test_failed = test_failed or error > tol
        
        v_mat = eng.GETFOURIERCOEFF(h_mat, x_t_mat, nargout=1)
        v = hutils.get_fourier_coeff(h, x_t)
        
        error = vutils.compare_mats(v, v_mat)
        test_failed = test_failed or error > tol
        
    return test_failed

# Flag for checking all tests at end.
failed_flag = False


# Verify without zero harmonic

# Verified for these three random seeds:

# np.random.seed(42)
np.random.seed(1023)
# np.random.seed(0)

print('With Zero Harmonic:')
nd = 5
h = np.array([0, 1, 2, 3, 6])
Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
X0 = np.random.rand(Nhc, nd)
test_failed = verify_hutils(nd, h, X0)

failed_flag = failed_flag or test_failed


print('Without Zero Harmonic:')
nd = 7
h = np.array([1, 2, 3, 5, 7, 9])
Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
# X0 = np.zeros((Nhc, nd))
# X0[0,0] = 0.5
# X0[1,1] = 1
# X0[2,2] = 0.75
# X0[3,3] = 0.3
X0 = np.random.rand(Nhc, nd)


test_failed = verify_hutils(nd, h, X0)

failed_flag = failed_flag or test_failed


if failed_flag:
    print('\n\nTest FAILED, investigate results further!\n')
else:
    print('\n\nTest passed.\n')
    