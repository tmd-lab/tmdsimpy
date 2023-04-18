"""
Script for testing the accuracy and time improvements of the harmonic utilities
that are implemented using the JAX library (allows for autodiff)

Performance tests are included to verify that the expected usage is not forcing
recompilation at every function evaluation. 
    
Notes:
    1. It would be better to have all the tolerances defined somewhere together
    rather than the current check of having them wherever they are used.
""" 

import sys
import numpy as np
import unittest
import timeit

import jax
jax.config.update("jax_enable_x64", True)

# Python Utilities
sys.path.append('../..')
import tmdsimpy.harmonic_utils as hutils
import tmdsimpy.jax.jax_harmonic_utils as jhutils


def verify_hutils(nd, h, X0, test_obj, tol=1e-11):
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
    
    Nt = 1 << 7
    
    # Compile the fourier coefficient function to verify that it doesn't break
    get_fourier = jax.jit(lambda x_t : jhutils.get_fourier_coeff(h, x_t))
    
    # Verifying GETFOURIERCOEFF / GETFOURIERCOEFF, 
    # looping over derivative order
    for order in range(0, 4):
        
        x_t_jax = jhutils.time_series_deriv(Nt, h, X0, order)
        x_t = hutils.time_series_deriv(Nt, h, X0, order)
        
        error = np.linalg.norm(x_t - x_t_jax)
        test_obj.assertLess(error, tol, 
                            'Time series for derivative order {} are incorrect.'.format(order))
        
        v_jax = get_fourier(x_t) # Compiled JAX version
        v = hutils.get_fourier_coeff(h, x_t)
        
        error = np.linalg.norm(v - v_jax)
        test_obj.assertLess(error, tol, 
                            'Fourier coefficients for derivative order {} are incorrect.'.format(order))
        
    return

def combined_derivs_jax(X0, h):
    """
    One function that includes both 0th and 1st order derivative to test
    if multiple compilations are required for repeat calls. 

    Parameters
    ----------
    X0 : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # print(X0.shape) # can be used to show when it is recompiling.
    
    Nt = 1 << 7
    
    x_t = jhutils.time_series_deriv(Nt, h, X0, 0)
    
    v_t = jhutils.time_series_deriv(Nt, h, X0, 1)
    
    return x_t + v_t
    
    
def combined_derivs(X0, h):
    """
    One function that includes both 0th and 1st order derivative to compare
    to the jax function with the normal functions

    Parameters
    ----------
    X0 : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # print(X0.shape) # can be used to show when it is recompiling.
    
    Nt = 1 << 7
    
    x_t = hutils.time_series_deriv(Nt, h, X0, 0)
    
    v_t = hutils.time_series_deriv(Nt, h, X0, 1)
    
    return x_t + v_t

class TestHarmonicUtils(unittest.TestCase):
    
    def test_with_h0(self):
        np.random.seed(1023)

        nd = 5
        h = np.array([0, 1, 2, 3, 6])
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        X0 = np.random.rand(Nhc, nd)
        verify_hutils(nd, h, X0, self)

    def test_without_h0(self):
        
        np.random.seed(1023)
        
        nd = 7
        h = np.array([1, 2, 3, 5, 7, 9])
        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        # X0 = np.zeros((Nhc, nd))
        # X0[0,0] = 0.5
        # X0[1,1] = 1
        # X0[2,2] = 0.75
        # X0[3,3] = 0.3
        X0 = np.random.rand(Nhc, nd)


        verify_hutils(nd, h, X0, self)
        
    def test_compile_speedup(self):
        
        
        h = np.array(range(9))
        Nhc = hutils.Nhc(h)
        
        
        ##################
        # Generate some different shapes of X0 and values to test
        np.random.seed(1023)
        
        X0 = np.random.rand(Nhc, 1)
        X1 = np.random.rand(Nhc, 1)
        X2 = np.random.rand(Nhc, 2)
        X3 = np.random.rand(Nhc, 3)
        
        ##################
        # Compile the function
        jit_fun = jax.jit(lambda X0 : combined_derivs_jax(X0, h))
        
        compile_time = timeit.timeit(lambda : jit_fun(X0).block_until_ready(), 
                                     number=1)

        # Verify that compiled form is faster for exact same inputs
        run_time = timeit.timeit(lambda : jit_fun(X0).block_until_ready(), 
                                     number=100)
        
        self.assertGreater(compile_time/run_time, 70, 'May be recompiling at each run.')
        
        self.assertLess(compile_time, 1, 
                        'Excessive time to compile the function, may be doing something poorly.')
        
        
        # Verify computation time is not recompiling for changing values of X
        run_time = timeit.timeit(lambda : jit_fun(X1).block_until_ready(), 
                                 number=100)
        
        self.assertGreater(compile_time/run_time, 70, 'May be recompiling at changing values.')
        
        # Run with some different sizes
        # These should separately compile and not break the compilation of the 
        # original size
        jit_fun(X2)
        jit_fun(X3)
        
        # Verify that the original set is still fast
        run_time = timeit.timeit(lambda : jit_fun(X0).block_until_ready(), 
                                     number=100)
        
        self.assertGreater(compile_time/run_time, 70, 'May be recompiling at each run.')
        
        ###################################
        # Check that values are correct for compiled functions with X0 and X1
        tol = 1e-12
        vxt_jax = jit_fun(X0)
        vxt = combined_derivs(X0, h)
        
        self.assertLess(np.linalg.norm(vxt_jax-vxt), tol, 
                        'Compiled function gives wrong time series')
        
        
        # Check that values are correct for compiled functions with X0 and X1
        vxt_jax = jit_fun(X1)
        vxt = combined_derivs(X1, h)
        
        self.assertLess(np.linalg.norm(vxt_jax-vxt), tol, 
                        'Compiled function gives wrong time series')
        
        pass

if __name__ == '__main__':
    unittest.main()