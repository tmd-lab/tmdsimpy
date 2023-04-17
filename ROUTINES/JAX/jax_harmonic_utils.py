"""
Subset of harmonic_utils using JAX

Autodiff is applied mainly for AFT at this point, thus only some functions
need to be converted to JAX. Other functions are not updated at this point to 
use JAX. 
"""

import numpy as np

from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Imports for decorating functions with jit calls at this level
# See specific function comments for where this cannot be done.
#
# import jax
# from functools import partial

# @partial(jax.jit, static_argnums=(0,1,3)) # May cause excessive recompile, do not use here.
def time_series_deriv(Nt, htuple, X0, order):
    """
    jax/jit version - set Nt, h to static
    
    Returns Derivative of a time series defined by a set of harmonics
    
    This function cannot be compiled with partial for the current 
    implementation. Rather this function gets different compiled versions for
    the top level function being compiled when AFT uses both displacement
    and velocity. 
    
    If this function was compiled with partial, it would be forced to recompile
    every time that order changed. This could cause excessive recompilation. 
    
    If the top level function gets compiled, it appears that two versions of 
    this are created or this code is inlined into the top level function. 
    This could be verified by profiling the final code to ensure that 
    there are not lots of recompiles during evaluations.
    
    Parameters
    ----------
    Nt : Number of times considered, must be even
    htuple : Harmonics considered, 0th harmonic must be first if included
            use tuple(h) to convert from np.array to a tuple.
    X0 : Harmonic Coefficients for Nhc x nd
    order : Order of the derivative returned
    
    Returns
    -------
    x_t : time series of each DOF, Nt x nd
    """
    
    h = np.array(htuple)
    
    assert ((np.equal(h, 0)).sum() == 0 or h[0] == 0), 'Zeroth harmonic must be first'
    
    nd = X0.shape[1] # Degrees of Freedom
    Nh = np.max(h)
    
    # Create list including all harmonic components
    X0full = jnp.zeros((2*Nh+1, nd), np.float64)
    if h[0] == 0:
        X0full = X0full.at[0, :].set(X0[0, :])
        X0full = X0full.at[2*h[1:]-1, :].set(X0[1::2, :])
        X0full = X0full.at[2*h[1:], :].set(X0[2::2, :])
        
    else:
        X0full = X0full.at[2*h-1, :].set(X0[0::2, :])
        X0full = X0full.at[2*h, :].set(X0[1::2, :])
        
    # Check that sufficient time is considered
    assert Nt > 2*Nh + 1, 'More times are required to avoid truncating harmonics.'
    
    if order > 0:
        D1 = np.zeros((2*Nh+1, 2*Nh+1))
        
        # If order is static, this for loop should be safe since it can be 
        # calculated at compile time (uses numpy operations)
        # if order is not static, JIT will throw an error.
        
        # Note that if top level functions are compiled, multiple versions of 
        # this can be compiled for different order cases
        
        for k in h[h != 0]:
            # Only rotates the derivatives for the non-zero harmonic components
            cosrows = (k-1)*2 + 1
            sinrows = (k-1)*2 + 2
            
            D1[cosrows, sinrows] = k
            D1[sinrows, cosrows] = -k
            
        # This is not particularly fast, consider optimizing this portion.
        #   D could be constructed just be noting if rows flip for odd/even
        #   and sign changes as appropriate.
        D = np.linalg.matrix_power(D1, order)
        
        X0full = D @ X0full
    
    # Extend X0full to have coefficients corresponding to Nt times for ifft
    #   Previous MATLAB implementation did this before rotating harmonics, but
    #   that seems rather inefficient in increasing the size of the matrix 
    #   multiplication
    Nht = int(Nt/2 -1)
    X0full = jnp.vstack((X0full,np.zeros((2*(Nht-Nh), nd)) ))
    Nt = 2*Nht+2

    # Fourier Coefficients    
    Xf = jnp.vstack((2*X0full[0, :], \
         X0full[1::2, :] - 1j*X0full[2::2], \
         jnp.zeros((1, nd)), \
         X0full[-2:0:-2, :] + 1j*X0full[-1:1:-2]))
        
    Xf = Xf * (Nt/2)
         
    assert Xf.shape[0] == Nt, 'Unexpected length of Fourier Coefficients'
    
    x_t = jnp.real(jnp.fft.ifft(Xf, axis=0))
    
    return x_t

# @partial(jax.jit, static_argnums=(0))
def get_fourier_coeff(htuple, x_t):
    """
    jax/jit version - set h to static
    
    Calculates the Fourier coefficients corresponding to the harmonics in h of
    the input x_t

    Parameters
    ----------
    htuple : Harmonics of interest, 0th harmonic must be first if included
            tuple(h) so that it is hashable
    x_t : Time history of interest, Nt x nd

    Returns
    -------
    v : Vector containing fourier coefficients of harmonics h
    """
    
    h = np.array(htuple)

    Nt, nd = x_t.shape
    Nhc = 2*(h != 0).sum() + (h == 0).sum() # Number of Harmonic Components
    n = h.shape[0] - (h[0] == 0)
    
    assert ((h == 0).sum() == 0 or h[0] == 0), 'Zeroth harmonic must be first'
    
    v = jnp.zeros((Nhc, nd))
    
    xf = jnp.fft.fft(x_t, axis=0)
        
    if h[0] == 0:
        v = v.at[0, :].set(jnp.real(xf[0, :])/Nt)
        zi = 1
    else:
        zi = 0

    # As long as h is treated as static, this is safe for this for loop
    for i in range(n):
        hi = h[i + zi]
        v = v.at[2*i+zi].set(jnp.real(xf[hi, :]) / (Nt/2))
        v = v.at[2*i+1+zi].set(-jnp.imag(xf[hi, :]) / (Nt/2))
    
    return v