"""
Submodule for postprocessing continuation results including interpolating.
"""

import numpy as np

import scipy.interpolate

def hermite_upsample(XlamP_full, XlamP_grad_full, upsample_freq=10, 
                     new_lams=None):
    """
    Function for upsampling continuation results using cubic hermite spline
    interpolation. 
    Continuation has taken M individual steps to solve the problem with
    N degrees of freedom. 

    Parameters
    ----------
    XlamP_full : (M, N) numpy.ndarray
        Solution points calculated with continuation. First dimension 
        corresponds to individual solutions. Second dimension corresponds
        to degrees of freedom at each solution point.
        The last column corresponds to the continuation control parameter lam
        and must be monotonically increasing for this function.
    XlamP_grad_full : (M, N) numpy.ndarray
        Gradients (prediction directions) at each of the continuation steps.
        The last column corresponds to the continuation control parameter lam
        and must be strictly greater than 0 for this function.
    upsample_freq : int, optional
        Factor of how many points should be included between each step. For the 
        default of 10, 9 points are added between each step resulting in 10x
        frequency. This argument is ignored if new_points is not None
        The default is 10.
    new_lams : 1D numpy.ndarray, optional
        Array of new values of lam to interpolate to. If None, then the 
        upsample_freq is used instead.
        The default is None.

    Returns
    -------
    XlamP_interp : (Minterp, N) numpy.ndarray
        Solutions at interpolated points where Minterp=M*upsample_freq or
        Minterp=new_points.shape[0]. 

    """
    
    if new_lams is None:
        new_steps = np.linspace(0, XlamP_full.shape[0]-1, 
                                 (XlamP_full.shape[0]-1)*upsample_freq+1)
        
        # New values of lambda to evaluate function at 
        new_lams = np.interp(new_steps, # x
                             range(XlamP_full.shape[0]), # xp
                             XlamP_full[:, -1]) # fp
    
    XlamP_interp = hermite_interp(XlamP_full, XlamP_grad_full, new_lams)
    
    return XlamP_interp


def hermite_interp(XlamP_full, XlamP_grad_full, lams):
    """
    Interpolate continuation solutions to specific values of the control
    parameter (lam, corresponding to the last column)

    Parameters
    ----------
    XlamP_full : (M, N) numpy.ndarray
        Solution points calculated with continuation. First dimension 
        corresponds to individual solutions. Second dimension corresponds
        to degrees of freedom at each solution point.
        The last column must monotonically increase.
    XlamP_grad_full : (M, N) numpy.ndarray
        Gradients (prediction directions) at each of the continuation steps.
        The last column must be strictly greater than 0.
    lams : (Minterp,) numpy.ndarray
        Values of the last variable of XlamP to interpolate solutions to. 
        In other words, XlamP_interp[:, -1] = lams.

    Returns
    -------
    XlamP_interp : (Minterp, N) numpy.ndarray
        Solutions at interpolated points.
    
    """
    
    assert np.all(np.diff(XlamP_full[:, -1]) > 0), \
            'The last column of XlamP_full must monotonically increase.'
            
    assert np.all(XlamP_grad_full[:, -1] > 0), \
            'All gradients in the last column of XlamP_grad_full must be positive.'
            
    assert lams.max() <= XlamP_full[-1, -1], \
            'lams.max() is out of bounds of input solutions.'
            
    assert lams.min() >= XlamP_full[0, -1], \
            'lams.min() is out of bounds of input solutions.'
    
    lam_full = XlamP_full[:, -1]
    dlam_full = XlamP_grad_full[:, -1:]
    
    
    
    interp_obj = scipy.interpolate.CubicHermiteSpline(
                                    lam_full, # x
                                    XlamP_full[:, :-1], # y
                                    XlamP_grad_full[:, :-1]/dlam_full, # dxdy
                                    axis=0,
                                    extrapolate=False)
    
    XlamP_interp = np.hstack((interp_obj(lams), lams.reshape(-1,1)))
    
    return XlamP_interp

def linear_interp(XlamP_full, new_values, reference_values=None):
    """
    Interpolates each column of the first argument to a new set of values

    Parameters
    ----------
    XlamP_full : (N, K) numpy.ndarray
        Input set of solutions that are to be interpolated to new values.
    new_values : (M,)
        New values that `XlamP_full` should be interpolated to.
    reference_values : (N,) numpy.ndarray or None, optional
        Reference values to compare `new_values` to corresponding to each
        row of `XlamP_full`. If None, then the last column of `XlamP_full` is
        used instead.
        The default is None.

    Returns
    -------
    XlamP_interp : (M,K) numpy.ndarray
        Interpolated values. Returns np.nan for rows where `new_values` is 
        outside of the bounds of `reference_values`.

    """
    
    if reference_values is None:
        reference_values = XlamP_full[:, -1]
    
    assert np.all(np.diff(reference_values) > 0), \
        'Reference values must be monotonically increasing.'
    
    N = reference_values.shape[0]
    
    frac_ind = np.interp(new_values, reference_values, np.arange(N),
                         left=np.nan, right=np.nan)
    
    frac_ind = np.atleast_1d(frac_ind)
    
    nan_mask = np.isnan(frac_ind)
    frac_ind[nan_mask] = 0
    
    whole_ind = np.int64(frac_ind)
    
    # prevent error if interpolating to exact last point
    next_ind = np.minimum(whole_ind + 1, N-1) 
    
    remainder_ind = frac_ind - whole_ind
    
    XlamP_interp = XlamP_full[whole_ind, :]
    
    XlamP_interp += remainder_ind.reshape(-1, 1)\
        *(XlamP_full[next_ind, :] - XlamP_full[whole_ind, :])
    
    XlamP_interp[nan_mask, :] = np.nan
    
    return XlamP_interp

