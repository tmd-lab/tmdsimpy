"""
Functions for postprocessing continuation results including interpolation 
methods. 
"""

import numpy as np

import scipy.interpolate

def hermite_upsample(XlamP_full, XlamP_grad_full, upsample_freq=10, 
                     new_points=None):
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
    XlamP_grad_full : (M, N) numpy.ndarray
        Gradients (prediction directions) at each of the continuation steps.
    upsample_freq : int, optional
        Factor of how many points should be included between each step. For the 
        default of 10, 9 points are added between each step resulting in 10x
        frequency. This argument is ignored if new_points is not None
        The default is 10.
    new_points : 1D numpy.ndarray, optional
        Array of fractional steps to interpolate to rather than using 
        upsample_freq. For instance 0.5 would correspond to half way between 
        the 0th and 1st indices of the solution points.
        The default is None.

    Returns
    -------
    XlamP_interp : (Minterp, N) numpy.ndarray
        Solutions at interpolated points where Minterp=M*upsample_freq or
        Minterp=new_points.shape[0]. 

    """
    
    if new_points is None:
        new_points = np.linspace(0, XlamP_full.shape[0]-1, 
                                 (XlamP_full.shape[0]-1)*upsample_freq+1)
    
    XlamP_interp = np.zeros((new_points.shape[0], XlamP_full.shape[1]))
    
    interp_obj = scipy.interpolate.CubicHermiteSpline(
                                    np.array(range(XlamP_full.shape[0])), # x
                                    XlamP_full, # y
                                    XlamP_grad_full, # dxdy
                                    axis=0,
                                    extrapolate=False)
    
    XlamP_interp = interp_obj(new_points)
    
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
        
    Notes
    -------
    1. Warning: The root finding to figure out where to interpolate to may not
    be completely robust for all cases due to small imaginary components when
    calculating the roots of a cubic polynomial. 
    
    2. Even though XlamP_full and XlamP_grad_full are checked to ensure that 
    lam is expected to monotonically increase, it may be possible to find 
    cases where lam does not monotonically increase even given these conditions. 
    In those cases, the function should thrown an error.
    
    
    

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
    dlam_full = XlamP_grad_full[:, -1]
    
    step_fracs = np.zeros_like(lams)
    
    # Find the fractional steps that will interpolate to the desired values of lam
    for ind in range(lams.shape[0]):
        
        lam_curr = lams[ind]
        
        # Find points on either side of interval
        left = np.searchsorted(lam_full, lam_curr, side='right')-1
        right = np.searchsorted(lam_full, lam_curr, side='left')
        
        if left == right:
            
            # Point is exactly captured in the solution already
            step_fracs[ind] = left
            
        else:
            # Find coefficients to the cubic polynomial for interpolating lam
            lamL = lam_full[left]
            dlamL = dlam_full[left]
            
            lamR = lam_full[right]
            dlamR = dlam_full[right]
            
            # rows are coefficients for x**3, x**2, x, 1
            # each column is a different shape function
            beam_shape_funs = np.array([[ 2,  1, -2,  1], 
                                        [-3, -2,  3, -1], 
                                        [ 0,  1,  0,  0], 
                                        [ 1,  0,  0,  0]])
            
            poly_coefs = beam_shape_funs @ np.array([lamL, dlamL, lamR, dlamR])
            
            
            # Find solution to the polynomial
            roots = _cubic_formula(poly_coefs[0], poly_coefs[1], 
                                   poly_coefs[2], poly_coefs[3]-lam_curr)
            
            # Verify single solution in bounds + real (or raise error)
            # One solution (the second) should be guaranteed to be real
            real_roots = np.abs(np.imag(roots) / np.real(roots)) < 1e-12
            
            if(real_roots.sum() == 0):
                roots = np.real(roots[1])
            else:
                roots = np.real(roots[real_roots])
                
            # Check in bounds
            in_bounds = np.logical_and(roots >= 0, roots <= 1)
            
            assert in_bounds.sum() == 1, \
                'Did not find exactly one real root in bounds.'
            
            # save solution
            step_fracs[ind] = roots[in_bounds][0] + left
    
    # Sample to the desired points
    XlamP_interp = hermite_upsample(XlamP_full, XlamP_grad_full, 
                                    new_points=step_fracs)
    
    return XlamP_interp

    
def _cubic_formula(a, b, c, d):
    """
    Calculate the roots of a cubic polynomial of the form
    a*x**3 + b*x**2 + c*x + d

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    roots : (3,) np.array
        Roots of the cubic polynomial
        
    References
    -------
    https://math.vanderbilt.edu/schectex/courses/cubic/

    """
    
    p = -b / (3*a)
    q = p**3 + (b*c - 3*a*d)/(6*a**2)
    r = c / (3*a)
    
    s = np.sqrt(complex(q**2 + (r - p**2)**3) )
    
    x0 = (q + s)**(1/3.0) + (q + s)**(1/3.0) + p
    x1 = (q + s)**(1/3.0) + (q - s)**(1/3.0) + p
    x2 = (q - s)**(1/3.0) + (q - s)**(1/3.0) + p
    
    return np.array([x0, x1, x2])