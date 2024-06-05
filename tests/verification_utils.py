import numpy as np



# Functions for verification work
def compare_mats(M, M_matlab, verbose=False):
    """
    Compare a matrix to an equivalent version that was loaded via MATLAB engine

    Parameters
    ----------
    M : Numpy matrix
    M_matlab : Matrix from matlab
    verbose : Option to print error

    Returns
    -------
    error : TYPE
        DESCRIPTION.

    """
    
    M_mat_np = np.array(M_matlab._data).reshape(M_matlab.size, order='F')
    
    error = np.max(np.abs(M.reshape(M_mat_np.shape) - M_mat_np))
    
    if verbose:
        print(f'Max Error is: {error:e}')
    
    return error
    

def check_grad(fun, U0, verbose=True, atol=1e-10, rtol=0.0, h=1e-5):
    """
    Default prints if verbose is True or if both atol and rtol are exceeded 
    
    Parameters
    ----------
    fun : TYPE
        DESCRIPTION.
    U0 : TYPE
        DESCRIPTION.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.
    atol : TYPE, optional
        DESCRIPTION. The default is 1e-10.
    rtol : TYPE, optional
        DESCRIPTION. The default is 0.0.
    h  : finite difference step size
            The default is 1e-5

    Returns
    -------
    grad_failed : is set to True if the gradient does not meet the specified 
                    tolerances, indicating that the gradient is considered 
                    incorrect.

    """
    
    U0 = U0*1.0 # ensure that there not integers where adding h would break test.
    
    Fnl, dFnldU = fun(U0)
    
    ########## Check dimensions to make sure valid inputs are passed.
    
    # Vector Length
    U0len = U0.shape[0]
    if U0len == 1 and len(U0.shape) == 2:
        U0len = U0.shape[1] # In case of 2D arrays
        
    # Grad Shape
    if (not isinstance(dFnldU, float)) and len(dFnldU.shape) == 2:
        # Normal rectangular array / gradient
        gradlen = dFnldU.shape[1]
    elif U0len == 1 or isinstance(dFnldU, float): 
        # 1D array since taking derivative w.r.t. scalar
        gradlen = 1
    else:
        # possible 1D array since taking derivative of a scalar w.r.t. a vector
        gradlen = dFnldU.shape[0]
    
    assert gradlen == U0len, 'Derivative dimensions do not match input vector.'
    
    if U0.shape[0] == 1:
        dFnldU = np.atleast_2d(dFnldU).T
    
    ########## Numerical Derivative
    
    dFnldU_num = np.zeros_like(dFnldU)
        
    for i in range(U0.shape[0]):
        U0[i] += h
        Fnl = fun(U0)[0]
        dFnldU_num[:, i] += Fnl #np.atleast_2d(Fnl).reshape((-1,))
        U0[i] -= 2*h
        Fnl = fun(U0)[0]
        dFnldU_num[:, i] -= Fnl # np.atleast_2d(Fnl).reshape((-1,))
        dFnldU_num[:, i]  = dFnldU_num[:, i] / (2*h)
        
        U0[i] += h
        
    # For Debugging:
    # import matplotlib
    # matplotlib.pyplot.spy(np.abs(dFnldU - dFnldU_num)> 1e-6)

    abs_error = np.max(np.abs(dFnldU - dFnldU_num))
    norm_error =  np.max(np.abs(dFnldU - dFnldU_num)) \
                    /( np.linalg.norm(dFnldU_num) + (np.linalg.norm(dFnldU_num)==0))
    
    grad_failed = (abs_error > atol and norm_error > rtol)
    
    if verbose or (abs_error > atol and norm_error > rtol) :
        print('Difference Between numerical and analytical Jacobian:', abs_error)
        print('Diff/norm(numerical Jacobian):', norm_error)
                
    return grad_failed
    