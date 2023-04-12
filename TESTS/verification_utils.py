import numpy as np



# Functions for verification work
def compare_mats(M, M_matlab, verbose=False):
    
    # Import MATLAB only if needed so can run some verification functions without
    import matlab.engine

    
    M_mat_np = np.array(M_matlab._data).reshape(M_matlab.size, order='F')
    
    error = np.max(np.abs(M.reshape(M_mat_np.shape) - M_mat_np))
    
    if verbose:
        print(f'Max Error is: {error:e}')
    
    return error
    

def check_grad(fun, U0, verbose=True, atol=1e-10, rtol=0.0):
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

    Returns
    -------
    grad_failed : is set to True if the gradient does not meet the specified 
                    tolerances, indicating that the gradient is considered 
                    incorrect.

    """
    h = 1e-5
    
    Fnl, dFnldU = fun(U0)
    
    if U0.shape[0] == 1:
        dFnldU = np.atleast_2d(dFnldU).T
    
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
    