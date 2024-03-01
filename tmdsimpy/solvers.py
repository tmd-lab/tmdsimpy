import numpy as np
import scipy.optimize
import scipy.linalg
import warnings

class NonlinearSolver:
    
    def __init__(self, Dscale=1):
        """
        Initialize nonlinear solver with solver settings

        Parameters
        ----------
        Dscale : To be implemented, something with scaling unknowns

        Returns
        -------
        None.

        """
        self.stored_factor = ()
    
    def lin_solve(self, A, b):
        """
        Solve the linear system A * x = b 

        Parameters
        ----------
        A : (N,N) np.array, 2d
            Linear system matrix.
        b : (N,) np.array, 1d
            Right hand side vector.

        Returns
        -------
        x : (N,) np.array, 1d
            Solution to the linear problem

        """
        x = np.linalg.solve(A, b)
        
        return x
    
    def lin_factor(self, A):
        """
        Factor a matrix A for later solving. This version simply stores and 
        fully solves later.

        Parameters
        ----------
        A : (N,N) np.array, 2d
            Linear system matrix for later solving.

        Returns
        -------
        factor_res : tuple
            Resulting data from factoring the matrix A, can be passed to 
            self.lin_factored_solve to solve the linear system. This solver
            version does not do anything other than return A in a tuple

        """
        
        return (A,)
    
    def lin_factored_solve(self, factor_res, b):
        """
        Solve the linear system with right hand side b and stored (factored)
        matrix from self.factor(A)

        Parameters
        ----------
        factor_res : tuple
            Collected data from self.lin_factor that will be used here. This
            version just is the tuple (A,)
        b : (N,) np.array, 1d
            Right hand side vector.

        Returns
        -------
        x : (N,) np.array, 1d
            Solution to the linear problem

        """
        A = factor_res[0]
        
        x = np.linalg.solve(A, b)
        
        return x
    
    def nsolve(self, fun, X0, verbose=True, xtol=None):
        
        if xtol is None:
            xtol = 1e-6*X0.shape[0]
        
        # scipy.optimize.show_options('root', 'hybr') # Does not support callback
        hybr_opts = {'xtol': xtol, 'maxfev': 0}
        
        sol = scipy.optimize.root(fun, X0, method='hybr', jac=True, options=hybr_opts)
        
        X = sol['x']
        
        # Jacobian from the optimization may not be exact at final point
        R, dRdX = fun(X)

        if(verbose):
            print(sol['message'], ' Nfev=', sol['nfev'], ' Njev=', sol['njev'])
        
        return X, R, dRdX, sol
    
    def eigs(self, K, M=None, subset_by_index=[0, 2]):
        """
        Conduct eigenvalue analysis for a linear system

        Parameters
        ----------
        K : (N,N) np.array
            Stiffness matrix for general second order system.
        M : (N,N) np.array, optional
            Mass matrix for general second order system. The default is None.
        subset_by_index : list of length 2, optional
            Subset indices for which eigenvalues should be calculated. 
            See scipy.linalg.eigh for more details. 
            Let M be the number of requested eigenvalues by this parameter.
            The default is [0, 2].

        Returns
        -------
        eigvals : (M,) np.array
            Eigenvalues of the linear problem.
        eigvecs : (N,M) np.array
            Eigenvectors of linear problem with columns corresponding to 
            individual eigenvalues.

        """
        
        subset_by_index[1] = min(subset_by_index[1], K.shape[0]-1)
        
        eigvals, eigvecs = scipy.linalg.eigh(K, M, 
                                             subset_by_index=subset_by_index)
        
        return eigvals, eigvecs
    
    def conditioning_wrapper(self, fun, CtoP, RPtoC=1.0):
        """
        Function to create a wrapper around fun to solve within conditioned 
        space

        Parameters
        ----------
        fun : function
            Function that is to be wrapped in condition space. Function should
            take two arguments, one is unknown vector Xp, other is optional 
            argument of calc_grad=True. 
            The calc_grad=True is only passed to fun, if calc_grad=False 
            is passed to the fun_conditioned that is returned.
        CtoP : (N,) numpy.ndarray
            Vector describing conversion from physical coordinates to 
            conditioned coordinates for the unknown vector that is input to 
            fun.
        RPtoC : float, optional
            Scales the full ouptut residual vector by this magnitude.
            DESCRIPTION. The default is 1.0.

        Returns
        -------
        fun_conditioned : function
            Function that describes the same nonlinear problem as fun, but in
            a conditioned space. 
            Function takes input of Xc where Xp = CtoP * Xc.
            Second optional input to the fuction is calc_grad=True.
            This function returns a residual vector for conditioned inputs.
            If calc_grad=True, the Jacobian in conditioned space is also 
            returned by this function.

        """
        
        return lambda Xc, calc_grad=True : _conditioned_fun(Xc, CtoP, RPtoC, \
                                                            calc_grad, fun)
        
def _conditioned_fun(Xc, CtoP, RPtoC, calc_grad, fun):
    """
    Private function for conditioned space residual / nonlinear solution.

    Parameters
    ----------
    Xc : (N,) numpy.ndarray
        Conditioned coordinates for the unknown solution.
    CtoP : (N,) numpy.ndarray
        Converstion to physical Xp coordinates as Xp = Xc * CtoP.
    RPtoC : float
        Scaled value to multiply the returned residual by.
    calc_grad : bool
        Flag for calculating the gradient if true.
    fun : function
        Function that is going to be put into conditioned space. Function 
        takes inputs of Xp and if calc_grad=False, it accepts calc_grad as
        an optional argument with default True.
        fun always returns a tuple with the first entry being the residual 
        vector. If calc_grad=True, a second entry of the tuple is (N,N) dRdXp

    Returns
    -------
    R : (N,) numpy.ndarray
        Residual function that would like to solve to be zeros. 
        ALways returned as first entry of a tuple
    dRdXc : (N,N) numpy.ndarray
        Jacobian matrix of derivative of residual w.r.t. Xc
        Only returned if calc_grad=True

    """
    
    Xp = Xc*CtoP
    if calc_grad:
        R, dRdXp = fun(Xp)
        
        Rc = RPtoC * R
        dRcdXc = RPtoC*dRdXp*CtoP
        
        return (Rc, dRcdXc)
        
    else:
        R = fun(Xp, calc_grad=False)[0]
    
        Rc = RPtoC * R[0]
        
        return (Rc,)