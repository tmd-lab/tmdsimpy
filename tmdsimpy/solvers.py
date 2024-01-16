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
        None.

        """
        self.stored_factor = (A,)
    
    def lin_factored_solve(self, b):
        """
        Solve the linear system with right hand side b and stored (factored)
        matrix from self.factor(A)

        Parameters
        ----------
        b : (N,) np.array, 1d
            Right hand side vector.

        Returns
        -------
        x : (N,) np.array, 1d
            Solution to the linear problem

        """
        A = self.stored_factor[0]
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
    