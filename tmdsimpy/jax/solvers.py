import numpy as np
import jax


from ..solvers import NonlinearSolver


class NonlinearSolverOMP(NonlinearSolver):
    """
    Notes
    ----------
    
    Parallel linear and nonlinear solver functions here. 
    
    Libraries used respond to OpenMP environment variables such as: \n
    > export OMP_PROC_BIND=spread # Spread threads out over physical cores \n
    > export OMP_NUM_THREADS=32 # Change 32 to desired number of threads
    """
    
    def __init__(self, config={}):
        """
        Initialize nonlinear solver with solver settings

        Parameters
        ----------
        config : Dictionary of settings to be used in the solver (see below).

        Returns
        -------
        None.
        
        config keys
        -------
        max_steps : int, default 20
            maximum number of iterations allowed in the nonlinear solver
        reform_freq : int, default 1
            Frequency of recalculating and refactoring Jacobian matrix of the
            nonlinear problem. 1 corresponds to Newton-Raphson of doing this 
            every step. Larger numbers will correspond to BFGS low rank updates
            in between steps with refactoring. 
            When reform_freq > 1, function being solved must accept the keyword
            calc_grad=True or calc_grad=False to differentiate if Jacobian 
            matrix should be calculated.
        verbose : Boolean, default true
            Flag for if output should be printed. 
        xtol : double, default None
            Convergence tolerance on the L2 norm of the step size (dX). If None, 
            code will set the value to be equal to 1e-6*X0.shape[0] where X0 
            is the initial guess for a given solution calculation. 
            if xtol is passed to nsolve, that value is used instead
        rtol : double, default None
            convergence toleranace on the L2 norm of the residual vector (R).
        etol : double, default None
            convergence tolerance on the energy norm of the inner product of 
            step (dX) and residual (R) or e=np.abs(dX @ R)
        xtol_rel : double, default None
            convergence tolerance on norm(dX) / norm(dX_step0)
        rtol_rel : double, default None
            convergence tolerance on norm(R) / norm(R_step0)
        etol_rel : double, default None
            convergence tolerance on norm(e) / norm(e_step0)
        stopping_tol: list, default ['xtol']
            List can contain options of 'xtol', 'rtol', 'etol', 'xtol_rel', 
            'rtol_rel', 'etol_rel'. If any of the listed tolerances are 
            satisfied, then iteration is considered converged and exits. 
            Futher development would allow for the list to contain lists of 
            these same options and in a sublist, all options would be required. 
            This has not been implemented. 
        accepting_tol : list, default []
            List that can contain the same set of strings as stopping_tol. 
            Once maximum interactions has been reached, if any of these 
            tolerances are satisified by the final step, then the solution
            is considered converged. This allows for looser tolerances to be
            accepted instead of non-convergence, while still using max 
            iterations to try to achieve the tighter tolerances.
        
        """
        
        
        default_config={'max_steps' : 20,
                        'reform_freq' : 1,
                        'verbose' : True, 
                        'xtol'    : None, 
                        'rtol'    : None,
                        'etol'    : None,
                        'xtol_rel' : None,
                        'rtol_rel' : None,
                        'etol_rel' : None,
                        'stopping_tol' : ['xtol'],
                        'accepting_tol' : []
                        }
        
        
        for key in config.keys():
            default_config[key] = config[key]
        
        self.config = default_config
        
        # Memory place for storing a factored matrix for later backsub
        self.stored_factor = ()
        
        pass
    
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
        x = jax.numpy.linalg.solve(A,b)
        
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
        lu_and_piv = jax.scipy.linalg.lu_factor(A)
        
        self.stored_factor = (lu_and_piv,)
        
        return
    
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
        lu_and_piv = self.stored_factor[0]
        x = jax.scipy.linalg.lu_solve(lu_and_piv, b)
        
        return x
    
    def nsolve(self, fun, X0, verbose=True, xtol=None, Dscale=1.0):
        
        if xtol is None:
            xtol = self.config['xtol']
            if xtol is None: 
                xtol = 1e-6*X0.shape[0]
            
        # Save out the setting from xtol in config, then will overwrite
        # here to update to be used for this call. 
        xtol_setting = self.config['xtol'] 
        self.config['xtol'] = xtol
        
        max_iter = self.config['max_steps']
        
        sol = {'message' : 'failed', 
               'nfev' : 0, 
               'njev' : 0,
               'success' : False}
            
        X0c = X0 / Dscale
        
        R0,dR0 = fun(Dscale*X0c)
        
        dX0c = np.linalg.solve( -(dR0*Dscale), R0)
        
        e0  = np.abs(R0 @ dX0c)
        r0 = np.sqrt(R0 @ R0)
        u0 = np.sqrt(dX0c @ dX0c)
        
        Xc = X0c + dX0c
        
        # Tracking for convergence rates
        elist = np.zeros(max_iter+1)
        rlist = np.zeros(max_iter+1)
        ulist = np.zeros(max_iter+1)
        
        elist[0] = np.sqrt(np.abs(e0))
        rlist[0] = r0
        ulist[0] = u0
                
        form =  '{:4d} & {: 6.4e} & {: 6.4e} & {: 6.4e} '\
                    + '& {: 6.4e} & {: 6.4e} & {: 6.4e}' \
                    + ' & {: 6.4f} & {: 6.4f} & {: 6.4f}'
        if verbose:
            print('Iter &     |R|     &     |e|     &     |dU|    '\
                  + '&  |R|/|R0|   &  |e|/|e0|   &  |dU|/|dU0| ' \
                  +'&  Rate R &  Rate E &   Rate U')
            
                    
            print(form.format(0, r0, e0, u0, 
                              1.0, 1.0, 1.0,
                              np.nan, np.nan, np.nan))
                    
        rate_r = np.nan
        rate_e = np.nan
        rate_u = np.nan
        
        converged = _check_convg(self.config['stopping_tol'], self.config, 
                                 r0, e0, u0, 
                                 r0/r0, e0/e0, u0/u0)
        
        if converged:
            max_iter = 0 # already converged, so skip iteration loop
            R = R0
            dRdX = dR0
            
            if verbose:
                print('Converged!')
                
            sol['message'] = 'Converged'
            sol['nfev'] = 1
            sol['njev'] = 1
            sol['success'] = True
        
        for i in range(max_iter):
            
            R, dRdX = fun(Xc*Dscale)
            
            dXc = np.linalg.solve( -(dRdX*Dscale), R)
            
            u_curr = np.sqrt(dXc @ dXc)
            e_curr = R @ dXc
            r_curr = np.sqrt(R @ R)
            
            elist[i+1] = np.sqrt(np.abs(e_curr))
            rlist[i+1] = r_curr
            ulist[i+1] = u_curr
            
            
            if verbose:
                
                if i >= 1:
                    # import pdb; pdb.set_trace()
                    
                    rate_r = np.log(rlist[i] / rlist[i+1]) / np.log(rlist[i-1] / rlist[i])
                    rate_e = np.log(elist[i] / elist[i+1]) / np.log(elist[i-1] / elist[i])
                    rate_u = np.log(ulist[i] / ulist[i+1]) / np.log(ulist[i-1] / ulist[i])
                    
                print(form.format(i+1, r_curr, e_curr, u_curr, 
                              r_curr/r0, np.abs(e_curr/e0), u_curr/u0,
                              rate_r, rate_e, rate_u))
            
            Xc += dXc
            
            
            converged = _check_convg(self.config['stopping_tol'], self.config, 
                                 r_curr, e_curr, u_curr, 
                                 r_curr/r0, e_curr/e0, u_curr/u0)
            
            if converged:
                
                if verbose:
                    print('Converged!')
                    
                sol['message'] = 'Converged'
                sol['nfev'] = i+2
                sol['njev'] = i+2
                sol['success'] = True
                    
                break
            
        
        if not sol['success']:
            # Check convergence against the second set of tolerances
            
            converged = _check_convg(self.config['accepting_tol'], self.config, 
                                 r_curr, e_curr, u_curr, 
                                 r_curr/r0, e_curr/e0, u_curr/u0)
            
            if converged:
                
                if verbose:
                    print('Converged on accepting tolerances at max_iter.')
                    
                sol['message'] = 'Converged'
                sol['nfev'] = i+2
                sol['njev'] = i+2
                sol['success'] = True
            
        X = Xc * Dscale
        
        # R, dRdX = fun(X)

        if(verbose):
            print(sol['message'], ' Nfev=', sol['nfev'], ' Njev=', sol['njev'])
        
        # Set xtol in config back to the value passed in. 
        self.config['xtol'] = xtol_setting
        
        return X, R, dRdX, sol
    
    
def _check_convg(check_list, tol_dict, r_curr, e_curr, u_curr, r_rel, e_rel, u_rel):
    """
    Helper function to determine if convergence has been achieved. 

    Parameters
    ----------
    check_list : List
        List of tolerances to be checked. See NonlinearSolverOMP.__init__
        documentation
    tol_dict : Dictionary
        Contains tolerances and values to be checked.
    r_curr : double
        Current value of norm(R)
    e_curr : double
        Current value of e
    u_curr : double
        Current value of norm(dU)
    r_rel : double
        Current value of norm(R)/norm(R0)
    e_rel : double
        Current value of e/e0
    u_rel : double
        Current value of norm(dU)/norm(dU0)
    
    Returns
    -------
    converged : Boolean
        returns True if solution meets convergence criteria.

    """
    
    converged = False
    
    # Make dictionary of current errors
    error_dict = {
                'xtol'    : u_curr, 
                'rtol'    : r_curr,
                'etol'    : e_curr,
                'xtol_rel' : e_rel,
                'rtol_rel' : r_rel,
                'etol_rel' : u_rel,
                }
    
    for key in check_list:
        converged = converged or (np.abs(error_dict[key]) < tol_dict[key])
    
    return converged