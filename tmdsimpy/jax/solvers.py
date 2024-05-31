import numpy as np
import jax


from ..solvers import NonlinearSolver


class NonlinearSolverOMP(NonlinearSolver):
    """
    Nonlinear solver object that contains several functions and solver settings

    Parameters
    ----------
    config : dict, optional
        Dictionary of settings to be used in the solver (see below).

    Notes
    ----------
    
    Parallel linear and nonlinear solver functions here. 
    
    Libraries used may respond to OpenMP environment variables such as: \n
    > export OMP_PROC_BIND=spread # Spread threads out over physical cores \n
    > export OMP_NUM_THREADS=32 # Change 32 to desired number of threads

    
    config dictionary keys
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
        matrix should be calculated. If calc_grad=True, then returned tuple
        should be (R, dRdX) if False, returned tuple should start with (R,), 
        but may return other values past the 0th index of tuple.
        Function may be a lambda function that completely ignores calc_grad, 
        for instance: `fun = lambda X, calc_grad=True : fun0(X)`
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
    line_search_iters : int, optional
        Number of iterations used in line search (self.line_search). 
        If 0, line search is not used.
        If line search is desired, a recommended value is less than 10, perhaps
        about 2 to 5.
        If it is greater than 0, then function being solved must accept 
        calc_grad=True or calc_grad=False as inputs.
        The default is 0.
    line_search_tol : float, optional
        If the line search function decreases to be less than the initial value
        times this tolerance, than it is accepted as converged.
        This is not intended to be a tight tolerance, line search is just 
        intended to quickly reduce the step in case of poor problem behavior.
        The default is 0.5.
    line_search_same_sign : bool, optional
        If true, line search tries to only a return a step that does not change
        the sign of the quantity deltaX @ R(X + alpha*deltaX).
        The default is True.
    """
    
    def __init__(self, config={}):
        
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
                        'accepting_tol' : [],
                        'line_search_iters' : 0,
                        'line_search_tol' : 0.5,
                        'line_search_same_sign' : True
                        }
        
        
        for key in config.keys():
            default_config[key] = config[key]
        
        self.config = default_config
        
        return

    def edit_config(self, new_config):
        """
        Edit the solver configuration settings after initialization.

        Parameters
        ----------
        new_config : dict
            Dictionary of key value pairs for the new settings. The settings
            that can be changed are the same as those when creating a new
            object

        Returns
        -------
        None.

        """

        for key in new_config.keys():
            self.config[key] = new_config[key]

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
        lu_and_piv : tuple
            Resulting data from factoring the matrix A, can be passed to 
            self.lin_factored_solve to solve the linear system.

        """
        lu_and_piv = jax.scipy.linalg.lu_factor(A)
        
        return lu_and_piv
    
    def lin_factored_solve(self, lu_and_piv, b):
        """
        Solve the linear system with right hand side b and stored (factored)
        matrix from self.factor(A)

        Parameters
        ----------
        lu_and_piv : tuple
            results from factoring a matrix with self.lin_factor(A)
        b : (N,) np.array, 1d
            Right hand side vector.

        Returns
        -------
        x : (N,) np.array, 1d
            Solution to the linear problem

        """
        x = jax.scipy.linalg.lu_solve(lu_and_piv, b)
        
        return x
    
    def line_search(self, fun, X, Rx, deltaX):
        """
        Line search algorithm to help in the numerical solution to a set of 
        nonlinear equations. This is used by nsolve.

        Parameters
        ----------
        fun : function handle
            Function to be solved, returns 
            R=fun(X, calc_grad={True or False})[0].
            Must accept the input argument calc_grad=True and calc_grad=False.
            Function may be a lambda function that completely ignores calc_grad
        X : (N,) numpy.ndarray
            Values of unknowns at initial point.
        Rx : (N,) numpy.ndarray
            Residual at X, equal to fun(X)[0].
        deltaX : (N,) numpy.ndarray
            Step direction of interest, generally calculated based on gradient
            solution step.

        Returns
        -------
        alpha : float
            Fraction of deltaX step that should be taken. Recommended update
            is X = X + alpha*deltaX
        sol : dict
            Description of final solution state. Has keys of 
            ['message', 'nfev']. 
            'nfev' is the number of function evaluations completed. 
            'message' describes how line search exited.
        
        See Also
        --------
        NonlinearSolverOMP : 
            Documentation for the solver class describes configurations 
            and settings of the numerical solver that are configured at 
            creation rather than at solution time. 
            Relevant settings are 'line_search_iters', 'line_search_tol',
            'line_search_same_sign'.
        nsolve :
            Nonlinear solver routine that calls this function (Newton-Raphson 
            + BFGS Solver)
        
        Notes
        -----
        The line search algorithm here is based on [1]_. In the finite element
        context, the objective is to find the zero of an 'energy' norm of 
        R^T deltaX. The bisection algorithm is used to find a solution for
        R(X + alpha*deltaX) with alpha in [0, 1]. A very loose tolerance
        is generally desired to minimize additional computational cost of this 
        function before returing to nsolve.
        
        References
        ----------
        .. [1] Matthies, H., Strang, G., 1979. The solution of nonlinear finite
        element equations. International Journal for Numerical Methods in 
        Engineering 14, 1613–1626. https://doi.org/10.1002/nme.1620141104

        """
        
        nfev = 0 # Count number of function evals
        message = 'Line search exited at max iterations.'
        
        Galpha0 = deltaX @ Rx
        
        alpha_bracket = [0, 1]
        
        # Verify that the sign of G(alpha) changes if a step of size deltaX
        # is taken
        R1 = fun(X + deltaX, calc_grad=False)[0]
        nfev += 1
        
        Galpha1 = deltaX @ R1
        
        # Initial Brackets of data
        Galpha_bracket = [Galpha0, Galpha1]
        R_bracket = [Rx, R1]
        
        # Start is converged? 
        converged = False
        
        if Galpha1 * Galpha0 > 0:
            # No line search is needed since there is expected to not a zero in
            # the segment based on the sign of G(alpha) being the same
            message = 'Line search is not needed, step is considered safe.'
            converged = True
            
        elif not self.config['line_search_same_sign'] and \
            np.abs(Galpha1) < self.config['line_search_tol']*np.abs(Galpha0):
            
            # alpha = 1.0 satisfies the tolerance, but changes the sign on 
            # line search
            message = 'Line search alpha=1 satisfies tolerance.'
            converged = True
            
            if self.config['verbose']:
                print(message)
            
        if converged:
            
            sol = {'nfev' : nfev,
                   'message' : message,
                   'R(alpha)' : R1,
                   'G(alpha)_bracket' : Galpha_bracket,
                   'R_bracket' : R_bracket,
                   'alpha_bracket' : [0, 1],
                   'Galpha0' : Galpha0}
            
            return 1.0, sol
        
        # Do a loop to get the solution to better converge
        
        for ind in range(self.config['line_search_iters']):
            
            alpha = 0.5*(alpha_bracket[0]+alpha_bracket[1])
            
            Rmid = fun(X + 0.5*(alpha_bracket[0]+alpha_bracket[1])*deltaX, 
                       calc_grad=False)[0]
            nfev += 1
            
            Gmid = deltaX @ Rmid
            
            if Gmid * Galpha_bracket[0] < 0:
                alpha_bracket[1] = alpha
                Galpha_bracket[1] = Gmid
                R_bracket[1] = Rmid
            else:
                alpha_bracket[0] = alpha
                Galpha_bracket[0] = Gmid
                R_bracket[0] = Rmid
            
            if self.config['line_search_same_sign'] and \
                np.abs(Galpha_bracket[0]) \
                < self.config['line_search_tol']*np.abs(Galpha0):
                
                message = 'Line search converged to tolerance with same sign '\
                    + 'as start.'
                # Convergence criteria only on the same sign bound if that
                # is a requirement of the output
                break 
            
            elif not self.config['line_search_same_sign'] and \
                np.abs(Gmid) < self.config['line_search_tol']*np.abs(Galpha0):
                
                message = 'Line search converged to tolerance.'
                
                # General convergence on either bound if do not need same sign
                break
            
        if self.config['line_search_same_sign']:
            # Avoid overshooting zero
            alpha = alpha_bracket[0]
            Ralpha = R_bracket[0]
            
        elif np.abs(Galpha_bracket[0]) < np.abs(Galpha_bracket[1]):
            # Lower bound is closer to solution
            alpha = alpha_bracket[0]
            Ralpha = R_bracket[0]
        else:
            alpha = alpha_bracket[1]
            Ralpha = R_bracket[1]
            
        if alpha == 0.0:
            # Must make some step even if it is worse on line search
            alpha = alpha_bracket[1]
            Ralpha = R_bracket[1]
            
            message += ' Line search changed from final solution to upper'\
                + ' bracket to prevent an alpha=0 step (no change in X).'
            
        if self.config['verbose']:
            print('Line search finished with '\
                  + '{} iterations, alpha={:.4f}.'.format(ind+1, alpha))
            print(message)
            
        sol = {'nfev' : nfev,
               'message' : message,
               'R(alpha)' : Ralpha,
               'G(alpha)_bracket' : Galpha_bracket,
               'R_bracket' : R_bracket,
               'alpha_bracket' : alpha_bracket,
               'Galpha0' : Galpha0}
                
        return alpha, sol
    
    def nsolve(self, fun, X0, verbose=None, xtol=None, Dscale=1.0):
        """
        Numerical nonlinear root finding solution to the problem of R = fun(X)
        
        Solver settings are set at initialization of NonlinearSolverOMP 
        (see that documentation).

        Parameters
        ----------
        fun : function handle 
            Function to be solved, function returns two arguments of R 
            (residual, (N,) numpy.ndarray) and dRdX (residual jacobian, 
            (N,N) numpy.ndarray).
            If config['reform_freq'] > 1, then fun should take two arguments
            The first is X, the second is a bool where if True, fun returns 
            a tuple of (R,dRdX). If false, fun just returns a tuple (R,)
            Function may return additional values in either tuple, but the
            additional values will be ignored here.
        X0 : (N,) numpy.ndarray
            Initial guess of the solution to the nonlinear problem.
        verbose : bool or None, optional
            Flag to print convergence information if True.
            If None, then the configuration setting for 'verbose' from 
            initialization will be used (default True)
            The default is None.
        xtol : float, optional
            Tolerance to check for convergence on the step size. 
            If None, then self.config['xtol'] is used. If that is also None, 
            then 1e-6*X0.shape[0] is used as the xtolerance.
            Passing in a value here does not change the config value 
            permanently (not parallel safe though)
            The default is None. 

        Returns
        -------
        X : (N,) numpy.ndarray
            Solution to the nonlinear problem that satisfies tolerances or from
            last step.
        R : (N,) numpy.ndarray
            Residual vector from the last function evaluation (does not in 
            generally correspond to value at X to save extra evaluation of fun).
        dRdX : (N,N) numpy.ndarray
            Last residual jacobian as evaluated during solution, not at final X.
        sol : dict
            Description of final convergence state. Has keys of 
            ['message', 'nfev', 'njev', 'success']. 'success' is a bool with 
            True corresponding to convergence. 'nfev' is the number of function
            evaluations completed. 'njev' is the number of jacobian evaluations.
            'message' is either 'Converged' or 'failed'. Use the bool from 
            'success' rather than the message for decisions. 
            'nfev' does not include any function evaluations done as part 
            of the line search routine. 
            
        See Also
        --------
        NonlinearSolverOMP : 
            Documentation for the solver class describes configurations 
            and settings of the numerical solver that are configured at 
            creation rather than at solution time. 
        line_search : 
            Class method that may be used to improve convergence of nsolve
            if the appropriate solver settings are used.
            
        Notes
        -----
        This function uses either a full Newton-Raphson (NR) solver approach or
        Broyden-Fletcher-Goldfarb-Shanno (BFGS), which uses fewer NR iterations
        with some approximations of Jacobian between NR iterations.
        For BFGS see Algorithm 7.4 in [1]_.
        
        The output status of current errors / norms of residual etc. are 
        intended to provide easy checks to see what is going on during 
        solutions. These outputs are slightly convoluted in that some outputs
        are for the previous iteration, check code for exact details.
            
        Other Parameters
        ----------------
        Dscale : float or numpy.ndarray, optional
            This argument does nothing. Conditioning of the numerical problem
            should be achieved by wrapping the problem of interest rather than
            here.
            
        References
        ----------
        .. [1] Nocedal, J., Wright, S.J., 2006. Numerical optimization, 2nd ed, 
        Springer series in operations research. Springer, New York.


        """
        
        ##########################################################
        # Initialization
        
        # xtol support with backwards compatibility 
        if xtol is None:
            xtol = self.config['xtol']
            if xtol is None: 
                xtol = 1e-6*X0.shape[0]
            
        if verbose is None:
            verbose = self.config['verbose']
            
        # Save out the setting from xtol in config, then will overwrite
        # here to update to be used for this call. 
        xtol_setting = self.config['xtol'] 
        self.config['xtol'] = xtol
        
        max_iter = self.config['max_steps']
        
        sol = {'message' : 'failed', 
               'nfev' : 0, 
               'njev' : 0,
               'success' : False}
        
        # Wrap function if using BFGS v. NR
        if self.config['reform_freq'] > 1:
            fun_R_dRdX = lambda X : fun(X, True)[0:2]
            fun_R = lambda X : fun(X, False)[0]
        else:
            fun_R_dRdX = lambda X : fun(X)[0:2]
            
        # Solution initialization
        X = X0
        
        # Previous iteration quantities, these are initialized to zero to 
        # prevent undefind variable names in python, but are redefined in the 
        # loop before they are used.
        deltaXminus1 = np.nan*np.zeros_like(X)
        Rminus1 = np.zeros_like(X)
        
        # Output printing form
        form =  '{:4d} & {: 6.4e} & {: 6.4e} & {: 6.4e} '\
                    + '& {: 6.4e} & {: 6.4e} & {: 6.4e}' \
                    + ' & {: 6.4f} & {: 6.4f} & {: 6.4f} ' \
                    + '& {:s}'
                    
        # Tracking for convergence rates
        elist = np.zeros(max_iter+1)
        rlist = np.zeros(max_iter+1)
        ulist = np.zeros(max_iter+1)
        
        rate_r = np.nan
        rate_e = np.nan
        rate_u = np.nan
            
        ##########################################################
        # Iteration Loop
        bfgs_ind = 0 # counter to check if it is time to do full NR again
        curr_iter = 'NR'
        no_nan_vals = True
        
        for i in range(max_iter):
            
            if bfgs_ind == 0: # Full Newton Update Update
                curr_iter = 'NR'
                
                R,dRdX = fun_R_dRdX(X)
                sol['nfev'] += 1
                sol['njev'] += 1
                
                if np.isnan(np.sum(R)):
                    if verbose: print('Stopping with NaN Residual')
                    no_nan_vals = False
                    break
                if np.isnan(np.sum(dRdX)):
                    if verbose: print('Stopping with NaN Jacobian')
                    no_nan_vals = False
                    break
                
                factored_data = self.lin_factor(dRdX)
                
                deltaX = -self.lin_factored_solve(factored_data, R)
                
                bfgs_s = np.zeros((X.shape[0], self.config['reform_freq']-1))
                bfgs_y = np.zeros((X.shape[0], self.config['reform_freq']-1))
                bfgs_p = np.zeros((self.config['reform_freq']-1))
                
            else: # BFGS Update        
                curr_iter = 'BFGS'
                
                R = fun_R(X)
                sol['nfev'] += 1
                
                if np.isnan(np.sum(R)):
                    if verbose: print('Stopping with NaN Residual')
                    no_nan_vals = False
                    break
            
                bfgs_s[:, bfgs_ind-1] = deltaXminus1
                bfgs_y[:, bfgs_ind-1] = R - Rminus1
                bfgs_p[bfgs_ind-1] = 1.0 / (bfgs_s[:, bfgs_ind-1] @ bfgs_y[:, bfgs_ind-1])
                
                # Apply the updated jacobian to R
                deltaX = -R # Negative added here compared to referenced algorithm so that result is directly step
                
                alpha = np.zeros(bfgs_ind)
                for kk in range(bfgs_ind-1, -1, -1):
                    alpha[kk] = bfgs_p[kk] * (bfgs_s[:, kk] @ deltaX)
                    deltaX = deltaX - alpha[kk]*bfgs_y[:, kk]
                
                deltaX = self.lin_factored_solve(factored_data, deltaX)
                
                for kk in range(0, bfgs_ind, 1):
                    beta = bfgs_p[kk] * (bfgs_y[:, kk] @ deltaX)
                    deltaX = deltaX + bfgs_s[:, kk]*(alpha[kk] - beta)
               
            ###### # Check Solution Step
            if np.isnan(np.sum(deltaX)):
                no_nan_vals = False
                if verbose: print('Stopping with NaN Step Direction')
                break
            
            ###### # Tolerance Calculations
            # Tolerances should not include line search scaling of deltaX
            u_curr = np.sqrt(deltaX @ deltaX)
            e_curr = R @ deltaXminus1
            r_curr = np.sqrt(R @ R)
            
            ###### # Line search Solution
            if self.config['line_search_iters'] > 0:
                alpha_ls,sol_ls = self.line_search(fun, X, R, deltaX)
                
                deltaX = alpha_ls * deltaX
            
            ###### # Update Solution
            X = X + deltaX

            ###### # Tolerance Checking
            
            if i == 0: # Store initial tolerances
                r0 = r_curr
                u0 = u_curr
                e0 = np.inf
                e_curr = np.inf
                
                rlist[0] = r0
                ulist[0] = u0
            if i == 1: 
                # Now have evaluated the residual so can give an initial e0
                e0 = np.abs(e_curr)
                
            elist[i] = np.sqrt(np.abs(e_curr))
            rlist[i] = r_curr
            ulist[i] = u_curr
            
            if verbose:
                if i == 0:
                    print('Iter &     |R|     &  |e_(i-1)|  &     |dU|    '\
                              + '&  |R|/|R0|   &  |e|/|e0|   &  |dU|/|dU0| ' \
                              +'&  Rate R &  Rate E &   Rate U '\
                              +'& NR/BFGS')
                
                if i >= 2:
                    # import pdb; pdb.set_trace()
                    
                    rate_r = np.log(rlist[i-1] / rlist[i]) / np.log(rlist[i-2] / rlist[i-1])
                    rate_u = np.log(ulist[i-1] / ulist[i]) / np.log(ulist[i-2] / ulist[i-1])
                    
                if i >= 3:
                    rate_e = np.log(elist[i-1] / elist[i]) / np.log(elist[i-2] / elist[i-1])
                    
                print(form.format(i, r_curr, e_curr, u_curr, 
                              r_curr/r0, e_curr/e0, u_curr/u0,
                              rate_r, rate_e, rate_u,
                              curr_iter))
            
            # Check for final convergence
            converged = _check_convg(self.config['stopping_tol'], self.config, 
                                     r_curr, e_curr, u_curr, 
                                     r_curr/r0, e_curr/e0, u_curr/u0)
            if converged:
                if verbose:
                    print('Converged!')
                sol['message'] = 'Converged'
                sol['success'] = True
                
                break
        
            ###### Setup next loop iteration
            
            # Increment and potentially reset BFGS counter
            bfgs_ind = (bfgs_ind + 1) % self.config['reform_freq']
            
            # Save R from this step since it is needed for BFGS
            Rminus1 = R
            
            # Save deltaX from this step since it is needed for calculating
            # energy norm for outputs 
            # (if energy norm goes negative, line search is useful)
            deltaXminus1 = deltaX
            
        ##########################################################
        # Final Clean Up and Return
        
        if no_nan_vals and not sol['success']:
            
            # Check convergence against the second set of tolerances
            converged = _check_convg(self.config['accepting_tol'], self.config, 
                                 r_curr, e_curr, u_curr, 
                                 r_curr/r0, e_curr/e0, u_curr/u0)
            
            if converged:
                
                if verbose:
                    print('Converged on accepting tolerances at max_iter.')
                    
                sol['message'] = 'Converged'
                sol['success'] = True

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
        List of tolerances to be checked. See NonlinearSolverOMP
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