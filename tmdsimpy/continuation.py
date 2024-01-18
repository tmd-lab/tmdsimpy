import numpy as np
from scipy.linalg import svd

class Continuation:
    """
    Parameters
    ----------
    solver : tmdsimpy.solvers.NonlinearSolver or similar 
        Object with routines for linear and nonlinear solutions. 
    ds0    : float, optional
        Size of first step
        The default is 0.01.
    CtoP : 1D numpy.ndarray, optional
        Scaling vector to convert from conditioned space Xc to physical 
        coordinates Xp as Xp = CtoP*Xc. If None, the vector is set on the 
        first continuation call to be numpy.ones(Xp.shape[0]). Corresponding
        to no conditioning (though dynamic conditioning may still apply).
        The default is None.
    RPtoC: 1D numpy.ndarray or None
        This is a conditioning vector applied to scale the residual.
        If None, the vector defaults to 1. Dynamic conditioning may still apply
        The default is None.
    config : Dictionary of settings, optional.
                FracLam : float, optional
                    Fraction of importance of lamda in arclength. 
                    1=lambda control, 0='displacement (X)' control, 
                    The default is 0.5.
                dsmax : float, optional
                    Maximum step size.
                    The default is 5*ds0.
                dsmin : float, optional
                    Minimum step size.
                    The default is ds0/5.
                MaxSteps : int, optional
                    Maximum number of allowed solution points in the 
                    continuation.
                    The default is 500.
                TargetFeval : int, optional
                    Target number of function evaluations for each step
                    used to adaptively adjust step size.
                    The default is 20.
                DynamicCtoP : bool, optional
                    If True, the CtoP vector is dynamically updated for each 
                    continuation step. The initial value of CtoP is used as 
                    a minimum value of CtoP for any step, but CtoP can increase
                    for some variables (each element independently). 
                    Dynamic conditioning is also applied to the residual vector
                    (RPtoC).
                    The default is False.
                verbose : int, optional
                    Number of steps to output updates at. 
                    If less than 0, all output is supressed. 
                    If 0, some output is still printed. 
                    The default is 100.
                xtol : float or None, optional
                    This tolerance is passed to the solver as
                    solver.nsolve(xtol=xtol)
                    The default is None.
                corrector : {'Ortho', 'Psuedo'}, optional
                    Option for the continuation constraint equation. 
                    Ortho requires that the correction be orthogonal to the 
                    prediction. 
                    Pseudo requires that a norm (in conditioned space)
                    of the solution minus the previous solution be a fixed
                    value.
                    The default is Ortho. 
                FracLamList : list, optional
                    List of FracLam values to try if the initial value of 
                    FracLam fails at a step. If FracLam is the first value in
                    this list, then the first value of this list is ignored.
                    The default is [].
                backtrackStop : float, optional
                    If continuation starts backtracking by 
                    more than this amount past the start value 
                    it will end before taking the maximum 
                    number of steps. Has not been fully tested.
                    The default is numpy.inf.
                nsolve_verbose : int, optional
                    Setting passed to solver as
                    solver.nsolve(verbose=nsolve_verbose)
                    The default is False (0).
                callback : function or None, optional
                    Function that is called after every solution. 
                    function is passed arguments of X,dirP_prev
                    corresponding to the solution at the current
                    point and the prediction direction that was 
                    used to calculate that point. Function is 
                    called after initial solution with np.nan 
                    vector of dirP_prev and twice after the final
                    converged solution. The final call has np.nan
                    vector for X, and has dirP_prev if one was to 
                    take another step (correponds to slope at final
                    solution) for interpolation.
                    The default is None.
    
    Terminology
    ----------
        X : numpy.array
            General vector of unknowns
        lam : float
            (lambda), control variable that continuation is following 
                (e.g., amplitude for EPMC, frequency for HBM)
        C : char
            variables in conditioned space, should all be Order(1). 
                Solutions are calculated in this space
        P : char
            variables in physical space - these are the values one is interested in.
        fun : function
            function for evaluations, all are done using physical coordinates 
            and conditioning is handled in this class.
    """
    
    def __init__(self, solver, ds0=0.01, CtoP=None, RPtoC=None, config={}):
        """
        Initialize Continuation Parameters

        Returns
        -------
        None.

        """
        
        self.solver = solver
        
        if CtoP is None:
            self.setCtoPto1 = True
        else:
            assert len(CtoP.shape) == 1, 'Conditioning vector is expected to be 1D'
            self.setCtoPto1 = False
            self.CtoP = np.abs(CtoP)
            
        if RPtoC is None:
            self.RPtoC = 1
        else:
            self.RPtoC = RPtoC
            
        default_config={'FracLam' : 0.5, 
                        'ds0' : ds0,
                        'dsmax' : 5*ds0, 
                        'dsmin' : ds0/5,
                        'MaxSteps' : 500,
                        'TargetNfev': 20, 
                        'DynamicCtoP': False,
                        'verbose' : 100, # Print every 100 Steps
                        'xtol'    : None, 
                        'corrector': 'Ortho', # Psuedo or Ortho
                        'FracLamList' : [], # List of vectors/numbers to multiply predictor by
                        'backtrackStop': np.inf, # Limit in how much backtracking past the start is allowed.
                        'nsolve_verbose' : False,
                        'callback' : None
                        }
        
        
        for key in config.keys():
            default_config[key] = config[key]
            
        # Make sure to always start with the value of 'FracLam' that is passed 
        # in before proceeding to the list of other possible values. 
        if len(default_config['FracLamList']) == 0 \
            or not (default_config['FracLamList'][0] == default_config['FracLam']):
            
            default_config['FracLamList'].insert(0, default_config['FracLam'])
            
        self.config = default_config
        
        
    def predict(self, fun, XlamP0, XlamPprev):
        """
        Predicts the direction of the next step with the correct sign and ds=1
        
        If multiple FracLam values are used in the case of nonconvergence, 
        then this function is repeatedly called, but those later calls should
        not have to re-evaluate the residual and re-find the null space since
        it has not changed. Therefore, this could be sped up by eliminating
        that work on repeat calls at the same XlamP0 value.

        Parameters
        ----------
        fun : Function that continuation is following
        XlamP0 : 1D numpy array of [physical coordinates, lambda]. Previous 
                 solution, so start of next step.
        XlamPprev : The start of the previous step (step before XlamP0)

        Returns
        -------
        dirC : Direction vector scaled to be a step size of ds = 1

        """
        
        R, dRdXP, dRdlamP = fun(XlamP0)
        
        # Conditioned space, N x N+1 matrix.
        dRdXlamC = np.hstack((dRdXP*self.CtoP[:-1], np.atleast_2d(dRdlamP).T*self.CtoP[-1]))
        
        # Null-Space Corresponds to where the fun equations are still satisfied,
        # and the distance can change by allowing motion.
        U,s,Vh = svd(dRdXlamC, overwrite_a=True)
        
        # Direction in conditioned space of the next step
        dirC = Vh[-1, :]
        
        # Arc Length Weighting Parameters
        b = self.config['FracLam']
        XC0 = XlamP0[:-1] / self.CtoP[:-1]
        c = (1-b) / np.linalg.norm(XC0)**2 # could store to eliminate an O(N) calculation each iteration. 
        
        # Scale Direction so that it takes a step size of ds=1
        step_sq = c*np.linalg.norm(dirC[:-1])**2 + b*dirC[-1]**2
        
        dirC = dirC / np.sqrt(step_sq)
        
        # Set the sign to be the correct direction based on the previous step
        # Use the same inner product space as the arclength to check the sign
        
        dXlamCprev = (XlamP0 - XlamPprev) / self.CtoP # Comparing in the current conditioned space
        
        signarg = c*dirC[:-1] @ dXlamCprev[:-1] + b*dirC[-1]*dXlamCprev[-1]
        
        sign = np.sign(signarg)
        
        if sign == 0:
            sign = 1 # choose direction arbitrarily if perfectly orthogonal
        
        dirC = dirC * sign
        
        # Dynamic Scaling of Residual Vector
        if self.config['DynamicCtoP']:
            diagdRdX = np.diag(dRdXlamC)
            self.RPtoC = 1/np.max(np.abs(diagdRdX[:-1]))
        
        return dirC
        
    def psuedo_arc_res(self, XlamC, XlamC0, ds, b, c):
        
        # Step Sizes in conditioned space
        dlamC = XlamC[-1] - XlamC0[-1]
        dXC   = XlamC[:-1] - XlamC0[:-1]
        
        #dstep_sq - step size squared
        dstep_sq = c*np.linalg.norm(dXC)**2 + b*dlamC**2
        
        dstep_sq_dXC   = 2*c*dXC
        dstep_sq_dlamC = 2*b*dlamC
        
        Rarc =  (dstep_sq - ds**2)/ds**2
        
        dRarcdXlamC = np.hstack((dstep_sq_dXC,dstep_sq_dlamC))/ds**2
        
        return Rarc, dRarcdXlamC
    
    def orthogonal_arc_res(self, XlamC, XlamC0, dirC, ds, b, c):
        
        # Current Point minus Position of the predictor
        dXlamC = XlamC - (XlamC0 + dirC*ds)
        
        # 1. Dot product is in the same inner product form as the arc length 
        # predictor
        # 2. If was at point 2*ds*dirC, dXlamC=ds*dirC, 
        # Inner product of dirC with itself is 1. Divide by ds so this case 
        # would have residual O(1)
        Rarc = (c*(dXlamC[:-1] @ dirC[:-1]) + b*dXlamC[-1]*dirC[-1])/ds
        
        dRarcdXlamC = np.hstack((c*dirC[:-1], b*dirC[-1]))/ds
        
        return Rarc, dRarcdXlamC
    
    def correct_res(self, fun, XlamC, XlamC0, ds, dirC=None, calc_grad=True):
        """
        Corrector Residual

        Parameters
        ----------
        fun : Function describing the N unknowns in X.
        XlamC : Test solution at the current point to evaluate residual at.
        XlamC0 : Solution at the previous point in conditioned space
        ds : Current Arc length step size
        dirC : Direction of the predictor step, only needed for orthogonal corrector

        Returns
        -------
        Raug : Residual vector with augmented equation for the arc length constraint.
        dRaugdXlamC : Gradient in conditioned space for the augmented residual.

        """
        XlamP = XlamC * self.CtoP
        
        if calc_grad:
            R, dRdXP, dRdlamP = fun(XlamP)
            
            dRdXlamC = np.hstack((dRdXP*self.CtoP[:-1], 
                                  np.atleast_2d(dRdlamP).T*self.CtoP[-1]))
        
        else:
            # No Gradient Calculation
            R = fun(XlamP, calc_grad)[0]
        
        
        # Relative Weighting of variables
        b = self.config['FracLam']
        c = (1-b) / np.linalg.norm(XlamC0[:-1])**2 # could store to eliminate an O(N) calculation each iteration. 
        
        if self.config['corrector'].upper() == 'PSEUDO':
            Rarc, dRarcdXlamC = self.psuedo_arc_res(XlamC, XlamC0, ds, b, c)
        elif self.config['corrector'].upper() == 'ORTHO':
            assert not (dirC is None), 'In proper call, need dirC for ortho corrector.'
            Rarc, dRarcdXlamC = self.orthogonal_arc_res(XlamC, XlamC0, dirC, ds, b, c)
        else:
            assert False, 'Invalid corrector type: {}'.format(self.config['corrector'].upper())
        
        # Augment R and dRdXlamC with the arc length equation
        Raug = np.hstack((self.RPtoC*R, Rarc))
        
        if calc_grad:
            dRaugdXlamC = np.vstack((self.RPtoC*dRdXlamC, dRarcdXlamC))
    
            return Raug, dRaugdXlamC
        else:
            return (Raug,)
    
    def continuation(self, fun, XlamP0, lam0, lam1):
        """
        Function runs a continuation from lam0 to lam1 where lam is the last 
        entry of the unknowns.

        Parameters
        ----------
        fun : function
            Residual function which takes as input XlamP (N+1,) and returns:
            R (N,), dRdXP (N,N), dRdlamP (N,) (inputs/outputs are 
            numpy.ndarray).
            Function may need to have an optional argument calc_grad=True
            if the function will be used with a nonlinear solver that requires
            two input arguments. e.g. 
            'fun = lambda Xlam, calc_grad=True : residual(X, calc_grad)'.
            When calc_grad is False, the function should return a tuple with 
            the first entry of the tuple being R, the other entries may be
            ignored.
            By default, it is assumed that fun only takes one input. If a 
            wrapper function for continuation receives calc_grad=False, then
            it is assumed that fun will accept a second bool input.
        XlamP0 : (N+1,) numpy.ndarray
            Initial Guess (Physical Coordinates) for the solution at lam0.
            The N+1 entry is ignored.
        lam0 : float
            starting value of lambda (continuation parameter)
        lam1 : float
            final value of lambda

        Returns
        -------
        XlamP_full : (M, N+1) numpy.ndarray
            Final history, rows are individual entries of XlamP 
            (physical coordinates), and M steps are taken.

        """
        
        # Check about removing all output
        silent = self.config['verbose'] < 0
        
        if silent:
            self.config['verbose'] = 0
        
        # Initialize Memory
        XlamP_full = np.zeros((self.config['MaxSteps'], XlamP0.shape[0]))        
        
        # Solve at Initial Point
        if not silent:
            print('Starting Continuation from ', lam0, ' to ', lam1)
        
        # No continuation, fixed at initial lam0
        # Not sure if fun accepts calc_grad, so will always calculate the gradient
        fun0 = lambda X, calc_grad=True : fun( np.hstack((X, lam0)) )[0:2]
        
        X, R, dRdX, sol = self.solver.nsolve(fun0, XlamP0[:-1], \
                                             xtol=self.config['xtol'], \
                                             verbose=self.config['nsolve_verbose'])
                
        assert sol['success'], 'Failed to converge to initial point, give a better initial guess.'
        
        if not silent:
            print('Converged to initial point! Starting continuation.')
        
        if self.config['callback'] is not None:
            # Callback save of initial solution
            # dirC is not yet calculated, so passing NaN
            self.config['callback'](np.hstack((X, lam0)), 
                                    np.hstack((X, lam0))*np.nan)
                
        # Define a Reference Direction as a previous solution for use in the 
        # predictor
        direct = np.sign(lam1 - lam0)
        XlamPprev = np.hstack((X, lam0 - direct))
        
        step = 0
        XlamP0 = np.hstack((X, lam0))
        XlamP_full[step] = XlamP0
        
        step += 1
        
        # Conditioning
        if self.setCtoPto1:
            self.CtoP = np.ones_like(XlamP0)
            
        if self.config['DynamicCtoP']:
            self.CtoP0 = np.copy(self.CtoP)
            
        ds = self.config['ds0']
        
        while step < self.config['MaxSteps'] \
            and direct*XlamP_full[step-1,-1] < direct*lam1 \
            and direct*XlamP_full[step-1,-1] > direct*(lam0-direct*self.config['backtrackStop']):
            
            # Update Conditioning Dynamically
            if self.config['DynamicCtoP']:
                self.CtoP = np.maximum(np.abs(XlamP_full[step-1]), self.CtoP0)
                
            for fracLam_ind in range(len(self.config['FracLamList'])):
                
                # Select the current value of weighting lambda v. other variables
                self.config['FracLam'] = self.config['FracLamList'][fracLam_ind]
                
                # Predict Direction
                dirC = self.predict(fun, XlamP0, XlamPprev)
                                
                # Correct
                correct_fun = lambda XlamC, calc_grad=True : \
                        self.correct_res(fun, XlamC, XlamP0/self.CtoP, 
                                         ds, dirC, calc_grad=calc_grad)
                
                XlamC, R, dRdX, sol = self.solver.nsolve(correct_fun, \
                                                         XlamP0/self.CtoP + dirC*ds,\
                                                         xtol=self.config['xtol'],\
                                                         verbose=self.config['nsolve_verbose'])
                
                # Retry with smaller steps if correction failed.
                while (not sol['success']) and ds > self.config['dsmin']:
                    ds = max(ds / 2, self.config['dsmin'])
                    
                    if self.config['verbose']:
                        print(sol['message'])
                        print('Failed to converge with ds=', 2*ds, '. Retrying with ds=', ds)
                        # print('norm(R)=', np.linalg.norm(R))
                    
                    # Correct Again
                    XlamC, R, dRdX, sol = self.solver.nsolve(correct_fun, \
                                                             XlamP0/self.CtoP + dirC*ds,\
                                                             xtol=self.config['xtol'],\
                                                             verbose=self.config['nsolve_verbose'])
            
                # Break out of loop over masks if have converged
                if sol['success']:
                    if fracLam_ind > 0 and self.config['verbose']:
                        print('Succeeded with FracLam index {} with value FracLam={}.'\
                              .format(fracLam_ind, self.config['FracLam']))
                    break
                
                
            if(not sol['success'] and not silent):
                print('Stopping since final solution failed to converge.')
                break
            
            # Store Iteration and Advance
            XlamP_full[step] = self.CtoP * XlamC
            
            # Debug check with if statement in case it accidently starts going 
            # backwards
            # if XlamP_full[step, -1] < XlamP_full[step-1, -1]:
            #     print('Started Backtracking')
            #     dirC = self.predict(fun, XlamP0, XlamPprev)
            #     pass
            
            if self.config['verbose'] and step % self.config['verbose'] == 0:
                print('Step=', step, ' converged: lam=', XlamP_full[step, -1], \
                      ' ds=', ds, ' and nfev=', sol['nfev'])
            
            # Heuristic For updating ds
            ds = ds * self.config['TargetNfev'] / sol['nfev']
            
            ds = min(max(ds, self.config['dsmin']), self.config['dsmax'])
            
            # TODO: Callback function
            if self.config['callback'] is not None:
                self.config['callback'](XlamP_full[step], dirC*self.CtoP)
            
            # Update information from previous steps
            XlamPprev = np.copy(XlamP0)
            XlamP0 = np.copy(XlamP_full[step])
            step += 1
            
        # Only return solved history.
        XlamP_full = XlamP_full[:step]
        
        # Callback - save the final dirC 
        if self.config['callback'] is not None:
            if not sol['success']:
                # Pass dirC for the previous step corresponding to the last
                # converged solution
                self.config['callback'](dirC*np.nan, dirC*self.CtoP)
            else:
                # Calculate dirC for the current solution to be saved
                dirC = self.predict(fun, XlamP0, XlamPprev)
                
                self.config['callback'](dirC*np.nan, dirC*self.CtoP)
        
        if not silent:
            print('Continuation complete, at lam=', XlamP_full[step-1, -1])
        
        return XlamP_full
    
