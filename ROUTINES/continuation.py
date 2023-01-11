import numpy as np
# from solvers import NonlinearSolver
from scipy.linalg import svd

class Continuation:
    """
    Terminology:
        X - General vector of unknowns
        lam - (lambda), control variable that continuation is following 
                (e.g., amplitude for EPMC, frequency for HBM)
        C - variables in conditioned space, should all be Order(1). 
                Solutions are calculated in this space
        P - variables in physical space - these are the values one is interested in.
        fun - function for evaluations, all are done using physical coordinates 
                and conditioning is handled in this class.
    """
    
    def __init__(self, solver, ds0=0.01, CtoP=None, RPtoC=None, config={}):
        """
        Initialize Continuation Parameters

        Parameters
        ----------
        solver : an object of type NonlinearSolver that will be used to do 
                    nonlinear solutions
        ds0    : Scalar, size of first step
        Dscale : TYPE, optional
            DESCRIPTION. The default is 1.
        config : Dictionary of settings:
                    FracLam : Fraction of importance of lamda in arclength. 
                        1=lambda control, 0='displacement (X)' control, default=0.5
                    dsmax : maximum step size, default 5*ds0
                    dsmin : minimum step size, default ds0/5
                    MaxSteps : Maximum number of allowed solution points in the continuation
                    TargetFeval : Target number of function evaluations for each step
                                    Used to adaptively adjust step size.
                    predMask : optionally pass in a list of values of FracLam.
                               These values will be tried if the initial value 
                               of 'FracLam' fails to converge. The class always 
                               starts with the value passed in for 'FracLam' 
                               before considering this list.
                    backtrackStop : if continuation starts backtracking by 
                                     more than this amount past the start value 
                                     it will end before taking the maximum 
                                     number of steps. Has not been fully tested.
                    
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
            self.CtoP = CtoP
            
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
                        'backtrackStop': np.inf # Limit in how much backtracking past the start is allowed.
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
    
    def correct_res(self, fun, XlamC, XlamC0, ds, dirC=None):
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
        
        R, dRdXP, dRdlamP = fun(XlamP)
        
        dRdXlamC = np.hstack((dRdXP*self.CtoP[:-1], np.atleast_2d(dRdlamP).T*self.CtoP[-1]))
        
        
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
        dRaugdXlamC = np.vstack((self.RPtoC*dRdXlamC, dRarcdXlamC))

        return Raug, dRaugdXlamC
    
    def continuation(self, fun, XlamP0, lam0, lam1):
        """
        Function runs a continuation from lam0 to lam1 where lam is the last 
        entry of the unknowns.

        Parameters
        ----------
        fun : Residual function which takes as input XlamP (N+1,) and returns:
                R (N,), dRdXP (N,N), dRdlamP (N,)
        XlamP0 : Initial Guess (Physical Coordinates)
            DESCRIPTION.
        lam0 : Scalar, starting value of lambda
        lam1 : Scalar, final value of lambda

        Returns
        -------
        XlamP_full : Final history, rows are individual entries of XlamP 
                     (physical coordinates)

        """
        
        XlamP_full = np.zeros((self.config['MaxSteps'], XlamP0.shape[0]))        
        
        # Solve at Initial Point
        print('Starting Continuation from ', lam0, ' to ', lam1)
        
        # No continuation, fixed at initial lam0
        fun0 = lambda X : fun( np.hstack((X, lam0)) )[0:2]
        
        X, R, dRdX, sol = self.solver.nsolve(fun0, XlamP0[:-1], \
                                             xtol=self.config['xtol'], \
                                             verbose=self.config['verbose'])
        
        assert sol['success'], 'Failed to converge to initial point, give a better initial guess.'
        
        print('Converged to initial point! Starting continuation.')
        
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
                correct_fun = lambda XlamC : self.correct_res(fun, XlamC, XlamP0/self.CtoP, ds, dirC)
                
                XlamC, R, dRdX, sol = self.solver.nsolve(correct_fun, \
                                                         XlamP0/self.CtoP + dirC*ds,\
                                                         xtol=self.config['xtol'],\
                                                         verbose=False)
                
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
                                                             verbose=False)
            
                # Break out of loop over masks if have converged
                if sol['success']:
                    if fracLam_ind > 0 and self.config['verbose']:
                        print('Succeeded with FracLam index {} with value FracLam={}.'\
                              .format(fracLam_ind, self.config['FracLam']))
                    break
                
                
            if(not sol['success']):
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
            
            # Update information from previous steps
            XlamPprev = np.copy(XlamP0)
            XlamP0 = np.copy(XlamP_full[step])
            step += 1
            
        # Only return solved history.
        XlamP_full = XlamP_full[:step]
        
        print('Continuation complete, at lam=', XlamP_full[step-1, -1])
        
        return XlamP_full
    