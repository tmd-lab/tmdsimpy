import numpy as np
import scipy.optimize
import scipy.linalg
import warnings

from .solvers import NonlinearSolver


class NonlinearSolverOMP(NonlinearSolver):
    
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
        
        self.max_iter = 15
        
        pass
    
    
    def nsolve(self, fun, X0, verbose=True, xtol=None, max_iter=None, Dscale=1.0):
        
        if xtol is None:
            xtol = 1e-6*X0.shape[0]
            
        if max_iter is None: 
            max_iter = self.max_iter
            
            
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
                
        if verbose:
            print('Iter & |R|    & |e| & |dU| & |R|/|R0| & |e|/|e0| & |dU|/|dU0| ' \
                  +'Rate R & Rate E & Rate U \n')
            
        form =  '{:4d} & {: 6.4e} & {: 6.4e} & {: 6.4e} '\
                    + '& {: 6.4e} & {: 6.4e} & {: 6.4e}' \
                    + ' & {: 6.4f} & {:6.4f} & {:6.4f}'
                    
        rate_r = np.nan
        rate_e = np.nan
        rate_u = np.nan
        
        for i in range(max_iter):
            
            R, dRdX = fun(Xc*Dscale)
            
            dXc = np.linalg.solve( -(dRdX*Dscale), R)
            
            u_curr = np.sqrt(dXc @ dXc)
            e_curr = R @ dXc
            r_curr = np.sqrt(R @ R)
            
            elist[i+1] = np.sqrt(np.abs(e_curr))
            rlist[i+1] = r_curr
            ulist[i+1] = u_curr
            
            if i >= 1:
                # import pdb; pdb.set_trace()
                
                rate_r = np.log(rlist[i] / rlist[i+1]) / np.log(rlist[i-1] / rlist[i])
                rate_e = np.log(elist[i] / elist[i+1]) / np.log(elist[i-1] / elist[i])
                rate_u = np.log(ulist[i] / ulist[i+1]) / np.log(ulist[i-1] / ulist[i])
            
            if verbose:
                print(form.format(i, r_curr, e_curr, u_curr, 
                              r_curr/r0, np.abs(e_curr/e0), u_curr/u0,
                              rate_r, rate_e, rate_u))
            
            Xc += dXc
            
            if u_curr < xtol:
                
                if verbose:
                    print('Converged!')
                    
                sol['message'] = 'Converged'
                sol['nfev'] = i+2
                sol['njev'] = i+2
                sol['success'] = True
                    
                break
            
        X = Xc * Dscale
        
        # R, dRdX = fun(X)

        if(verbose):
            print(sol['message'], ' Nfev=', sol['nfev'], ' Njev=', sol['njev'])
        
        return X, R, dRdX, sol