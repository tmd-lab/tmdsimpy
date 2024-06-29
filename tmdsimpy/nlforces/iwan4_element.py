import numpy as np
from .nonlinear_force import HystereticForce

# Harmonic Functions for AFT
from .. import harmonic_utils as hutils


class Iwan4Force(HystereticForce):
    """
    Implementation of the 4-parameter Iwan model for hysteresis in joints.
    
    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system 
        to the `Nnl` local nonlinear DOFs.
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    kt : float
        Tangential stiffness coefficient
    Fs : float
        Slip force
    chi : float
        Controls microslip damping slope. Recommended to have `chi` > -1.
        Smaller values of `chi` may not work.
    beta : float, positive
        Controls discontinuity at beginning of macroslip (zero is smooth)
    Nsliders : int, optional
        Number of discrete sliders for the Iwan element. 
        Note that this does not include 1 additional slider for the 
        delta function at phimax.
        Default is 100 (commonly used in literature).
    alphasliders : float, optional
        determines the non-uniform discretization (see Segalman (2005))
        For midpoint rule, using anything other than 1.0 has 
        significantly higher error.
        The default is 1.0.
    
    Notes
    -----
    
    The 4-parameter Iwan model for jointed connections from [1]_.
    
    Not Verified for more than one input displacement to the Iwan model 
    (after Q mapping)
    
    Implementation absorbs kt into the probability distribution. Sliders start 
    to slip at displacement phi.
    
    Here sliders are represented with the formulation :
    
    >>> fstuck = (u - up) + fp
    ... 
    ... if fstuck > phi: # stuck
    ...     force = fstuck
    ... else:  # Slipping
    ...     force = phi
    
    with the real slider force being kt*f at each instant. f and fp therefore 
    have units of displacement not force.
    
    Quadrature points based on [1]_, but with some modification so 
    that the quadrature weights are independent of stick/slip for the sake of 
    computational efficiency
    
    References
    ----------
    .. [1] 
       Segalman, D.J., 2005. A Four-Parameter Iwan Model for Lap-Type 
       Joints. J. Appl. Mech 72, 752–760.
    
    """
    
    def __init__(self, Q, T, kt, Fs, chi, beta, Nsliders=100, alphasliders=1.0):
        
        self.Q = Q
        self.T = T
        self.kt = kt*1.0
        self.Fs = Fs*1.0
        self.chi = chi*1.0
        self.beta = beta*1.0
        self.Nsliders = Nsliders
        
        self.phimax = self.Fs * (1 + self.beta) / self.kt \
                        / (self.beta + (self.chi + 1)/(self.chi+2))
        
        self.R = self.Fs*(self.chi + 1) / self.phimax**(self.chi+2)\
                        / (self.beta + (self.chi + 1)/(self.chi+2))
                        
        self.S = self.Fs*self.beta / self.phimax\
                        / (self.beta + (self.chi + 1)/(self.chi+2))
        
        # self.S = self.Fs / self.phimax *(self.beta / (self.beta + (self.chi+1)/(self.chi+2)))
                        
        if(alphasliders > 1):
            deltaphi1 = self.phimax * (alphasliders - 1) / (alphasliders**(Nsliders) - 1)
        else:
            deltaphi1 = self.phimax / Nsliders
        
        delta_phis = deltaphi1 * alphasliders**np.array(range(Nsliders))
        
        self.phisliders = np.concatenate( (np.cumsum(delta_phis) - delta_phis*0.5,\
                                          np.array([self.phimax])) )
        
        # Segalman, 2005, eqn 25
        R = self.Fs*(self.chi + 1) / (self.phimax**(self.chi+2) \
                                     *(self.beta + (self.chi + 1)/(self.chi + 2)))
        
        # Segalman, 2005, eqn 26
        S = self.Fs*self.beta / (self.phimax \
                                   *(self.beta + (self.chi + 1)/(self.chi + 2)))
        
        # These absorb kt
        self.sliderweights = np.concatenate( \
                             (R * self.phisliders[:-1]**(self.chi)*delta_phis, \
                              np.atleast_1d(S)) )
        
        assert self.Q.shape[0] == 1, 'Not tested for simultaneous Iwan elements.'
        
        self.init_history()
        
    
    def set_prestress_mu(self):
        """
        Set friction coefficient to a different value (generally 0.0) for
        prestress analysis
        """
        assert False, 'Prestress mu is not implemented for Iwan Element.'
        
    def init_history(self):
        self.up = 0
        self.fp = 0
        self.fpsliders = np.zeros((self.Nsliders+1)) # Slider at the delta is not counted in Nsliders
        
        return
        
    def init_history_harmonic(self, unlth0, h=np.array([0])):
        """
        Initialize History

        Parameters
        ----------
        unlt0 : 0th Harmonic Displacement as a reference configuration. 
                Not required to use as slider reference, but makes a good 
                invariant choice.
        h : List of harmonics. Only use default if not interested in harmonic 
            information and derivatives.
            The default is [0].

        Returns
        -------
        None.

        """
                
        self.up = unlth0
        self.fp = 0
        self.fpsliders = np.zeros((self.Nsliders+1)) # Slider at the delta is not counted in Nsliders
        self.dupduh = np.zeros((hutils.Nhc(h)))
        
        self.dupduh[0] = 1 # Base slider position taken as zeroth harmonic 
        
        self.dfpduh = np.zeros((1,hutils.Nhc(h)))
        self.dfpslidersduh = np.zeros((self.Nsliders+1,hutils.Nhc(h)))
        
        return
    
    def force(self, X, update_hist=False):
        """
        To use the Iwan element with a static analysis, this force needs to be
        implemented. In addition a number of routines related to history variables
        are also required (e.g., see Jenkins)
        """
        
        unl = self.Q @ X
        
        fnl, dfnldunl, dfnlsliders_dunl = self.instant_force(unl, np.zeros_like(unl), update_prev=update_hist)
        
        fnl = np.atleast_1d(fnl)
        dfnldunl = np.atleast_2d(dfnldunl)
            
        F = self.T @ fnl
        
        dFdX = self.T @ dfnldunl @ self.Q
        
        return F, dFdX
        
    
    def instant_force(self, unl, unldot, update_prev=False):
        """
        Gives the force for a given loading instant using the previous history
        Then updates the stored history. 

        Parameters
        ----------
        unl : Displacements
        unldot : Velocities

        Returns
        -------
        fnl : Nonlinear Forces
        dfnldunl : derivative of forces w.r.t. current displacements
        dfnldup : derivative of forces w.r.t. previous displacements
        dfnldfp : derivative of forces w.r.t. previous forces

        """
        
        # Stuck Force
        fnlsliders = unl - self.up + self.fpsliders

        # Mask of stuck sliders == places with unit derivative
        dfnlsliders_dunl = np.less_equal(np.abs(fnlsliders), self.phisliders)
        
        fnlsliders[np.logical_not(dfnlsliders_dunl)] \
                            = self.phisliders[np.logical_not(dfnlsliders_dunl)]\
                                *np.sign(fnlsliders[np.logical_not(dfnlsliders_dunl)])

        # # Additional derivative information does not need to be output:
        # dfnlsliders_dup = -dfnlsliders_dunl
        # dfnlsliders_dfp = dfnlsliders_dunl
        
        # Integration
        fnl = fnlsliders @ self.sliderweights
        dfnldunl = dfnlsliders_dunl @ self.sliderweights
        
        if update_prev:
            # Update History
            self.up = unl
            self.fp = fnl
            self.fpsliders = fnlsliders
        
        return fnl, dfnldunl, dfnlsliders_dunl
    
    def instant_force_harmonic(self, unl, unldot, h, cst, update_prev=False):
        """
        For evaluating a force state, uses history initialized in init_history_harmonic.
        Updates history for the next call based on the current results. 
                
        Parameters
        ----------
        unl : Local displacements for Force
        unltdot : Local velocities for Force
        
        Returns
        -------
        fnl : Local nonlinear forces (Ndnl, Ndnl)
        dfduh : Derivative of forces w.r.t. displacement harmonic coefficients (Ndnl, Ndnl, Nhc)
        dfdudh : Derivative of forces w.r.t. velocities harmonic coefficients (Ndnl, Ndnl, Nhc)

        """
        
        # Number of nonlinear DOFs
        Nhc = hutils.Nhc(h)
        
        dfduh = np.zeros((Nhc))
        # dfdudh = np.zeros((Nhc))
        
        dfslidersduh = np.zeros((Nhc))
        
        fnl, dfnldunl, dfnlsliders_dunl = self.instant_force(unl, unldot, update_prev=update_prev)
        
        fnl = np.atleast_1d(fnl)
        
        dfnlsliders_duh = np.einsum('i,j->ij', dfnlsliders_dunl, cst-self.dupduh) \
                + dfnlsliders_dunl.reshape(-1,1)*self.dfpslidersduh # this line is dfnlsliders_dfslidersp*...
        
        dfduh = np.einsum('ij,i->j', dfnlsliders_duh, self.sliderweights) 
        
        dfduh = dfduh.reshape((1,1,-1))
        dfdudh = np.zeros_like(dfduh)
        
        # Save derivatives into history for next call. 
        self.dupduh = cst
        self.dfpduh = dfduh
        self.dfpslidersduh = dfnlsliders_duh 
                
        return fnl, dfduh, dfdudh
    
    
    
    
    