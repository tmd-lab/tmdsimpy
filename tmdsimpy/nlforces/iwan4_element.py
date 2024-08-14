import numpy as np
from .nonlinear_force import HystereticForce

# Harmonic Functions for AFT
from ..utils import harmonic as hutils


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
        Tangential stiffness coefficient.
    Fs : float
        Slip force.
    chi : float
        Controls microslip damping slope. Recommended to have `chi > -1`.
        Smaller values of `chi` may not work.
    beta : float, positive
        Controls discontinuity at beginning of macroslip (zero is smooth).
    Nsliders : int, optional
        Number of discrete sliders for the Iwan element. 
        Note that this does not include 1 additional slider for the 
        delta function at phimax.
        Default is 100 (commonly used in literature).
    alphasliders : float, optional
        Determines the non-uniform discretization (see [1]_).
        For midpoint rule, using anything other than 1.0 has 
        significantly higher error.
        The default is 1.0.
    
    See Also
    --------
    VectorIwan4 :
        Implementation that uses a more efficient vectorization of local forces
        for AFT calculation.
    
    Notes
    -----
    
    The 4-parameter Iwan model for jointed connections from [1]_.
    
    Not Verified for more than one input displacement to the Iwan model 
    (after `Q` mapping, i.e., `Nnl==1`). Functionality is not implemented for
    `Nnl > 1`.
    
    Implementation absorbs `kt` into the probability distribution. Sliders start 
    to slip at displacement `phi`.
    
    Here sliders are represented with the formulation :
    
    >>> fstuck = (u - up) + fp
    ... 
    ... if fstuck > phi: # stuck
    ...     force = fstuck
    ... else:  # Slipping
    ...     force = phi
    
    with the real slider force being `kt*f` at each instant. `f` and `fp`
    therefore 
    have units of displacement not force.
    
    Quadrature points based on [1]_, but with some modification so 
    that the quadrature weights are independent of stick/slip for the sake of 
    computational efficiency.
    
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
        Not implemented for Iwan element.
        
        Returns
        -------
        None
        
        Notes
        -----
        Intention is to 
        set friction coefficient to zero while saving initial value in a 
        different variable. Useful for prestress analysis.
        
        This is non-trivial for the Iwan implementation, so it is not
        yet implemented. One can simply not include the nonlinear force
        to get the same effect with the Iwan element.
        
        """
        
        assert False, 'Prestress mu is not implemented for Iwan Element.'
        
    def init_history(self):
        """
        Method to initialize history variables for the hysteretic model.
        
        This consists of setting previous displacements and forces
        to be zero.

        Returns
        -------
        None.

        """
        
        self.up = 0
        self.fp = 0
        self.fpsliders = np.zeros((self.Nsliders+1)) # Slider at the delta is not counted in Nsliders
        
        return
        
    def init_history_harmonic(self, unlth0, h=np.array([0])):
        """
        Initialize history variables for harmonic (AFT) analysis.

        Parameters
        ----------
        unlth0 : (Nnl,) numpy.ndarray
            Zeroth harmonic contributions to a time series of displacements.
            History displacements are initialized at this value.
        h : numpy.ndarray, sorted
            List of harmonics used in subsequent analysis.
            The default is `numpy.array([0])`.

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
        Calculate global nonlinear forces for some global displacement vector.

        Parameters
        ----------
        X : (N,) numpy.ndarray
            Global displacements
        update_hist : bool, optional
            Flag to save displacement and force from the evaluation as history
            variables for subsequent calls to this function.
            The default is False.

        Returns
        -------
        F : (N,) numpy.ndarray
            Global nonlinear force
        dFdX : (N,N) numpy.ndarray
            Derivative of `F` with respect to `X`.
        
        """
        
        unl = self.Q @ X
        
        fnl, dfnldunl, dfnlsliders_dunl = self.instant_force(unl, 
                                                    np.zeros_like(unl),
                                                    update_prev=update_hist)
        
        fnl = np.atleast_1d(fnl)
        dfnldunl = np.atleast_2d(dfnldunl)
            
        F = self.T @ fnl
        
        dFdX = self.T @ dfnldunl @ self.Q
        
        return F, dFdX
        
    
    def instant_force(self, unl, unldot, update_prev=False):
        """
        Calculates local force based on local nonlinear displacements.

        Parameters
        ----------
        unl : (Nnl,) numpy.ndarray
            Local nonlinear displacements to evaluate force at.
        unldot : (Nnl,) numpy.ndarray
            Local nonlinear velocities to evaluate force at.
        update_prev : bool, optional
            Flag to store the results of the evaluation for the start of the
            subsequent step. 
            The default is False.

        Returns
        -------
        fnl : float
            Evaluated local nonlinear forces.
        dfnldunl : float
            Derivative of `fnl` with respect to `unl`.
        dfnlsliders_dunl : (Nsliders+1,) numpy.ndarray
            Derivative of `fnl` at each slider with respect to `unl`.

        Notes
        -----
        
        Implementation only allows for a single nonlinear element, thus
        shapes of first two outputs are reduced to scalar.
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
        Evaluates the force at a instantaneous set of displacement and velocity
        along with harmonic derivatives.
        
        Parameters
        ----------
        unl : (Nnl,) numpy.ndarray
            Local nonlinear displacements to evaluate force at.
        unldot : (Nnl,) numpy.ndarray
            Local nonlinear velocities to evaluate force at.
        h : 1D numpy.ndarray, sorted
            List of harmonics used in subsequent analysis. Corresponds
            to `Nhc` harmonic components.
        cst : (Nhc,) numpy.ndarray
            Evaluation of harmonics without coefficients at the given instant 
            in time. 
            If zeroth harmonic is included, the first entry is 1.0. 
            Beyond that, it is cosine and then sine at the appropriate harmonic
            for the given instant in time then the next harmonic etc.
        update_prev : bool, optional
            Flag to store the results of the evaluation for the start of the
            subsequent step. 
            The default is False.
        
        Returns
        -------
        fnl : (1,) numpy.ndarray
            Local nonlinear forces
        dfduh : (1, 1, Nhc) numpy.ndarray
            Derivative of `fnl` with respect to displacement harmonic
            coefficients.
        dfdudh : (1, 1, Nhc) numpy.ndarray
            Derivative of `fnl` with respect to velocities harmonic
            coefficients.

        Notes
        -----
        
        Starts calculation based on `init_history_harmonic`.
        
        Only implemented for a single nonlinear element or `Nnl == 1`.
        
        """
        
        # Number of nonlinear DOFs
        Nhc = hutils.Nhc(h)
        
        dfduh = np.zeros((Nhc))
        # dfdudh = np.zeros((Nhc))
        # dfslidersduh = np.zeros((Nhc))
        
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
