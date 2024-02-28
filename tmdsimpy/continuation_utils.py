"""
Utility functions for continuation including functions to pass as inputs to 
save results during continuation etc.
"""

from os.path import exists
import numpy as np
from . import harmonic_utils as hutils

def combine_callback_funs(funs_list, XlamP, dirP_prev):
    """
    Calls a set of callback functions for continuation
    
    Parameters
    ----------
    funs_list : List of Functions
        List of callback functions that are called with arguments XlamP,dirP_prev
    XlamP : np.array
        Solution at the current step in physical coordinates (including 
        continuation parameter)
    dirP_prev : np.array
        Direction for prediction from previous step to this step. 
        Can be used to get slope of solution at previous point w.r.t. lam 
        by dividing dirP / dirP[-1]

    Returns
    -------
    None.

    """
    
    for fun in funs_list:
        fun(XlamP, dirP_prev)
        
    return

def continuation_save(XlamP, dirP_prev, fname):
    """
    Saves continuation variables. Saved output will 2D arrays with the same 
    names as inputs here. Each row of output will correspond to a single 
    solution point.

    Parameters
    ----------
    XlamP : np.array
        Solution.
    dirP_prev : np.array
        Prediction direction at previous solution.
    fname : String
        Filename to save variables to. Should have file extension of .npz

    Returns
    -------
    None.

    """
    
    if exists(fname):
        # Load previous solutions + Augment
        loaded = np.load(fname, allow_pickle=False)
        
        XlamP_full = np.vstack((loaded['XlamP'], XlamP.reshape(1,-1)))
        dirP_prev_full = np.vstack((loaded['dirP_prev'], dirP_prev.reshape(1,-1)))
        
    else: 
        # Rename for consistent save line
        XlamP_full = XlamP.reshape(1,-1)
        dirP_prev_full = dirP_prev.reshape(1,-1)
    
    # Save augmented solutions
    np.savez(fname, XlamP=XlamP_full, dirP_prev=dirP_prev_full)
    
    return

def print_epmc_stats(XlamP, dirP_prev, fname):
    """
    Saves EPMC key statistics to a text file for easy monitoring

    Parameters
    ----------
    XlamP : np.array
        Solution to EPMC continuation.
    dirP_prev : np.array
        Prediction direction at previous solution.
    fname : String
        Filename to save variables to.

    Returns
    -------
    None.

    """
    
    # If file doesn't exist, write a header
    write_header = not exists(fname)
    
    # Write current stats to file
    with open(fname, 'a') as file:
        
        if write_header:
            header_format = '{:^18s} & {:^18s} & {:^22s} \n'
            file.write(header_format.format('Amplitude (log)',
                                            'Frequency [Hz]',
                                            'Damping [Frac Crit]'))
            
        body_format = '{: ^18.3f} & {: ^18.3f} & {: ^22.3e} \n'
        
        Amp = XlamP[-1] # Log Amplitude
        freq = XlamP[-3] / 2 / np.pi # Frequency in Hz
        damp = XlamP[-2] / 2 / XlamP[-3] # Zeta (fraction critical damping)
        
        file.write(body_format.format(Amp, freq, damp))
    
    return

def print_hbm_amp_stats(XlamP, dirP_prev, fname, h, order,
                        output_recov, output_harmonic):
    """
    Saves HBM key statistics to a text file for easy monitoring
    This only works correctly for HBM with amplitude controlled.

    Parameters
    ----------
    XlamP : np.array
        Solution to EPMC continuation.
    dirP_prev : np.array
        Prediction direction at previous solution.
    fname : String
        Filename to save variables to.
    h : numpy.ndarray, sorted
        List of harmonics included in HBM
    order : int, zero or positive
        order of the derivative that is controlled. order=0 means 
        displacement output, order=2 means acceleration output
    output_recov: (Ndof,) numpy.ndarray
        Recovery vector to be used to output response at a specific DOF
        where Ndof is the number of degrees of freedom of the system. 
    output_harmonic : int
        Which harmonic the amplitude at the output_recov dof should be output 
        for. Behavior is undefined if output_harmonic is not included in h

    Returns
    -------
    None.

    """
    
    # If file doesn't exist, write a header
    write_header = not exists(fname)
    
    # Write current stats to file
    with open(fname, 'a') as file:
        
        if write_header:
            header_format = '{:^18s} & {:^18s} & {:^22s} \n'
            file.write(header_format.format('Frequency [Hz]',
                                            'Force Scaling [N]',
                                            'Recov Amp [m/s^{}]'.format(order)))
            
        body_format = '{: ^18.3f} & {: ^18.3f} & {: ^22.3e} \n'
        
        freq = XlamP[-1] / 2 / np.pi # Frequency in Hz
        force =  XlamP[-2] # Scaling factor (generally N)

        Ndof = output_recov.shape[0]
        Nhc_before = hutils.Nhc(h[h < output_harmonic])
        
        amp_cos = output_recov @ XlamP[Nhc_before*Ndof:(Nhc_before+1)*Ndof]
        amp_sin = output_recov @ XlamP[(Nhc_before+1)*Ndof:(Nhc_before+2)*Ndof]
        
        amp = np.sqrt(amp_cos**2 + amp_sin**2)*(XlamP[-1]**order)
        
        file.write(body_format.format(freq, force, amp))
    
    return