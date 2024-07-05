"""
Utility functions for continuation including callback functions.

See Also
--------
tmdsimpy.Continuation : 
    Class for solving continuation problems
tmdsimpy.postprocess.continuation :
    Module for post-processing functions from continuation solutions
"""

from os.path import exists
import numpy as np
from . import harmonic as hutils

def combine_callback_funs(funs_list, XlamP, dirP_prev):
    """
    Calls a set of callback functions for continuation.
    
    Parameters
    ----------
    funs_list : list of functions
        List of callback functions that are called with arguments 
        `XlamP,dirP_prev`.
    XlamP : (N+1,) numpy.ndarray
        Solution at the current step in physical coordinates (including 
        continuation parameter).
    dirP_prev : (N+1,) numpy.ndarray
        Direction for prediction from previous step to this step. 
        Can be used to get slope of solution at previous point w.r.t. lam 
        by dividing `dirP / dirP[-1]`

    Returns
    -------
    None.
    
    See Also
    --------
    tmdsimpy.Continuation : 
        Class for continuation where this function is intended to be used 
        as a callback function.
    
    Notes
    -----
    Use this function when you want to call multiple callback functions with 
    continuation as 
    
    >>> funs_list = [] # fill this in
    ...
    ... lambda XlamP, dirP_prev : combine_callback_funs(funs_list, 
    ...                                                 XlamP, dirP_prev)

    """
    
    for fun in funs_list:
        fun(XlamP, dirP_prev)
        
    return

def continuation_save(XlamP, dirP_prev, fname):
    """
    Saves continuation data to a file.
    
    Saves continuation variables. Saved output will 2D arrays with the same 
    names as inputs here. Each row of output will correspond to a single 
    solution point.

    Parameters
    ----------
    XlamP : (N+1,) numpy.ndarray
        Solution at the current step in physical coordinates (including 
        continuation parameter).
    dirP_prev : (N+1,) numpy.ndarray
        Direction for prediction from previous step to this step. 
        Can be used to get slope of solution at previous point w.r.t. lam 
        by dividing `dirP / dirP[-1]`
    fname : str
        Filename to save variables to. Should have file extension of .npz

    Returns
    -------
    None.

    See Also
    --------
    tmdsimpy.Continuation : 
        Class for continuation where this function is intended to be used 
        as a callback function.
    combine_callback_funs : 
        Function for combining multiple callback functions for `Continuation`.

    Examples
    --------
    Define a callback function handle to pass to `tmdsimpy.Continuation` as
    
    >>> callback_fun = lambda XlamP, dirP_prev : continuation_save(XlamP, 
    ...                                            dirP_prev, 'saved_data.npz')

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
    Saves EPMC key statistics to a text file for easy monitoring.
    
    This function is intended for monitoring active runs. For final solution
    details, save the full solution and plot based on those results, not based
    on the results written out by this summary file.

    Parameters
    ----------
    XlamP : (N+1,) numpy.ndarray
        Solution at the current step in physical coordinates (including 
        continuation parameter) to an EPMC set of equations.
        First entries are harmonic displacements of mass normalized mode, 
        `XlamP[-3]` is frequency (rad/s), 
        `XlamP[-2]` is mass proportional self excitation factor,
        `XlamP[-1]` is log10(modal amplitude).
    dirP_prev : (N+1,) numpy.ndarray
        Direction for prediction from previous step to this step. 
        Can be used to get slope of solution at previous point w.r.t. lam 
        by dividing `dirP / dirP[-1]`
    fname : str
        Filename to save variables to. The output is a text file. 
        Recommended file extension is '.dat'.

    Returns
    -------
    None.
    
    Notes
    -----
    
    The output includes the log modal amplitude, the natural frequency, and
    the damping factor for the current solution.

    See Also
    --------
    tmdsimpy.Continuation : 
        Class for continuation where this function is intended to be used 
        as a callback function.
    tmdsimpy.VibrationSystem.epmc_res : 
        EPMC residual function. This assumes that `XlamP` matches the unknowns
        vector for this residual function.
    combine_callback_funs : 
        Function for combining multiple callback functions for `Continuation`.
    continuation_save : 
        Function for saving the full solution points.

    Examples
    --------
    Define a callback function handle to pass to `tmdsimpy.Continuation` as
    
    >>> callback_fun = lambda XlamP, dirP_prev : print_epmc_stats(XlamP, 
    ...                                            dirP_prev, 'epmc_sum.dat')

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
    Saves HBM amplitude control key statistics to a text file for easy 
    monitoring.
    
    This function is intended for monitoring active runs. For final solution
    details, save the full solution and plot based on those results, not based
    on the results written out by this summary file.
    Function only intended to work when amplitude control is applied with HBM
    (Harmonic Balance Method).

    Parameters
    ----------
    XlamP : (N+1,) numpy.ndarray
        Solution at the current step in physical coordinates (including 
        continuation parameter) to a HBM amplitude control set of equations.
        First entries are harmonic displacements, 
        `XlamP[-2]` is force scaling, 
        `XlamP[-1]` is frequency (rad/s).
    dirP_prev : (N+1,) numpy.ndarray
        Direction for prediction from previous step to this step. 
        Can be used to get slope of solution at previous point w.r.t. `lam` 
        by dividing `dirP / dirP[-1]`
    fname : str
        Filename to save variables to. The output is a text file. 
        Recommended file extension is '.dat'.
    h : numpy.ndarray, sorted
        List of harmonics included in HBM solution.
    order : int, zero or positive
        Order of the derivative that is output. `order=0` means 
        displacement output, `order=2` means acceleration output.
    output_recov: (Ndof,) numpy.ndarray
        Recovery vector to be used to output response at a specific DOF
        where `Ndof` is the number of degrees of freedom of the system. 
    output_harmonic : int
        Which harmonic the amplitude at the output_recov dof should be output 
        for. Behavior is undefined if `output_harmonic` is not included in `h`.

    Returns
    -------
    None.
    
    Notes
    -----
    
    `Ndof` is the number of DOFs of the system while `N` is the number of 
    unknowns for the HBM residual at a given `lam` value.
    
    The output includes (for the current solution) frequency, 
    force scaling magnitude, and the amplitude at the DOF defined by 
    `output_recov` corresponding to `output_harmonic` and `order`.
    Other functions in this module correspond to other variants of HBM.

    See Also
    --------
    tmdsimpy.Continuation : 
        Class for continuation where this function is intended to be used 
        as a callback function.
    tmdsimpy.VibrationSystem.hbm_amp_control_res : 
        HBM amplitude control residual function. 
        This assumes that `XlamP` matches the unknowns vector for this residual
        function.
    combine_callback_funs : 
        Function for combining multiple callback functions for `Continuation`.
    continuation_save : 
        Function for saving the full solution points.

    Examples
    --------
    Define a callback function handle to pass to `tmdsimpy.Continuation` as
    
    >>> import numpy as np 
    ... 
    ... h = np.arange(5)
    ... order = 2 # Acceleration
    ... output_recov = np.array([1, 0, 0])
    ... output_harmonic = 1
    ... 
    ... callback_fun = lambda XlamP, dirP_prev : print_hbm_amp_stats(XlamP, 
    ...                    dirP_prev, 'hbm_sum.dat', h, order,
    ...                    output_recov, output_harmonic)

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

        amp = _calc_harmonic_resp(XlamP, XlamP[-1], h, order, 
                                  output_recov, output_harmonic)
        
        file.write(body_format.format(freq, force, amp))
    
    return

def print_hbm_amp_phase_stats(XlamP, dirP_prev, fname, freq, amp, h, order, 
                              output_recov, output_harmonic):
    """
    Saves HBM amplitude and phase control key statistics to a text file for 
    easy monitoring.
    
    This function is intended for monitoring active runs. For final solution
    details, save the full solution and plot based on those results, not based
    on the results written out by this summary file.
    Function only intended to work when amplitude and phase control is applied
    with HBM (Harmonic Balance Method).

    Parameters
    ----------
    XlamP : (N+1,) numpy.ndarray
        Solution at the current step in physical coordinates (including 
        continuation parameter) to a HBM amplitude control set of equations.
        
        Assumes that this is harmonic displacements, then cosine scaling of 
        force, sine scaling of force, ignored lam variable (but requires 
        exactly 1 variable after the two force scaling values for indexing)
    dirP_prev : (N+1,) numpy.ndarray
        Direction for prediction from previous step to this step. 
        Can be used to get slope of solution at previous point w.r.t. `lam` 
        by dividing `dirP / dirP[-1]`
    fname : str
        Filename to save variables to. The output is a text file. 
        Recommended file extension is '.dat'.
    freq : float
        Frequency to be printed, rad/s.
    amp : amplitude
        Control amplitude that should be printed.
    h : numpy.ndarray, sorted
        List of harmonics included in HBM solution.
    order : int, zero or positive
        Order of the derivative that is output. `order=0` means 
        displacement output, `order=2` means acceleration output.
    output_recov: (Ndof,) numpy.ndarray
        Recovery vector to be used to output response at a specific DOF
        where `Ndof` is the number of degrees of freedom of the system. 
    output_harmonic : int
        Which harmonic the amplitude at the output_recov dof should be output 
        for. Behavior is undefined if `output_harmonic` is not included in `h`.

    Returns
    -------
    None.
    
    Notes
    -----
    
    
    `Ndof` is the number of DOFs of the system while `N` is the number of 
    unknowns for the HBM residual at a given `lam` value.
    
    Frequency and amplitude of 1st harmonic are only printed from those 
    arguments and not based on XlamP. Define your anonymous function 
    appropriately.
    
    The output includes (for the current solution) frequency, 
    force scaling magnitude for sine and cosine, 
    and the amplitude at the DOF defined by 
    `output_recov` corresponding to `output_harmonic` and `order`.
    Other functions in this module correspond to other variants of HBM.
    
    See Also
    --------
    tmdsimpy.Continuation : 
        Class for continuation where this function is intended to be used 
        as a callback function.
    tmdsimpy.VibrationSystem.hbm_amp_phase_control_res : 
        HBM amplitude and phase control residual function. 
        `XlamP` works when it matches this unknown vector.
    tmdsimpy.VibrationSystem.hbm_amp_phase_control_dA_res :
        HBM with amplitude and phase control residual function.
        `XlamP` works when it matches this unknown vector.
    combine_callback_funs : 
        Function for combining multiple callback functions for `Continuation`.
    continuation_save : 
        Function for saving the full solution points.

    Examples
    --------
    Define a callback function handle to pass to `tmdsimpy.Continuation` 
    for continuation with respect to frequency at a constant amplitude 
    with `tmdsimpy.VibrationSystem.hbm_amp_phase_control_res`
    
    >>> import numpy as np 
    ... 
    ... h = np.arange(5)
    ... order = 2 # Acceleration
    ... output_recov = np.array([1, 0, 0])
    ... output_harmonic = 1
    ...
    ... control_amp = 40 # m/s^2 since order is 2.
    ... 
    ... callback_fun = lambda XlamP, dirP_prev : print_hbm_amp_stats(XlamP,
    ...                    dirP_prev, 'hbm_sum.dat', XlamP[-1], control_amp, h,
    ...                    order, output_recov, output_harmonic)


    Define a callback function handle to pass to `tmdsimpy.Continuation` 
    for continuation with respect to frequency at a constant amplitude 
    with `tmdsimpy.VibrationSystem.hbm_amp_phase_control_dA_res`
    
    >>> import numpy as np 
    ... 
    ... h = np.arange(5)
    ... order = 2 # Acceleration
    ... output_recov = np.array([1, 0, 0])
    ... output_harmonic = 1
    ...
    ... constant_freq = 1 # rad/s.
    ... 
    ... callback_fun = lambda XlamP, dirP_prev : print_hbm_amp_stats(XlamP,
    ...                    dirP_prev, 'hbm_sum.dat', constant_freq, XlamP[-1],
    ...                    h, order, output_recov, output_harmonic)
    
    """
    
    # If file doesn't exist, write a header
    write_header = not exists(fname)
    
    # Write current stats to file
    with open(fname, 'a') as file:
        
        if write_header:
            header_format = '{:^18s} & {:^18s} & {:^18s} & {:^18s} & {:^18s} \n'
            file.write(header_format.format('Frequency [Hz]',
                                            'Amplitude [m/s^{}]'.format(order),
                                            'Force Cosine [N]',
                                            'Force Sine [N]',
                                            'Recov Amp [m/s^{}]'.format(order)))
            
        body_format = '{: ^18.3f} & {: ^18.3f} & {: ^18.3e} '\
                        + '& {: ^18.3e} & {: ^18.3e} \n'
        
        freq_hz = freq / 2 / np.pi # Frequency in Hz

        amp_recov = _calc_harmonic_resp(XlamP, freq, h, order, 
                                  output_recov, output_harmonic)
        
        file.write(body_format.format(freq_hz, amp, XlamP[-3], XlamP[-2],
                                      amp_recov))


def print_vprnm_stats(XlamP, dirP_prev, fname, h, order,
                        output_recov_harmonic_list):
    """
    Saves VPRNM key statistics to a text file for easy monitoring.

    This function is intended for monitoring active runs. For final solution
    details, save the full solution and plot based on those results, not based
    on the results written out by this summary file.
    Function only intended to work for basic VPRNM, see module for other
    options.
    
    Parameters
    ----------
    XlamP : (N+1,) numpy.ndarray
        Solution at the current step in physical coordinates (including 
        continuation parameter) to a VPRNM set of equations.
        First entries are harmonic displacements, `XlamP[-2]` is frequency
        (rad/s), `XlamP[-1]` is force scaling.
    dirP_prev : (N+1,) numpy.ndarray
        Direction for prediction from previous step to this step. 
        Can be used to get slope of solution at previous point w.r.t. `lam` 
        by dividing `dirP / dirP[-1]`
    fname : str
        Filename to save variables to. The output is a text file. 
        Recommended file extension is '.dat'.
    h : numpy.ndarray, sorted
        List of harmonics included in VPRNM solution.
    order : int, zero or positive
        Order of the derivative that is output. `order=0` means 
        displacement output, `order=2` means acceleration output.
    output_recov_harmonic_list : list of tuples
        Each tuple contains 1. a `(Ndof,) numpy.ndarray` describing the output
        DOF and 2. an int describing which harmonic should be output.
        Behavior is undefined if output harmonic is not included in `h`.

    Returns
    -------
    None.
    
    Notes
    -----
    
    `Ndof` is the number of DOFs of the system while `N` is the number of 
    unknowns for the VPRNM residual at a given `lam` value.

    The output includes (for the current solution) force magnitude, frequency, 
    and the amplitude at the DOFs and harmonics defined by 
    `output_recov_harmonic_list` at displacement derivative `order`.
    Other functions in this module correspond to other variants of VPRNM/HBM.

    See Also
    --------
    tmdsimpy.Continuation : 
        Class for continuation where this function is intended to be used 
        as a callback function.
    tmdsimpy.VibrationSystem.vprnm_res : 
        VPRNM residual function. 
        This assumes that `XlamP` matches the unknowns vector for this residual
        function.
    combine_callback_funs : 
        Function for combining multiple callback functions for `Continuation`.
    continuation_save : 
        Function for saving the full solution points.

    Examples
    --------
    Define a callback function handle to pass to `tmdsimpy.Continuation` as
    
    >>> import numpy as np 
    ... 
    ... h = np.arange(5)
    ... order = 2 # Acceleration
    ... 
    ... output_recov1 = np.array([1, 0, 0])
    ... output_harmonic1 = 1
    ... 
    ... output_recov2 = np.array([0, 1, 0])
    ... output_harmonic2 = 3
    ...
    ... output_recov_harmonic_list = [(output_recov1, output_harmonic1),
    ...                               (output_recov2, output_harmonic2)]
    ... 
    ... callback_fun = lambda XlamP, dirP_prev : print_vprnm_stats(XlamP, 
    ...                    dirP_prev, 'vprnm_sum.dat', h, order,
    ...                    output_recov_harmonic_list)

    """
    
    # If file doesn't exist, write a header
    write_header = not exists(fname)
    
    # Write current stats to file
    with open(fname, 'a') as file:
        
        if write_header:
            output_dofs = '& {:^22s} '.format('Recov Amp [m/s^{}]'.format(order))
            
            header_format = '{:^18s} & {:^18s} ' + \
                        len(output_recov_harmonic_list)*output_dofs \
                        + '\n'
            
            file.write(header_format.format('Force Scaling [N]',
                                            'Frequency [Hz]'))
            
            harmonic_line = ('{:^18s} & {:^18s} '.format('','')) \
                + ''.join(['& {:^22s}'.format('dof={}, h={}'.format(ind, rh[1])) \
                           for ind,rh in enumerate(output_recov_harmonic_list)]) \
                + '\n'
            
            file.write(harmonic_line)
            
        body_format = '{: ^18.3f} & {: ^18.3f} {} \n'
        
        force =  XlamP[-1] # Scaling factor (generally N)
        freq = XlamP[-2] / 2 / np.pi # Frequency in Hz
        
        amp_subcomponent = '& {: ^22.3e}'
        
        amp_list = [None] * len(output_recov_harmonic_list)
        
        for ind, rh in enumerate(output_recov_harmonic_list):
            
            output_recov = rh[0]
            output_harmonic = rh[1]
            
            amp_list[ind] = _calc_harmonic_resp(XlamP, XlamP[-2], h, order, 
                                                output_recov, output_harmonic)
            
        amp_string = ''.join([amp_subcomponent.format(ampi) for ampi in amp_list])
        
        file.write(body_format.format(force, freq, amp_string))
   
def print_vprnm_amp_phase_stats(XlamP, dirP_prev, fname, h, control_order,
                                output_order, output_recov_harmonic_list):
    """
    Saves VPRNM Amplitude and Phase Control key statistics to a text file for 
    easy monitoring.

    This function is intended for monitoring active runs. For final solution
    details, save the full solution and plot based on those results, not based
    on the results written out by this summary file.
    Function only intended to work for basic VPRNM, see module for other
    options.
    
    Parameters
    ----------
    XlamP : (N+1,) numpy.ndarray
        Solution at the current step in physical coordinates (including 
        continuation parameter) to a VPRNM amplitude control set of equations.
        First entries are harmonic displacements,
        `XlamP[-4]` is cosine force scaling, `XlamP[-3]` is sine force scaling, 
        `XlamP[-2]` is frequency (rad/s), `XlamP[-1]` is amplitude level.
    dirP_prev : (N+1,) numpy.ndarray
        Direction for prediction from previous step to this step. 
        Can be used to get slope of solution at previous point w.r.t. `lam` 
        by dividing `dirP / dirP[-1]`
    fname : str
        Filename to save variables to. The output is a text file. 
        Recommended file extension is '.dat'.
    h : numpy.ndarray, sorted
        List of harmonics included in VPRNM solution.
    control_order : int, zero or positive
        Order of the derivative that is controlled. order=0 means 
        displacement output, order=2 means acceleration output
        Solely determines the header for the column, but does not
        effect any of the values since VPRNM directly controls the derivative
        order of the amplitude.
    output_order : int, zero or positive
        Order of the derivative that is output. `order=0` means 
        displacement output, `order=2` means acceleration output.
    output_recov_harmonic_list : list of tuples
        Each tuple contains 1. a `(Ndof,) numpy.ndarray` describing the output
        DOF and 2. an int describing which harmonic should be output.
        Behavior is undefined if output harmonic is not included in `h`.

    Returns
    -------
    None.
    
    Notes
    -----
    
    `Ndof` is the number of DOFs of the system while `N` is the number of 
    unknowns for the VPRNM residual at a given `lam` value.

    The output includes (for the current solution) force magnitude, frequency, 
    and the amplitude at the DOFs and harmonics defined by 
    `output_recov_harmonic_list` at displacement derivative `order`.
    Other functions in this module correspond to other variants of VPRNM/HBM.

    See Also
    --------
    tmdsimpy.Continuation : 
        Class for continuation where this function is intended to be used 
        as a callback function.
    tmdsimpy.VibrationSystem.vprnm_amp_phase_res : 
        VPRNM residual function. 
        This assumes that `XlamP` matches the unknowns vector for this residual
        function.
    combine_callback_funs : 
        Function for combining multiple callback functions for `Continuation`.
    continuation_save : 
        Function for saving the full solution points.

    Examples
    --------
    Define a callback function handle to pass to `tmdsimpy.Continuation` as
    
    >>> import numpy as np 
    ... 
    ... h = np.arange(5)
    ... control_order = 2 # Acceleration
    ... output_order = 2 # Acceleration
    ... 
    ... output_recov1 = np.array([1, 0, 0])
    ... output_harmonic1 = 1
    ... 
    ... output_recov2 = np.array([0, 1, 0])
    ... output_harmonic2 = 3
    ...
    ... output_recov_harmonic_list = [(output_recov1, output_harmonic1),
    ...                               (output_recov2, output_harmonic2)]
    ... 
    ... callback_fun = lambda XlamP, dirP_prev : print_vprnm_amp_phase_stats(
    ...                    XlamP, dirP_prev, 'vprnm_sum.dat', h, control_order,
    ...                    output_order, output_recov_harmonic_list)

    """
    
    
    # If file doesn't exist, write a header
    write_header = not exists(fname)
    
    # Write current stats to file
    with open(fname, 'a') as file:
        
        if write_header:
            output_dofs = '& {:^22s} '.format('Recov Amp [m/s^{}]'.format(output_order))
            
            header_format = '{:^18s} & {:^18s} & {:^18s} & {:^18s} ' + \
                        len(output_recov_harmonic_list)*output_dofs \
                        + '\n'
            
            file.write(header_format.format('Amp Control [m/s^{}]'.format(control_order),
                                            'Force Cosine [N]',
                                            'Force Sine [N]',
                                            'Frequency [Hz]'))
            
            harmonic_line = ('{:^18s} & {:^18s} '.format('','')) \
                + ''.join(['& {:^22s}'.format('dof={}, h={}'.format(ind, rh[1])) \
                           for ind,rh in enumerate(output_recov_harmonic_list)]) \
                + '\n'
            
            file.write(harmonic_line)
            
        body_format = '{: ^18.3f} & {: ^18.3f} & {: ^18.3f} & {: ^18.3f} {} \n'
        
        amp_control = XlamP[-1] # Amplitude control
        force_cos =  XlamP[-4] # Scaling factor (generally N)
        force_sin =  XlamP[-3] # Scaling factor (generally N)
        freq = XlamP[-2] / 2 / np.pi # Frequency in Hz
        
        amp_subcomponent = '& {: ^22.3e}'
        
        amp_list = [None] * len(output_recov_harmonic_list)
        
        for ind, rh in enumerate(output_recov_harmonic_list):
            
            output_recov = rh[0]
            output_harmonic = rh[1]
            
            amp_list[ind] = _calc_harmonic_resp(XlamP, XlamP[-2], h, 
                                                output_order, 
                                                output_recov, 
                                                output_harmonic)
            
        amp_string = ''.join([amp_subcomponent.format(ampi) for ampi in amp_list])
        
        file.write(body_format.format(amp_control, force_cos, force_sin, 
                                      freq, amp_string))
        
def _calc_harmonic_resp(U, w, h, order, output_recov, output_harmonic):
    """
    Function for calculating the harmonic response at a given harmonic, 
    extraction vector, and power of the frequency

    Parameters
    ----------
    U : (N+a,) numpy.ndarray
        harmonic displacements. This is only indexed from the beginning so 
        other items may be included at the end.
    w : float
        frequency in rad/s.
    h : numpy.ndarray, sorted
        List of harmonics included in harmonic solution
    order : int, zero or positive
        Order of the derivative that is controlled. order=0 means 
        displacement output, order=2 means acceleration output.
        This is the derivative of displacement that is output.
    output_recov : (N,) numpy.ndarray
        Vector for extracting the DOF of interest from the displacements.
    output_harmonic : int
        harmonic of interest from the list h.

    Returns
    -------
    amp : float
        Response amplitude for DOF, harmonic, order that are input.
        
    Notes
    -----
    This is just intended for use in callback summary functions and is does
    not have tests written for it.

    """
    
    Ndof = output_recov.shape[0]
    Nhc_before = hutils.Nhc(h[h < output_harmonic])
    
    amp_cos = output_recov @ U[Nhc_before*Ndof:(Nhc_before+1)*Ndof]
    amp_sin = output_recov @ U[(Nhc_before+1)*Ndof:(Nhc_before+2)*Ndof]
    
    amp = np.sqrt(amp_cos**2 + amp_sin**2) *((output_harmonic*w)**order)
                        
    return amp
                                
