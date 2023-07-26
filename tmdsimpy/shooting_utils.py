import numpy as np

from .vibration_system import VibrationSystem

# from scipy.linalg import eigh # ONLY HERMITIAN - DOESN'T WORK HERE.

def postproc_shooting(vib_sys, XlamP_shoot, Fl, Nt=128):
    """
    Calculate post processing on shooting to determine time series data and 
    stability

    Parameters
    ----------
    vib_sys : TYPE
        DESCRIPTION.
    XlamP_shoot : TYPE
        DESCRIPTION.
    Fl : TYPE
        DESCRIPTION.
    Nt : TYPE, optional
        DESCRIPTION. The default is 128.

    Returns
    -------
    None.

    """
    
    max_eig = np.zeros(XlamP_shoot.shape[0])
    
    y_t = np.zeros((Nt+1, XlamP_shoot.shape[0]))
    ydot_t = np.zeros((Nt+1, XlamP_shoot.shape[0]))
    
    for i in range(XlamP_shoot.shape[0]):
        
        Uw = XlamP_shoot[i, :]
        
        R,dRdX,dRdw,aux_res = vib_sys.shooting_res(Uw, Fl, Nt=Nt, return_aux=True)
        
        y_t[:, i] = aux_res[1]
        ydot_t[:, i] = aux_res[2]
        
        monodromy = aux_res[0]
        
        eigvals = np.linalg.eigvals(monodromy)
        
        max_eig[i] = np.max(np.abs(eigvals))
        

    stable = max_eig <= 1.0
    
    return y_t, ydot_t, stable, max_eig