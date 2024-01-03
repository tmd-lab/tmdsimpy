import numpy as np
from .vibration_system import VibrationSystem
from . import harmonic_utils as hutils

# Separate out parallel functions into a different module file to avoid errors
from . import _shared_mem_utils as smutils


# Parallel tools packages
import multiprocessing as mp

# if mp.get_start_method() == 'fork':
if mp.get_start_method(allow_none=True) is None:
    from multiprocessing import set_start_method
    set_start_method("spawn")
    
from functools import partial #, reduce


import warnings
warnings.warn('Need to write a test for RoughContact returning local AFT results.')


class VibrationSystemSM(VibrationSystem):
    """
    Modifications to VibrationSystem to improve shared memory execution.
    """
    
    
    
    def total_aft(self, U, w, h, Nt=128, aft_tol=1e-7):
        """
        Apply Alternating Time Frequency Method to calculate nonlinear force
        coefficients for all nonlinear forces in system with modifications for 
        shared memory
        
        Nhc is the number of harmonic components in that h represents
        can be calculated by harmonic_utils.Nhc(h)
        
        Parameters
        ----------
        U : np.array (n * Nhc,) 
            Harmonic DOFs, displacements, np.hstack((U0, U1c, U1s...)) with 
            harmonics h
        w : double
            Frequency
        h : 1D np.array
            Sorted list of harmonics
        Nt : integer, power of 2
            Number of Time Steps for AFT. The default is 128.
        aft_tol : double
            Tolerance for AFT. The default is 1e-7.

        Returns
        -------
        Fnl : np.array (n*Nhc,)
            Nonlinear Force Harmonic Coefficients
        dFnldU : np.array (n*Nhc, n*Nhc)
            Jacobian of Fnl w.r.t. Harmonic DOFs
        dFnldw : np.array (n*Nhc,)
            Derivative of Fnl w.r.t. frequency
        
        """
        
        # Counting:
        Nhc = hutils.Nhc(h) # Number of Harmonic Components
        Ndof = self.M.shape[0]
        
        # Initialize Memory
        Fnl = np.zeros((Nhc*Ndof,), np.double)
        dFnldU = np.zeros((Nhc*Ndof,Nhc*Ndof), np.double)
        dFnldw = np.zeros((Nhc*Ndof,), np.double)
        
        # Initial testing indicates pool creation time is not a major concern, 
        # so use safe approach to avoid duplicate pools / failing to close pools
        # with mp.get_context("spawn").Pool(processes = mp.cpu_count()) as pool:
        with mp.Pool(processes = mp.cpu_count()) as pool:
            
            # Rough estimate is that want num_items = 10x processors to allow
            # for automatic load balance. Therefore, unlikely to be worth 
            # dividing the list into individual parts.
            # However, not dividing may require more memory, so no guarantees.
            res_per_process = pool.map(
                                    partial(smutils._single_aft, U, w, h, Nt, aft_tol), 
                                    self.nonlinear_forces
                                       )

            # # Try dividing between processors so that reduce reloading and 
            # # potential recompile?
            # res_per_process = pool.map(
            #                         partial(smutils._single_aft, U, w, h, Nt, aft_tol), 
            #                         smutils.divide_list(self.nonlinear_forces, pool._processes)
            #                            )

        # Convert parallel global forces into local forces with a serial for loop
        for ind, nlforce in enumerate(self.nonlinear_forces):
            
            Flocal = res_per_process[ind][0]
            dFdUlocal = res_per_process[ind][1]
            dFdwlocal = res_per_process[ind][2]
            
            Ndnl = Flocal.shape[0]
            
            Fnl += np.reshape(nlforce.T @ Flocal, (U.shape[0],), 'F')
            dFnldU += np.kron(np.eye(Nhc), nlforce.T) @ dFdUlocal \
                                                    @ np.kron(np.eye(Nhc), nlforce.Q)
            
            dFnldw += np.reshape(nlforce.T @ \
                                np.reshape(dFdwlocal, (Ndnl, Nhc)), \
                                (U.shape[0],), 'F')
            
        
        # # Old implement
        # for nlforce in self.nonlinear_forces:
        #     Fnl_curr, dFnldU_curr, dFnldw_curr = nlforce.aft(U, w, h, Nt, aft_tol)
        #    
        #     Fnl += Fnl_curr
        #     dFnldU += dFnldU_curr
        #     dFnldw += dFnldw_curr
        
        return Fnl, dFnldU, dFnldw
    
