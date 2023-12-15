"""
Example Script for a Simulation of the Brake-Reuss Beam

Friction Model: Elastic Dry Friction or Rough Contact (TBD)

Nonlinear Modal Analysis: Extended Periodic Motion Concept

Model: 232 Zero Thickness Elements (ZTEs) [Hyper Reduction Paper]
        Model file: matrices/ROM_U_232ELS4py.mat
        Model file must be downloaded from storage elsewhere. See README.md

Reference Papers:
    

TODO : 
    1. Readme for file downloads for matrices etc.
    2. References in intro / with more detail
    3. Terminology/nomenclature in this comment?

"""

import sys
import numpy as np
from scipy import io as sio
import warnings

sys.path.append('../..')
from tmdsimpy.solvers import NonlinearSolver


###############################################################################
####### Load System Matrices from .mat File                             #######
###############################################################################

system_matrices = sio.loadmat('./matrices/ROM_U_232ELS4py.mat')

######## Sanity Checks on Loaded Matrices

# Sizes
assert system_matrices['M'].shape == system_matrices['K'].shape, \
        'Mass and stiffness matrices are not the same size, this will not work.'

if not (system_matrices['M'].shape == (859, 859)):
    warnings.warn("Warning: Mass and stiffness matrices are not the expected "\
                  "size for the UROM 232 Model.")

# Approximate Frequencies Without Contact
# If running a different ROM, these will vary slightly
solver = NonlinearSolver()

eigvals, eigvecs = solver.eigs(system_matrices['K'], system_matrices['M'], 
                               subset_by_index=[0, 9])

expected_eigvals = np.array([1.855211e+01, 1.701181e+05, 8.196151e+05, 
                              1.368695e+07, 1.543605e+07, 1.871511e+07, 
                              1.975941e+07, 2.692282e+07, 3.442458e+07, 
                              8.631324e+07])

first_eig_ratio = eigvals[0] / eigvals[1]
eig_diff = np.abs((eigvals - expected_eigvals)/expected_eigvals)[1:].max()

if first_eig_ratio > 1e-3:
    warnings.warn("Expected rigid body mode of first eigenvalue "
          "is high: ratio eigvals[0]/eigvals[1]={:.3e}".format(first_eig_ratio))

if eig_diff > 1e-4:
    warnings.warn("Eigenvalues differed by fraction: {:.3e}".format(eig_diff))

# Check that the integrated area is as expected
ref_area = 0.002921034742767
area_error = (system_matrices['Tm'].sum() - ref_area) / ref_area

assert area_error < 1e-4, 'Quadrature integration matrix gives wrong contact area.'


###############################################################################
####### Create Vibration System                                         #######
###############################################################################



###############################################################################
####### Add Nonlinear Forces to System                                  #######
###############################################################################




###############################################################################
####### Prestress Analysis                                              #######
###############################################################################



###############################################################################
####### Updated Eigenvalue Analysis After Prestress                     #######
###############################################################################



###############################################################################
####### EPMC Initial Guess                                              #######
###############################################################################




###############################################################################
####### EPMC Continuation                                               #######
###############################################################################

