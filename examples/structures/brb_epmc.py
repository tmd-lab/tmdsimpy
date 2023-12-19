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
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.jax.nlforces.roughcontact.rough_contact import RoughContactFriction


###############################################################################
####### User Inputs                                                     #######
###############################################################################

# File name for a .mat file that contains the matrices appropriately formatted
# to be able to describe the structural system. 
system_fname = './matrices/ROM_U_232ELS4py.mat'


# Estimated displacements. Because the rough contact stiffness goes to zero
# when contact is at zero displacement, an initial estimate of the normal 
# displacement is needed to get an initial stiffness to generate an initial
# guess for the nonlinear solver. 
uxyn_est = np.array([0, 0, .1e-5])


###############################################################################
####### Load System Matrices from .mat File                             #######
###############################################################################

system_matrices = sio.loadmat(system_fname)

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
####### Friction Model Parameters                                       #######
###############################################################################

# TODO : update this to load friction model parameters from a file.

ElasticMod = 192.31e9 # Pa
PoissonRatio = 0.3
Radius = 1.365e-3 # m
TangentMod = 620e6 # Pa
YieldStress = 330e6 # Pa 
mu = 0.03

area_density = 1.371e6 # Asperities / m^2
max_gap = 28.05e-6 # m

gaps = np.array([0.0, max_gap])
gap_weights = np.ones(2)*0.5*area_density

prestress = (12002+12075+12670)*1.0/3; # N per bolt

###############################################################################
####### Create Vibration System                                         #######
###############################################################################

vib_sys = VibrationSystem(system_matrices['M'], system_matrices['K'], 
                          ab=[0.01, 0.0])

###############################################################################
####### Add Nonlinear Forces to System                                  #######
###############################################################################

# Number of nonlinear frictional elements, Number of Nodes
Nnl,Nnodes = system_matrices['Qm'].shape 

# Need to convert sparse loads into arrays so that operations are expected shapes
# Sparse matrices from matlab are loaded as matrices rather than numpy arrays
# and behave differently than numpy arrays.
Qm = np.array(system_matrices['Qm'].todense()) 
Tm = np.array(system_matrices['Tm'].todense())

# Pull out for reference convenience
L  = system_matrices['L']

QL = np.kron(Qm, np.eye(3)) @ L[1:3*Nnodes+1, :]
LTT = L[1:3*Nnodes+1, :].T @ np.kron(Tm, np.eye(3))

for i in range(Nnl):
    
    Ls = (QL[i*3:(i*3+3), :])
    Lf = (LTT[:, i*3:(i*3+3)])

    tmp_nl_force = RoughContactFriction(Ls, Lf, ElasticMod, PoissonRatio, 
                                        Radius, TangentMod, YieldStress, mu,
                                        gaps=gaps, gap_weights=gap_weights)
    
    vib_sys.add_nl_force(tmp_nl_force)
    
    
# Create a reference nonlinear element that can be used for initial guesses
ref_nlforce = RoughContactFriction(np.eye(3), np.eye(3), ElasticMod, 
                                   PoissonRatio, Radius, TangentMod, 
                                   YieldStress, mu,
                                   gaps=gaps, gap_weights=gap_weights)

###############################################################################
####### Prestress Analysis                                              #######
###############################################################################

vib_sys.set_prestress_mu()

Fv = system_matrices['Fv'][:, 0]

# Get an estimate of the stiffness at a contact
t, dtduxyn = ref_nlforce.force(uxyn_est)

# linearized stiffness matrix with normal contact friction 
# Tangent friction is set to zero for prestress so do the same here.
Kstuck = np.zeros((L.shape[0], L.shape[0]))

place_normal = np.eye(3)
place_normal[0,0] = 0
place_normal[1,1] = 0
kn_mat = Tm @ (dtduxyn[2,2] * Qm)
Kstuck[:3*Nnodes, :3*Nnodes] += np.kron(kn_mat, place_normal)

K0 = system_matrices['K'] + L.T @ Kstuck @ L

# Calculate an initial guess
X0 = np.linalg.solve(K0,(Fv * prestress))

# function to solve
pre_fun = lambda U : vib_sys.static_res(U, Fv*prestress)

R0, dR0dX = pre_fun(X0)

print('Residual norm of initial guess: {:.4e}'.format(np.linalg.norm(dR0dX)))

# solve function
solver = NonlinearSolver()

import time

t0 = time.time()
Xpre, R, dRdX, sol = solver.nsolve(pre_fun, X0)

t1 = time.time()

print('Residual norm: {:.4e}'.format(np.linalg.norm(R)))

print('Static Solution Run Time : {:.3e} s'.format(t1 - t0))

vib_sys.reset_real_mu()

###############################################################################
####### Updated Eigenvalue Analysis After Prestress                     #######
###############################################################################

# Recalculate stiffness with real mu
Rpre, dRpredX = vib_sys.static_res(Xpre, Fv*prestress)


eigvals, eigvecs = solver.eigs(dRpredX, system_matrices['M'], 
                                subset_by_index=[0, 9], symmetric=False)

print('Prestress State Frequencies: [Hz]')
print(np.sqrt(eigvals)/(2*np.pi))


###############################################################################
####### EPMC Initial Guess                                              #######
###############################################################################




###############################################################################
####### EPMC Continuation                                               #######
###############################################################################

