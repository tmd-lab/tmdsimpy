"""
Example Script for a Simulation of the Brake-Reuss Beam utilizing an advanced
rough contact model. This recreates one of the simulations from [1]. For full
simulation descriptions and many details about the model and methods see [1].

Friction Model: Rough Contact [1] (TODO: Add flag for Elastic Dry Friction)

Nonlinear Modal Analysis: Extended Periodic Motion Concept

Model: 232 Zero Thickness Elements (ZTEs) [Hyper Reduction Paper]
        Model file: matrices/ROM_U_232ELS4py.mat
        Model file must be downloaded from storage elsewhere. See README.md
        
Surface Parameters: Surface parameters for rough contact are identified in [1]
        Surface Parameters file: matrices/combined_14sep21_R1_4py.mat
        Surface parameter file must be downloaded from storage elsewhere. See
        README.md

Reference Papers:
 [1] Porter, Justin H., and Matthew R. W. Brake. "Towards a Predictive, 
     Physics-Based Friction Model for the Dynamics of Jointed Structures." 
     Mechanical Systems and Signal Processing 192 (June 1, 2023): 110210.
     https://doi.org/10.1016/j.ymssp.2023.110210.


TODO : 
    1. Readme for file downloads for matrices etc.
    2. Terminology/nomenclature in this comment?
    3. Add elastic dry friction flag option?

The JAX matrix solves in here should automatically use OpenMP Parallelism.
You can control the parallelism by invoking these commands on Linux prior 
to execution
> export OMP_PROC_BIND=spread # Spread threads out over physical cores
> export OMP_NUM_THREADS=32 # Change 32 to desired number of threads

non-blocking jax parallelism for friction evaluations appears to spreads across
available threads ignoring the OMP_NUM_THREADS environment variable.

"""

import sys
import numpy as np
from scipy import io as sio
import warnings

sys.path.append('../..')
from tmdsimpy import harmonic_utils as hutils

from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.jax.solvers import NonlinearSolverOMP

from tmdsimpy.continuation import Continuation
import tmdsimpy.continuation_utils as cont_utils

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

# Surface Parameters for the rough contact model
surface_fname = './matrices/combined_14sep21_R1_4py.mat'

# Log Amplitude Start
Astart = -7.7

# Log Amplitude End
Aend = -4.2

# Continuation Step Size (starting)
ds = 0.1
dsmax = 0.2
dsmin = 0.02


fast_sol = False # Choose speed or accuracy

if fast_sol:
    # Run with reduced harmonics and AFT time steps to keep time within a 
    # few minutes
    h_max = 1
    Nt = 1<<3
    FracLam = 0.5 # Continuation weighting
else:
    # Normal - settings for higher accuracy as used in previous papers
    h_max = 3
    Nt = 1<<7
    
    ds = 0.08
    dsmax = 0.125*1.4
    # Adjust weighting of amplitude v. other in continuation to hopefully 
    # reduce turning around. Higher puts more emphasis on continuation 
    # parameter (amplitude)
    FracLam = 0.50     

run_profilers = False # Run profiling of code operations to identify bottlenecks
static_profile = './results/static_profile' # File to save static analysis profiling
continue_profile = './results/continue_profile' # File to save continuation profiling in



###############################################################################
####### Solver Settings                                                 #######
###############################################################################

########################################
# Static Solver settings

# The static solution does not have a good initial guess and may be 
# ill-conditioned (conditioning for residual function is not implemented)
# therefore, it is recommended to use reform_freq=1 and thus a full Newton 
# scheme. 

static_config={'max_steps' : 30,
                'reform_freq' : 1,
                'verbose' : True, 
                'xtol'    : None, 
                'stopping_tol' : ['xtol']
                }

# solve function - can use python library routines or custom ones
# static_solver = NonlinearSolver() # scipy nonlinear solver
static_solver = NonlinearSolverOMP(config=static_config) # Custom Newton-Raphson solver


########################################
# EPMC Solver settings

epmc_config={'max_steps' : 12, # balance with reform_freq
            'reform_freq' : 2,
            'verbose' : True, 
            'xtol'    : None, # Just use the one passed from continuation
            'rtol'    : 1e-9,
            'etol'    : None,
            'xtol_rel' : 1e-6, 
            'rtol_rel' : None,
            'etol_rel' : None,
            'stopping_tol' : ['xtol'],
            'accepting_tol' : ['xtol_rel', 'rtol']
            }

# solve function - can use python library routines or custom ones
# epmc_solver = NonlinearSolver() # scipy nonlinear solver
epmc_solver = NonlinearSolverOMP(config=epmc_config) # Custom Newton-Raphson solver



###############################################################################
####### EPMC Output Save Information                                    #######
###############################################################################

epmc_full_name = './results/brb_epmc_bb_full.npz' # Detailed full output (numpy binary)
epmc_dat = './results/brb_epmc_bb_sum.dat' # Summary file output (text file)

call_list = [lambda XlamP, dirP_prev : cont_utils.continuation_save(XlamP, dirP_prev, epmc_full_name),
             lambda XlamP, dirP_prev : cont_utils.print_epmc_stats(XlamP, dirP_prev, epmc_dat)]

callback_funs = lambda XlamP, dirP_prev : cont_utils.combine_callback_funs(\
                                                   call_list, XlamP, dirP_prev)

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

eigvals, eigvecs = static_solver.eigs(system_matrices['K'], system_matrices['M'], 
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

surface_pars = sio.loadmat(surface_fname)

ElasticMod = 192.31e9 # Pa
PoissonRatio = 0.3
Radius = surface_pars['Re'][0, 0] # m
TangentMod = 620e6 # Pa
YieldStress = 331.7e6 # Pa 
mu = 0.03

area_density = surface_pars['area_density'][0,0] # Asperities / m^2
max_gap = surface_pars['z_max'][0,0] # m

normzinterp = surface_pars['normzinterp'][0]
pzinterp    = surface_pars['pzinterp'][0]

gaps = np.linspace(0, 1.0, 101) * max_gap

trap_weights = np.ones_like(gaps)
trap_weights[1:-1] = 2.0
trap_weights = trap_weights / trap_weights.sum()

gap_weights = area_density * trap_weights * np.interp(gaps/max_gap, 
                                                      normzinterp, pzinterp)

prestress = (12002+12075+12670)*1.0/3; # N per bolt

###############################################################################
####### Create Vibration System                                         #######
###############################################################################
# Initial Guess Constructed based only on mass prop damping.
# TODO : Add better damping calculation after prestress
import warnings
warnings.warn('Linear damping is only initialized as mass proportional at a fixed frequency here.')
damp_ab = [0.087e-2*2*(168.622*2*np.pi), 0.0]

vib_sys = VibrationSystem(system_matrices['M'], system_matrices['K'], 
                          ab=damp_ab)

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

QL = np.kron(Qm, np.eye(3)) @ L[:3*Nnodes, :]
LTT = L[:3*Nnodes, :].T @ np.kron(Tm, np.eye(3))

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
pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv*prestress)

R0, dR0dX = pre_fun(X0)

print('Residual norm of initial guess: {:.4e}'.format(np.linalg.norm(dR0dX)))

if run_profilers:
    
    import cProfile
    
    cProfile.run('Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0, verbose=True, xtol=1e-13)', 
                 static_profile)
    
    print('Static run time saved to {}. This can be loaded and investigated.'.format(static_profile))
    print('See https://docs.python.org/3/library/profile.html for more details.')
    
    """
    # Load and investigate profile: 
    import pstats
    from pstats import SortKey
    p = pstats.Stats('static_profile')
    p.sort_stats(SortKey.TIME).print_stats(10)
    """
    
else:
    import time
    
    t0 = time.time()
    Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0, verbose=True, xtol=1e-13)
    
    t1 = time.time()
    
    print('Static Solution Run Time : {:.3e} s'.format(t1 - t0))

print('Residual norm: {:.4e}'.format(np.linalg.norm(R)))

# Update history variables after static so sliders reset
vib_sys.update_force_history(Xpre)

# Use the prestress solution as the intial slider positions for AFT as well
vib_sys.set_aft_initialize(Xpre)

# Reset to real friction coefficient after updating frictionless slider
# positions
vib_sys.reset_real_mu()

###############################################################################
####### Updated Eigenvalue Analysis After Prestress                     #######
###############################################################################

# Recalculate stiffness with real mu
Rpre, dRpredX = vib_sys.static_res(Xpre, Fv*prestress)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

print('Using using  (Kpre + Kpre.T)/2 version for eigen analysis')

Kpre = (dRpredX + dRpredX.T) / 2.0

eigvals, eigvecs = static_solver.eigs(Kpre, system_matrices['M'], 
                                      subset_by_index=[0, 9])


print('Prestress State Frequencies: [Hz]')
print(np.sqrt(eigvals)/(2*np.pi))

# Mass normalize eigenvectors
norm = np.diag(eigvecs.T @ system_matrices['M'] @ eigvecs)
eigvecs = eigvecs / np.sqrt(norm)

# Displacement at accel for eigenvectors
resp_amp = system_matrices['R'][2, :] @ eigvecs
print('Response amplitudes at tip accel: [m]')
print(resp_amp)

print('Expected frequencies from previous MATLAB / Paper (Flat Mesoscale):'\
      +' 168.5026, 580.4082, 1177.6498 Hz')


###############################################################################
####### EPMC Initial Guess                                              #######
###############################################################################

h = np.array(range(h_max+1))

Nhc = hutils.Nhc(h)

Ndof = vib_sys.M.shape[0]

Fl = np.zeros(Nhc*Ndof)

# Static Forces
Fl[:Ndof] = prestress*Fv
Fl[Ndof:2*Ndof] = system_matrices['R'][2, :] # No cosine component at accel

Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Static Displacements
Uwxa0[:Ndof] = Xpre

# Mode Shape
mode_ind = 0
Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping
zeta = damp_ab[0] / 2 / Uwxa0[-3] 
Uwxa0[-2] = 2*Uwxa0[-3]*zeta

# Amplitude
Uwxa0[-1] = Astart


###############################################################################
####### Profile Single EPMC Residual                                    #######
###############################################################################

t0 = time.time()

R = vib_sys.epmc_res(Uwxa0, Fl, h, Nt=Nt, calc_grad=True)[0]

str(R[0]) # This forces JAX operations to block

t1 = time.time()

print('EPMC Residual Run Time (with gradient): {: 7.3f} s'.format(t1 - t0))

t0 = time.time()

R = vib_sys.epmc_res(Uwxa0, Fl, h, Nt=Nt, calc_grad=False)[0]

str(R[0]) # This forces JAX operations to block

t1 = time.time()

print('EPMC Residual Run Time (without gradient): {: 7.3f} s'.format(t1 - t0))


###############################################################################
####### EPMC Continuation                                               #######
###############################################################################

epmc_fun = lambda Uwxa, calc_grad=True : vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt, calc_grad=calc_grad)

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 4,
                   'MaxSteps'   : 40,
                   'dsmin'      : dsmin,
                   'dsmax'      : dsmax,
                   'verbose'    : 1,
                   'xtol'       : 1e-6*np.sqrt(Uwxa0.shape[0]), 
                   'corrector'  : 'Ortho', # Ortho, Pseudo
                   'nsolve_verbose' : True,
                   'callback' : callback_funs,
                   'FracLam' : FracLam,
                   'FracLamList' : [0.9, 0.1, 1.0, 0.0],
                   'backtrackStop' : 0.05 # stop if it backtracks to before start.
                   }

CtoPstatic = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-5)
CtoP = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-3) # Increasing delta means that these cofficients will be smaller in conditioned space.

CtoP[:Ndof] = CtoPstatic[:Ndof] # Allow different CtoP for static displacements than harmonics.
CtoP[-3:-1] = np.abs(Uwxa0[-3:-1]) # Exactly take damping and frequency regardless of delta
CtoP[-1] = np.abs(Aend-Astart)


cont_solver = Continuation(epmc_solver, ds0=ds, CtoP=CtoP, config=continue_config)

if run_profilers:
    
    cProfile.run('Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)', 
                 continue_profile)
    
    print('Continuation run time saved to {}.'.format(continue_profile))
        
else:
    
    t0 = time.time()
    
    Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)
    
    t1 = time.time()
    
    print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))

if Uwxa_full[-1][-1] < Aend and epmc_config['reform_freq'] > 1:
    print("If continuation did not complete, you may want to try again with"\
          +"a smaller number for epmc_config['reform_freq'].")

###############################################################################
####### Open Debug Terminal at End for User to Query Variables          #######
###############################################################################

# # This needs to be uncommented to open up the debug terminal.
# import pdb; pdb.set_trace()
