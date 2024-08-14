"""
This is an example script for a simulation of the Brake-Reuss Beam (BRB)
utilizing an advanced rough contact model [1]_. 

Parameters 
----------
    -meso or --meso_scale_included: int, optional
        Command line argument. If 0 then mesoscale topology is not used. 
        Otherwise, it will used loaded mesoscale topology. Default behavior
        set in script by `default_mesoscale`.
    -system or --system_filename: str, optional
        Command line argument. Filename for the system matrices. 
        Default behavior set in script by `default_sys_fname` (122 ZTE model).
        Running a second time with a different system file will likely cause
        an error unless you delete the saved outputs of this function. 
        See variables `epmc_full_name` and `epmc_dat` for output filenames.
    -p or --profile_run: int, optional
        Command line argument. If 0, the profiler will not be run. If nonzero
        argument is passed then the execution will be profiled with cProfile.
        Default behavior set in script by `default_run_profilers`.

Notes
-----

This script can be directly run, or it can be called from the command line as:
    
    `python3 -u brb_epmc.py -meso 1`
    
where after -meso, you can put 1 to include mesoscale topology or 0 to do a 
flat interface. Additional command line parameters documented above can also
be used.
Additional discussion about running this script is provided in the README.md
as well as discussion about verify the results.

Summary of Key Model Details: 
    Friction Model: 
        Physics-Based Rough Contact [1]_
    Nonlinear Modal Analysis: 
        Extended Periodic Motion Concept [2]_
    Structural Model: 
        122 Zero Thickness Elements (ZTEs) Mesh (or download
        232 ZTE model from location listed in README.md) [3]_
        122 ZTE model provided in file `data/BRB_ROM_U_122ELS4py.mat`
    Surface Parameters: 
        Surface parameters for rough contact are identified in [1]_. 
        Parameters are provided in file `data/combined_14sep21_R1_4py.mat`
        
List of execution steps (numbered code blocks below). Steps 1-7 are loading
all of the data and checking that it is correct. Steps 8 and 9 create the model
Steps 10-13 and 15 do all of the needed computational steps. Step 14 is just
for understanding the computation time. Step 16 is optional and commented out.
    1. Command Line Defaults 
    2. User Inputs 
    3. Friction Model Parameters 
    4. Solver Settings 
    5. Command Line Inputs Parsing 
    6. EPMC Output Save Information 
    7. Load System Matrices from .mat File  
    8. Create Vibration System  <----------------This starts creating the model
    9. Add Nonlinear Forces to System  
    10. Prestress Analysis <------------This is the first nonlinear calculation
    11. Updated Eigenvalue Analysis After Prestress
    12. Updated Damping Matrix After Prestress
    13. EPMC Initial Guess
    14. Time Single EPMC Residual
    15. EPMC Continuation <------------------------Main computation of interest
    16. Open Debug Terminal at End for User to Query Variables 
        (optional, uncomment to use)

Parallelism: 
    The execution uses numpy and jax.numpy libraries for various
    linear algebra operations. The numpy operations respond to OpenMP 
    environment variables and thus you can set the number of used threads in 
    a linux terminal with:
    
    >> export OMP_PROC_BIND=spread # Spread threads out over physical cores
    >> export OMP_NUM_THREADS=32 # Change 32 to desired number of threads

    The JAX evaluations of the nonlinear forces are JIT compiled and are called
    in a non-blocking fashion by default. In practical terms, this means all
    threads will be used to simultaneously execute the nonlinear force 
    evaluations.
    
Simulations with 122 ZTE model take about 5-10 minutes on Computer with 
12 cores, 24 threads, 32 GB RAM, 2.1 GHz processor

See Also
--------

Documentation in individual functions, classes, and methods called here 
provides additional description of the usage. 

References
----------
.. [1] Porter, Justin H., and Matthew R. W. Brake. "Towards a Predictive, 
   Physics-Based Friction Model for the Dynamics of Jointed Structures." 
   Mechanical Systems and Signal Processing 192 (June 1, 2023): 110210.
   https://doi.org/10.1016/j.ymssp.2023.110210.
   
.. [2] Krack, Malte. "Nonlinear Modal Analysis of Nonconservative Systems: 
    Extension of the Periodic Motion Concept." Computers & Structures 154 
    (July 1, 2015): 59–71. https://doi.org/10.1016/j.compstruc.2015.03.008.
    
.. [3] Balaji, Nidish Narayanaa, Tobias Dreher, Malte Krack, and Matthew R. W. 
   Brake. "Reduced Order Modeling for the Dynamics of Jointed Structures 
   through Hyper-Reduced Interface Representation.” Mechanical Systems and 
   Signal Processing 149 (February 15, 2021): 107249. 
   https://doi.org/10.1016/j.ymssp.2020.107249.

"""

import sys
import numpy as np
from scipy import io as sio
from scipy.interpolate import LinearNDInterpolator
import warnings
import time
import argparse # parse command line arguments
    
sys.path.append('../..')
import tmdsimpy.utils.harmonic as hutils

from tmdsimpy.jax.solvers import NonlinearSolverOMP

from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.continuation as cont_utils

from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.jax.nlforces.roughcontact.rough_contact import RoughContactFriction

###############################################################################
####### 1. Command Line Defaults                                        #######
###############################################################################

# These defaults can be changed if running in an IDE without giving command
# line arguments to look at different systems

# Set this to 1 to use mesoscale or 0 to not use mesoscale by default
# Command line input will override this if given. Example: 
# python3 -u brb_epmc.py -meso 1
default_mesoscale = 1 

# default filename for a .mat file that contains the matrices appropriately 
# formatted to be able to describe the structural system. 
# Command line input will override this if given
# python3 -u brb_epmc.py -system './data/BRB_ROM_U_232ELS4py.mat'
default_sys_fname = './data/BRB_ROM_U_122ELS4py.mat'

# Flag for running profiler. Command line argument will override this flag
# If flag is not zero, then the code simulation will be profiled. Otherwise, 
# it will not be profiled. Example:
# python3 -u brb_epmc.py -p 1
default_run_profilers = 0 

# Default value for fast solution flag. If not zero, then a fast solution will
# be calculated using reduced harmonics and AFT steps. Example
# python3 -u brb_epmc.py -f 1
default_fast_sol = 0

###############################################################################
####### 2. User Inputs                                                  #######
###############################################################################

# These can be modified, but likely are fine as they are given here. Modifying
# these parameters will likely mean that the results cannot be verified
# with the script 'compare_brb_epmc.py' after running. 

# Log10 Amplitude Start (mass normalized modal amplitude)
Astart = -7.7

# Log10 Amplitude End
Aend = -4.2

# Choose speed or accuracy - edit other parameters based on which value
# of this flag you are using.
fast_sol = False 

if fast_sol:
    # Run with reduced harmonics and AFT time steps to keep time within a 
    # few minutes
    h_max = 1 # harmonics 0 and 1
    Nt = 1<<3 # 2**3 = 8 AFT steps 
    FracLam = 0.5 # Continuation weighting
    
    ds = 0.1
    dsmax = 0.2
    dsmin = 0.02
else:
    # Normal - settings for higher accuracy as used in previous papers
    h_max = 3 # harmonics 0, 1, 2, 3
    Nt = 1<<7 # 2**7 = 128 AFT steps 
    
    ds = 0.08
    dsmax = 0.125*1.4
    dsmin = 0.02
    
    # Adjust weighting of amplitude v. other in continuation to hopefully 
    # reduce turning around. Higher puts more emphasis on continuation 
    # parameter (amplitude)
    FracLam = 0.50     

###############################################################################
####### 3. Friction Model Parameters                                    #######
###############################################################################

# These can be modified, but likely are fine as they are given here. Modifying
# these parameters will likely mean that the results cannot be verified
# with the script 'compare_brb_epmc.py' after running. 

# Surface Parameters for the rough contact model - from ref [1]
surface_fname = './data/brb_surface_data.mat'

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

mesoscale_xygap = surface_pars['mesoscale_xygap']

###############################################################################
####### 4. Solver Settings                                              #######
###############################################################################

# These solver settings are likely fine, and should probably not be changed.

########################################
# Static Solver settings

# Estimated displacements. Because the rough contact stiffness goes to zero
# when contact is at zero displacement, an initial estimate of the normal 
# displacement is needed to get an initial stiffness to generate an initial
# guess for the nonlinear solver. 
# This displacement should be large enough that at least one asperity with a
# non-zero weight is in contact.
# The value of 0.1e-5 should be reasonable for the BRB here.
uxyn_est = np.array([0, 0, .1e-5])

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

# Custom Newton-Raphson solver
static_solver = NonlinearSolverOMP(config=static_config) 


########################################
# EPMC Solver settings

epmc_config={'max_steps' : 12, # balance with reform_freq
            'reform_freq' : 2, #>1 corresponds to BFGS 
            'verbose' : True, 
            'xtol'    : None, # Just use the one passed from continuation
            'rtol'    : 1e-9,
            'etol'    : None,
            'xtol_rel' : 1e-6, 
            'rtol_rel' : None,
            'etol_rel' : None,
            'stopping_tol' : ['xtol'], # stop on xtol 
            'accepting_tol' : ['xtol_rel', 'rtol'] # accept solution on these
            }

# Custom Newton-Raphson solver
epmc_solver = NonlinearSolverOMP(config=epmc_config)

###############################################################################
####### 5. Command Line Inputs Parsing                                  #######
###############################################################################

# Do not edit these to change which system is run. This just processes based
# on what is already passed to this script from the command line and the 
# defaults set above

parser = argparse.ArgumentParser()
parser.add_argument("-meso", "--meso_scale_included", type=int, nargs='?', 
                    const=1, default=default_mesoscale)

parser.add_argument("-system", "--system_filename", type=str, nargs='?', const=1, 
                    default=default_sys_fname)

parser.add_argument("-p", "--profile_run", type=str, nargs='?', const=1, 
                    default=default_run_profilers)

args = parser.parse_args()
mesoscale_TF = args.meso_scale_included != 0
system_fname = args.system_filename
run_profilers = args.profile_run != 0

print('Using system from file: {}'.format(system_fname))
print('Mesoscale topology will be used? {}'.format(mesoscale_TF))
print('Profiler will be run? {}'.format(run_profilers))

###############################################################################
####### 6. EPMC Output Save Information                                 #######
###############################################################################

if mesoscale_TF:
    
    # Detailed full output (numpy binary)
    epmc_full_name = './results/brb_epmc_meso_full.npz' 
    
    # Summary file output (text file)
    epmc_dat = './results/brb_epmc_meso_sum.dat' 
    
    # File to save static analysis profiling
    static_profile = './results/static_meso_profile' 
    
    # File to save continuation profiling in
    continue_profile = './results/continue_meso_profile' 

else:
    
    # Detailed full output (numpy binary)
    epmc_full_name = './results/brb_epmc_flat_full.npz' 
    
    # Summary file output (text file)
    epmc_dat = './results/brb_epmc_flat_sum.dat' 
    
    # File to save static analysis profiling
    static_profile = './results/static_flat_profile' 
    
    # File to save continuation profiling in
    continue_profile = './results/continue_flat_profile' 

call_list = [lambda XlamP, dirP_prev : cont_utils.continuation_save(XlamP, 
                                                dirP_prev, epmc_full_name),
             lambda XlamP, dirP_prev : cont_utils.print_epmc_stats(XlamP, 
                                                          dirP_prev, epmc_dat)]

callback_funs = lambda XlamP, dirP_prev : cont_utils.combine_callback_funs(\
                                                   call_list, XlamP, dirP_prev)

###############################################################################
####### 7. Load System Matrices from .mat File                          #######
###############################################################################

# This block is just sanity checks that the system is what is expected. 
# This should not be editted. 

system_matrices = sio.loadmat(system_fname)

######## Sanity Checks on Loaded Matrices

# Sizes
assert system_matrices['M'].shape == system_matrices['K'].shape, \
       'Mass and stiffness matrices are not the same size, this will not work.'

if not (system_matrices['M'].shape == (481, 481)):
    warnings.warn("Warning: Mass and stiffness matrices are not the expected "\
                  "size for the UROM 122 Model.")

# Approximate Frequencies Without Contact
# If running a different ROM, these will vary slightly

eigvals, eigvecs = static_solver.eigs(system_matrices['K'], 
                                      system_matrices['M'], 
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
area_error = np.abs(system_matrices['Tm'].sum() - ref_area) / ref_area

assert area_error < 2e-3, \
        'Quadrature integration matrix gives wrong contact area.'

###############################################################################
####### 8. Create Vibration System                                      #######
###############################################################################

# This is a somewhat arbitrary damping value for initialization. However, 
# this later gets set to match 1st and 2nd bending mode damping in block #12
damp_ab = [0.087e-2*2*(168.622*2*np.pi), 0.0]

vib_sys = VibrationSystem(system_matrices['M'], system_matrices['K'], 
                          ab=damp_ab)

###############################################################################
####### 9. Add Nonlinear Forces to System                               #######
###############################################################################

# Number of nonlinear frictional elements, Number of Nodes
Nnl,Nnodes = system_matrices['Qm'].shape 

# Need to convert sparse loads into arrays so that operations are expected shapes
# Sparse matrices from matlab are loaded as matrices rather than numpy arrays
# and behave differently than numpy arrays.
Qm = np.array(system_matrices['Qm'].todense()) 
Tm = np.array(system_matrices['Tm'].todense())

# Pull out for reference convenience - null space transformation matrix
L  = system_matrices['L']

QL = np.kron(Qm, np.eye(3)) @ L[:3*Nnodes, :]
LTT = L[:3*Nnodes, :].T @ np.kron(Tm, np.eye(3))


# Calculate the mesoscale gaps of each node point
interp_obj = LinearNDInterpolator(mesoscale_xygap[:, :2], # x, y
                                  mesoscale_xygap[:, 2]) # gaps

meso_gap_nodes = interp_obj(system_matrices['node_coords'][:, 0], # node x
                            system_matrices['node_coords'][:, 1]) # node y

# interpolate mesoscale at nodes to quadrature points
meso_gap_quads = Qm @ meso_gap_nodes

# move so something is initially in contact
meso_gap_quads = meso_gap_quads - meso_gap_quads.min() 

# Set mesoscale to zero if not using it
meso_gap_quads = mesoscale_TF * meso_gap_quads 

for i in range(Nnl):
    
    Ls = (QL[i*3:(i*3+3), :])
    Lf = (LTT[:, i*3:(i*3+3)])

    tmp_nl_force = RoughContactFriction(Ls, Lf, ElasticMod, PoissonRatio, 
                                        Radius, TangentMod, YieldStress, mu,
                                        gaps=gaps, gap_weights=gap_weights,
                                        meso_gap=meso_gap_quads[i])
    
    vib_sys.add_nl_force(tmp_nl_force)
    
    
# Create a reference nonlinear element that can be used for initial guesses
# e.g., to extract an estimate of the stiffness
ref_nlforce = RoughContactFriction(np.eye(3), np.eye(3), ElasticMod, 
                                   PoissonRatio, Radius, TangentMod, 
                                   YieldStress, mu,
                                   gaps=gaps, gap_weights=gap_weights)

###############################################################################
####### 10. Prestress Analysis                                          #######
###############################################################################

# For prestress analysis, set the friction coefficient to zero. 
# This will change results and is related to broader questions about residual 
# tractions.
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
    
    cProfile.run('Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0, '\
                 +'verbose=True, xtol=1e-13)', 
                 static_profile)
    
    print(('Static run time saved to {}. This can be loaded and '\
           +'investigated.').format(static_profile))
        
    print('See https://docs.python.org/3/library/profile.html for more.')
    
    """
    # Load and investigate profile: 
    import pstats
    from pstats import SortKey
    p = pstats.Stats('static_meso_profile')
    p.sort_stats(SortKey.TIME).print_stats(10)
    """
    
else:
    t0 = time.time()
    Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                              verbose=True, xtol=1e-13)
    
    t1 = time.time()
    
    print('Static Solution Run Time : {:.3e} s'.format(t1 - t0))

print('Residual norm: {:.4e}'.format(np.linalg.norm(R)))

# Update history variables after static so sliders reset
# If you do not do this, then the residual traction field will be different.
vib_sys.update_force_history(Xpre)

# Use the prestress solution as the intial slider positions for AFT as well
# This influences residual tractions and may slightly change the results of
# the simulation.
vib_sys.set_aft_initialize(Xpre)

# Reset to real friction coefficient after updating frictionless slider
# positions
# This is needed so that the friction coefficient is used in EPMC 
# (rather than 0 tangential forces)
vib_sys.reset_real_mu()

###############################################################################
####### 11. Updated Eigenvalue Analysis After Prestress                 #######
###############################################################################

# Recalculate stiffness with real mu (including stiffness from friction)
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
print('Response amplitudes at tip accel '\
      + '(magnitudes near 1 indicate bending modes): [m]')
print(resp_amp)

print('Expected bending mode frequencies from previous MATLAB / Paper'\
      +' (Flat Mesoscale): 168.5026, 580.4082, 1177.6498 Hz')

print('Expected bending mode frequencies from previous MATLAB / Paper'\
      +' (With Mesoscale): 180.6293, 595.2815, 1199.9013 Hz')

###############################################################################
####### 12. Updated Damping Matrix After Prestress                      #######
###############################################################################

# This block resets the damping matrix after prestress analysis to 
# achieve the desired levels of viscous linear damping for the first and second
# bending modes

# First and Second Bending Modes damping ratios (taken from experiments on BRB)
desired_zeta = np.array([0.087e-2, 0.034e-2]) 

# 1st and 2nd bending mode = total 1st and 3rd modes
omega_12 = np.array([np.sqrt(eigvals)[0:3:2]]).reshape(2, 1) 

# Matrix problem for proportional damping
prop_mat = np.hstack((1/(2.0*omega_12), omega_12/2.0))

pre_ab = np.linalg.solve(prop_mat, desired_zeta)

vib_sys.set_new_C(C=pre_ab[0]*vib_sys.M + pre_ab[1]*Kpre)

###############################################################################
####### 13. EPMC Initial Guess                                          #######
###############################################################################

h = np.array(range(h_max+1))

Nhc = hutils.Nhc(h)

Ndof = vib_sys.M.shape[0]

Fl = np.zeros(Nhc*Ndof)

# Static Forces
Fl[:Ndof] = prestress*Fv # EPMC static force

# EPMC phase constraint - No cosine component at accel
Fl[Ndof:2*Ndof] = system_matrices['R'][2, :] 

Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Static Displacements (prediction for 0th harmonic)
Uwxa0[:Ndof] = Xpre

# Mode Shape (from linearized system for prediction)
mode_ind = 0
Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = desired_zeta[0] # This is what mass/stiff prop damping should give
Uwxa0[-2] = 2*Uwxa0[-3]*zeta

# Amplitude (Desired starting amplitude)
Uwxa0[-1] = Astart

###############################################################################
####### 14. Time Single EPMC Residual                                   #######
###############################################################################

# This block is not necessary for the computation, but is useful for 
# understanding the computational costs

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
####### 15. EPMC Continuation                                           #######
###############################################################################

# This block actually executes the full continuation for the EPMC solution.

epmc_fun = lambda Uwxa, calc_grad=True : vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt, 
                                                          calc_grad=calc_grad)

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 4,
                   'MaxSteps'   : 40, # May need more depending on ds and dsmin
                   'dsmin'      : dsmin,
                   'dsmax'      : dsmax,
                   'verbose'    : 1,
                   'xtol'       : 1e-6*np.sqrt(Uwxa0.shape[0]), 
                   'corrector'  : 'Ortho', # Ortho, Pseudo
                   'nsolve_verbose' : True,
                   'callback' : callback_funs,
                   'FracLam' : FracLam,
                   'FracLamList' : [0.9, 0.1, 1.0, 0.0],
                   'backtrackStop' : 0.05 # stop if backtracks to before lam0
                   }

# The conditioning of the static displacements should be small since these
# displacements are small, but very important
CtoPstatic = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-5)

# Increasing delta means that these cofficients will be smaller in conditioned 
# space. - reduces importance of higher harmonics when calculating arc length
CtoP = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-3) 

# Allow different CtoP for static displacements than harmonics.
CtoP[:Ndof] = CtoPstatic[:Ndof] 

# Exactly take damping and frequency regardless of delta for conditioning
CtoP[-3:-1] = np.abs(Uwxa0[-3:-1]) 

# scale so step size is similar order as fraction of total distance from 
# start to end
CtoP[-1] = np.abs(Aend-Astart) 

cont_solver = Continuation(epmc_solver, ds0=ds, CtoP=CtoP, 
                           config=continue_config)

if run_profilers:
    
    cProfile.run('Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, '\
                 +'Astart, Aend)', 
                 continue_profile)
    
    print('Continuation computation time profile saved to '\
          + '{}.'.format(continue_profile))
        
else:
    
    t0 = time.time()
    
    Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)
    
    t1 = time.time()
    
    print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))

if Uwxa_full[-1][-1] < Aend and epmc_config['reform_freq'] > 1:
    print("If continuation did not complete, you may want to try again with"\
          +"a smaller number for epmc_config['reform_freq'].")

###############################################################################
####### 16. Open Debug Terminal at End for User to Query Variables      #######
###############################################################################

# # This needs to be uncommented to open up the debug terminal.
# import pdb; pdb.set_trace()
