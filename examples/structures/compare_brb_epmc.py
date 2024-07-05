"""
Verification routine to compare results from './brb_epmc.py' to original 
results from the paper: 
    
    Porter, J.H., Brake, M.R.W., 2023. Towards a predictive, physics-based 
    friction model for the dynamics of jointed structures. Mechanical Systems 
    and Signal Processing 192, 110210. 
    https://doi.org/10.1016/j.ymssp.2023.110210

Original results were generated in MATLAB and saved to yaml file: 
    './brb_epmc_flat.yaml' and './brb_epmc_meso.yaml' for cases with plasticity
    and either a flat or curved mesoscale topology respectively. Both with 
    friction coefficients of 0.03.
    
This script should be run with the same command line arguments as brb_epmc.py
"""

import sys
import numpy as np
from scipy import io as sio # Loading accel extraction matrix
import yaml # Import for the reference data
import argparse # parse command line arguments

import matplotlib.pyplot as plt

sys.path.append('../..')
import tmdsimpy.postprocess.continuation as cpost


###############################################################################
####### Command Line Defaults                                           #######
###############################################################################

# These defaults can be changed if running in an IDE without giving command
# line arguments to look at different systems

# Set this to 1 to use mesoscale or 0 to not use mesoscale by default
# Command line input will override this if given.
default_mesoscale = 1 

# Default mesh name. Command line input will override this if given
default_sys_fname = './data/BRB_ROM_U_122ELS4py.mat'


###############################################################################
####### Command Line Inputs Parsing                                     #######
###############################################################################

# Do not edit these to change which system is run

parser = argparse.ArgumentParser()
parser.add_argument("-meso", "--meso_scale_included", type=int, nargs='?', 
                    const=1, default=default_mesoscale)

parser.add_argument("-system", "--system_filename", type=str, nargs='?', 
                    const=1, default=default_sys_fname)

args = parser.parse_args()
mesoscale_TF = args.meso_scale_included != 0
system_fname = args.system_filename

print('Using system from file: {}'.format(system_fname))
print('Comparing results with mesoscale topology? {}'.format(mesoscale_TF))

###############################################################################
####### User Inputs                                                     #######
###############################################################################

# YAML file with backbone information from previous paper that is used as 
# reference solution for verifying the correctness of the present code
if mesoscale_TF:
    reference_yaml = './results/brb_epmc_meso.yaml'
else: 
    reference_yaml = './results/brb_epmc_flat.yaml'

    

# Matrices that were used in the new solution. These are only needed to get
# the matrix for extracting the accelerometer position that was used in 
# plotting for the previous paper.
matrices_for_new = system_fname

# EPMC results saved during continuation for the new simulation.
if mesoscale_TF:
    new_results = './results/brb_epmc_meso_full.npz'
else: 
    new_results = './results/brb_epmc_flat_full.npz'


###############################################################################
####### Load Reference Data from Rough Contact Model Paper              #######
###############################################################################

with open(reference_yaml, 'r') as file:
    ref_dict = yaml.safe_load(file)

###############################################################################
####### Load Extraction Matrix                                          #######
###############################################################################

system_matrices = sio.loadmat(matrices_for_new)

recover_vec = np.asarray(system_matrices['R'][5, :])


###############################################################################
####### Load and Interpolate EPMC Results                               #######
###############################################################################

###  Load 
epmc_py = np.load(new_results)

XlamP = epmc_py['XlamP'][:-1, :]

XlamP_grad = epmc_py['dirP_prev'][1:, :]


### Upsample for Smooth Plotting
XlamP_line = cpost.hermite_upsample(XlamP, XlamP_grad, upsample_freq=50)

### Interpolate for Error Checking
XlamP_err = cpost.hermite_interp(XlamP, XlamP_grad, 
                                 np.log10(ref_dict['modal_amplitude'][:-1]))



###############################################################################
####### Convert EPMC Results to Quantities of Interest                  #######
###############################################################################

    
Ndof = recover_vec.shape[0]

##########
# Calculate frequency and damping values

XlamP_set = [XlamP, XlamP_line, XlamP_err]

freq = len(XlamP_set) * [None]
zeta = len(XlamP_set) * [None]
modal_q = len(XlamP_set) * [None]
mode_shape_disp = len(XlamP_set) * [None]

for ind in range(len(XlamP_set)):

    XlamP_curr = XlamP_set[ind]    

    freq[ind] = XlamP_curr[:, -3]
    zeta[ind] = XlamP_curr[:, -2] / 2.0 / XlamP_curr[:, -3]
    
    modal_q[ind] = 10**XlamP_curr[:, -1]
    
    # Assumes that the 0th harmonic is included (friction would probably fail 
    # without this)
    Q1c = XlamP_curr[:, Ndof:2*Ndof] @ recover_vec
    Q1s = XlamP_curr[:, 2*Ndof:3*Ndof] @ recover_vec
    
    mode_shape_disp[ind] = np.sqrt(Q1c**2 + Q1s**2)


###############################################################################
####### Calculate Error Metrics                                         #######
###############################################################################

freq_err = np.abs(np.array(ref_dict['frequency_rad_per_s'])[:-1] - freq[2]) \
            / np.array(ref_dict['frequency_rad_per_s'])[:-1]

damp_err = np.abs(np.array(ref_dict['damping_factor_frac_crit'])[:-1] - zeta[2]) \
            / np.array(ref_dict['damping_factor_frac_crit'])[:-1]
            

mode_disp_ref = np.array(ref_dict['displacement_amplitude']) \
                    /np.array(ref_dict['modal_amplitude'])
                    
mode_disp_err = np.abs(mode_disp_ref[:-1] - mode_shape_disp[2]) \
                    / mode_disp_ref[:-1]
                    
print('Maximum Frequency: {: 6.4f} % Error'.format(freq_err.max()*100))
print('Maximum Damping: {: 6.4f} % Error'.format(damp_err.max()*100))

print('Maximum Damping in Friction Dominated: {: 6.4f} % Error'.format(\
                               damp_err[zeta[2] > 3e-3].max()*100))

print('Maximum Accel Modal Displacement: {: 6.4f} % Error'.format(\
                                              mode_disp_err.max()*100))

if mesoscale_TF:
    print('Expected Errors for 232 ZTE (same mesh, previous order):'\
          +' (0.0719, 5.5288, 0.0411, 0.0742)%')
    
    print('Expected Errors for 122 ZTE (reduced mesh, previous order):'\
          +' (0.1501, 92.5023*, 2.6476, 0.0762)% *Due to interpolation.')
        
    
else:
    print('Expected Errors for 232 ZTE (same mesh, previous order):'\
          +' (0.0707, 2.0150, 0.1290, 0.0715)%')
    
    print('Expected Errors for 122 ZTE (reduced mesh, previous order):'\
          +' (0.1491, 4.2942, 3.1760, 0.0806)%')
    
###############################################################################
####### Plot Comparisons                                                #######
###############################################################################


####### Frequency 

plt.plot(np.array(ref_dict['modal_amplitude']), 
         np.array(ref_dict['frequency_rad_per_s'])/2.0/np.pi,
         '-x',
         label='Porter and Brake (2023)')

p = plt.plot(modal_q[0], freq[0]/2.0/np.pi, 'o', label='TMDSimPy')
plt.plot(modal_q[1], freq[1]/2.0/np.pi, '--', 
         label='TMDSimPy - Hermite Interp', 
         color=p[0].get_color())

ax = plt.gca()
ax.set_xscale('log')

plt.legend()
plt.xlabel('Modal Amplitude')
plt.ylabel('Frequency [Hz]')
plt.show()


####### Damping

plt.plot(np.array(ref_dict['modal_amplitude']), 
         np.array(ref_dict['damping_factor_frac_crit']),
         '-x',
         label='Porter and Brake (2023)')

p = plt.plot(modal_q[0], zeta[0], 'o', label='TMDSimPy')
plt.plot(modal_q[1], zeta[1], '--', label='TMDSimPy - Hermite Interp', 
         color=p[0].get_color())

ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')

plt.legend()
plt.xlabel('Modal Amplitude')
plt.ylabel('Fraction Critical Damping')
plt.show()

####### Modal Amplitude versus Accelerometer Amplitude 
# This checks that the mode shape evolves in a similar way by checking the 
# extracted degree of freedom for the accelerometer that was used for plotting
# in the original paper. 

mode_disp_ref = np.array(ref_dict['displacement_amplitude']) \
                    /np.array(ref_dict['modal_amplitude'])

plt.plot(np.array(ref_dict['modal_amplitude']), mode_disp_ref,
            '-x', label='Porter and Brake (2023)')

p = plt.plot(modal_q[0], mode_shape_disp[0], 'o', label='TMDSimPy')

plt.plot(modal_q[1], mode_shape_disp[1], '--', label='TMDSimPy - Hermite Interp', 
         color=p[0].get_color())

ax = plt.gca()
ax.set_xscale('log')

plt.legend()
plt.xlabel('Modal Amplitude')
plt.ylabel('Mass Norm. Modal Displacement at Accel')
plt.show()
