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
"""

import sys
import numpy as np
from scipy import io as sio # Loading accel extraction matrix
import yaml # Import for the reference data

import matplotlib.pyplot as plt


sys.path.append('../..')
import tmdsimpy


###############################################################################
####### User Inputs                                                     #######
###############################################################################

# YAML file with backbone information from previous paper that is used as 
# reference solution for verifying the correctness of the present code
reference_yaml = './results/brb_epmc_flat.yaml'

# Matrices that were used in the new solution. These are only needed to get
# the matrix for extracting the accelerometer position that was used in 
# plotting for the previous paper.
matrices_for_new = './matrices/ROM_U_232ELS4py.mat'

# EPMC results saved during continuation for the new simulation.
new_results = './results/brb_epmc_bb_full.npz'


###############################################################################
####### Load Reference Data from Rough Contact Model Paper              #######
###############################################################################

with open(reference_yaml, 'r') as file:
    ref_dict = yaml.safe_load(file)

###############################################################################
####### Load Extraction Matrix                                          #######
###############################################################################

system_matrices = sio.loadmat(matrices_for_new)

recover_vec = np.asarray(system_matrices['R'][2, :])


###############################################################################
####### Load and Convert EPMC Results                                   #######
###############################################################################

epmc_py = np.load(new_results)

XlamP = epmc_py['XlamP'][:-1, :]

##########
# Calculate frequency and damping values

freq = XlamP[:, -3]
zeta = XlamP[:, -2] / 2.0 / XlamP[:, -3]

modal_q = 10**XlamP[:, -1]




###############################################################################
####### Plot Comparisons                                                #######
###############################################################################

plt.plot(np.array(ref_dict['modal_amplitude']), 
         np.array(ref_dict['frequency_rad_per_s'])/2.0/np.pi,
         '-x',
         label='Porter and Brake (2023)')

plt.plot(modal_q, freq/2.0/np.pi, 'o--', label='TMDSimPy')

ax = plt.gca()
ax.set_xscale('log')

plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Modal Amplitude')
plt.show()



plt.plot(np.array(ref_dict['modal_amplitude']), 
         np.array(ref_dict['damping_factor_frac_crit']),
         '-x',
         label='Porter and Brake (2023)')

plt.plot(modal_q, zeta, 'o--', label='TMDSimPy')

ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')

plt.legend()
plt.xlabel('Fraction Critical Damping')
plt.ylabel('Modal Amplitude')
plt.show()




