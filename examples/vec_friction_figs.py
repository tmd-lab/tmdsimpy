"""
Figures showing the procedure for the vectorized Jenkins algorithm
"""

import sys
import numpy as np

sys.path.append('..')

from tmdsimpy.nlforces.vector_jenkins import VectorJenkins
from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4


import tmdsimpy.utils.harmonic as hutils


import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3


# plt.rcParams['text.usetex'] = True
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')

mpl.style.use('seaborn-v0_8-colorblind')

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14 # Default 10


###############################################################################
###  Jenkins Algorithm Evaluations                                          ###
###############################################################################


# Simple Mapping to displacements
Q = np.array([[1.0]])
T = np.array([[1.0]])

# # Jenkins parameters
# kt = 2.0
# Fs = 3.0

# vec_force = VectorJenkins(Q, T, kt, Fs)


# Iwan model parameters

kt = 0.25 # N/m, Match Jenkins
Fs = 0.2 # N, Match Jenkins
chi = -0.5 # Have a more full hysteresis loop than chi=0.0
beta = 0.0 # Smooth Transition
Nsliders = 100

vec_force = VectorIwan4(Q, T, kt, Fs, chi, beta, Nsliders=Nsliders, alphasliders=1.0)



# vec_force.init_history(unlth0=0)

Nt = 1 << 7

w = 1.7

h = np.array([0, 1, 2, 3])
Unl = np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0]]).T
unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl

cst = hutils.time_series_deriv(Nt, h, np.eye(hutils.Nhc(h)), 0)

    
Fnl, dFnldU = vec_force.aft(Unl, w, h, Nt=Nt)[0:2]

ft = vec_force.local_force_history(unlt, unlt*0.0, h, cst, Unl[0])[0]

###############################################################################
###  Plotting History                                                       ###
###############################################################################

tau = np.linspace(0, 1, Nt+1)
ft = np.hstack((ft[:, 0], ft[0]))
unlt = np.hstack((unlt[:, 0], unlt[0]))


# Critical Points
dup = unlt - np.roll(unlt, 1, axis=0) # du to current
dun = np.roll(unlt, -1, axis=0) - unlt # du to next

vector_set = np.equal(np.sign(dup), np.sign(dun))
vector_set[0] = False

vector_set = np.logical_not(vector_set)


# Plots of Critical points and Other

fig, axs = plt.subplots(2, gridspec_kw={'hspace': 0.05, 'wspace': 0.2})

axs[0].plot(tau, unlt, label='Total Motion')
axs[0].set_xlim((0, 1))
axs[0].set_ylim((-11, 11))
axs[0].set_ylabel('$\\boldsymbol{X}$ [m]')
# axs[0].legend(framealpha=1.0, frameon=False)
axs[0].set_title('Displacements', pad=-20)

axs[1].plot(tau, ft, color='0.8')
axs[1].plot(tau, ft, '.', label='All Points')
axs[1].plot(tau[vector_set], ft[vector_set], 'or', label='Critical Points')
axs[1].set_xlim((0, 1))
axs[1].set_ylim((-1.2*Fs, 2.6*Fs))
axs[1].set_ylabel('$\\boldsymbol{f_{nl}}$ [N]')
axs[1].set_xlabel('$\\boldsymbol{t/T}$')
axs[1].set_title('Nonlinear Forces', pad=-20)
axs[1].legend(framealpha=1.0, frameon=False, loc='upper left', 
              handletextpad=-0.05, bbox_to_anchor=(-0.05, 1.05))


axs[0].tick_params(bottom=True, top=True, left=True, right=True, direction="in")
axs[1].tick_params(bottom=True, top=True, left=True, right=True, direction="in")
axs[0].xaxis.set_tick_params(labelbottom=False)

# fig.savefig('vec_iwan.eps', bbox_inches='tight')

plt.show()
