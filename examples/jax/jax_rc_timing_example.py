"""
Timing information for rough contact modeling, specifically the 
Mindlin-Iwan Fit Model (MIF) takes about 100x longer when using 100 radii 
to add microslip at the asperity level.
"""

# Standard imports
import numpy as np
import sys
import timeit

# Imports of Custom Functions and Classes
sys.path.append('../..')
import tmdsimpy.harmonic_utils as hutils

# JAX version w/o vectorization
from tmdsimpy.jax.nlforces.roughcontact.rough_contact import RoughContactFriction


###############################################################################
######## Build Models with somewhat arbitrary parameters               ########
###############################################################################

ElasticMod = 192850000000.000000
PoissonRatio = 0.290000
Radius = 4.174432305996625e-03
TangentMod = 1928500000.000000
YieldStress = np.inf
mu = 2.000000e-01
asp_num_quad_points = 100


# Construct New Rough Contact Models to Match the old ones
Q = np.eye(3)
T = np.eye(3)

gaps = np.linspace(0, 1e-6, 100)
gap_weights = np.ones_like(gaps) / gaps.shape[0]

tan_model = RoughContactFriction(Q, T, ElasticMod, PoissonRatio, Radius, 
                                 TangentMod, YieldStress, mu,
                                 gaps=gaps, gap_weights=gap_weights,
                                 tangent_model='TAN')

mif_model = RoughContactFriction(Q, T, ElasticMod, PoissonRatio, Radius, 
                                 TangentMod, YieldStress, mu,
                                 gaps=gaps, gap_weights=gap_weights,
                                 tangent_model='MIF', N_radial_quad=100)

###############################################################################
######## Define Exact Case to Time                                     ########
###############################################################################

h = np.arange(4)
Nt = 1<<7

Ndof = 3
Nhc = hutils.Nhc(h)

U = np.zeros(Nhc*Ndof)
U[:Ndof] = np.array([0.0, 0.0, 3e-6]) # Static
U[Ndof:2*Ndof] = np.array([5e-6, -1e-5, 1e-6]) # First Harmonic Cosine
U[5*Ndof:6*Ndof] = np.array([-2e-6, 1e-6, 1e-6])

w = 1.75

###############################################################################
######## Compile Time                                                  ########
###############################################################################


tan_compile = timeit.timeit(lambda : tan_model.aft(U, w, h, Nt=Nt), number=1)
mif_compile = timeit.timeit(lambda : mif_model.aft(U, w, h, Nt=Nt), number=1)


print('Compile Times:')
print('TAN: {} sec, MIF: {} sec'.format(tan_compile, mif_compile))


###############################################################################
######## Repeat Evaluation Time                                        ########
###############################################################################

num_times = 20

tan_repeat = timeit.timeit(lambda : tan_model.aft(U, w, h, Nt=Nt), number=num_times)
mif_repeat = timeit.timeit(lambda : mif_model.aft(U, w, h, Nt=Nt), number=num_times)

print('Evaluation Times (averaged over {}):'.format(num_times))
print('TAN: {} sec, MIF: {} sec'.format(tan_repeat/num_times, 
                                        mif_repeat/num_times))


