"""
A Python module for tribomechadynamics (TMD) simulations.

Notes
-----

This code was developed for [1]_, [2]_, [3]_.

References
----------

.. [1]
   Porter, J. H., and M. R. W. Brake. 2024. "Tracking Superharmonic
   Resonances for Nonlinear Vibration of Conservative and Hysteretic Single 
   Degree of Freedom Systems." Mechanical Systems and Signal Processing 
   215:111410. https://doi.org/10.1016/j.ymssp.2024.111410.
   arXiv:2401.08790

.. [2]
   Porter, J. H., and M. R. W. Brake. Under Review. "Efficient Model 
   Reduction and Prediction of Superharmonic Resonances in Frictional and 
   Hysteretic Systems." Mechanical Systems and Signal Processing.
   arXiv:2405.15918.

.. [3]
   Porter, J. H. 2024. Modal Interactions and Jointed Structures. PhD Thesis.
   Rice University.
"""


from .continuation import Continuation
from .vibration_system import VibrationSystem
from .solvers import NonlinearSolver

# Explicit modifications to '__all__'

# things imported here that should be in __all__
add_to_all = ['Continuation', 'NonlinearSolver', 'VibrationSystem']

# files that have imported contents here, so should not be in __all__
remove_from_all = ['continuation',
                   'vibration_system',
                   'solvers',
                   'harmonic_utils', # depricated file
                   'continuation_utils' # depricated file
                   ]


# Generate a list of submodules
import os
from pathlib import Path

search_path = os.path.dirname(os.path.abspath(__file__))

__all__ = [Path(f).stem for f in os.listdir(search_path)]

# remove anything that starts with '_'
__all__ = [f for f in __all__ if f[0] != '_']

__all__ += add_to_all

__all__ = [f for f in __all__ if not f in remove_from_all]
