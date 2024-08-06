"""
Nonlinear force implementations

See Also
--------
tmdsimpy.jax.nlforces :
    Nonlinear forces implemented with JAX for automatic differentiation.
"""
from .nonlinear_force import NonlinearForce, InstantaneousForce, HystereticForce

from .general_poly_stiffness import GenPolyForce
from .iwan4_element import Iwan4Force
from .unilateral_spring import UnilateralSpring
from .cubic_stiffness import CubicForce
from .iwan_bb_conserve import ConservativeIwanBB
from .cubic_damping import CubicDamping
from .quintic_stiffness import QuinticForce
from .vector_jenkins import VectorJenkins
from .jenkins_element import JenkinsForce
from .vector_iwan4 import VectorIwan4

# Explicit modifications to '__all__'

# things imported here that should be in __all__
add_to_all = ['NonlinearForce', 'InstantaneousForce', 'HystereticForce',
              'ConservativeIwanBB',
              'CubicDamping',
              'CubicForce',
              'GenPolyForce',
              'Iwan4Force',
              'JenkinsForce',
              'QuinticForce',
              'UnilateralSpring',
              'VectorIwan4',
              'VectorJenkins'
              ]

# files that have imported contents here, so should not be in __all__
remove_from_all = ['nonlinear_force',
                   'general_poly_stiffness',
                   'iwan4_element',
                   'unilateral_spring',
                   'cubic_stiffness',
                   'iwan_bb_conserve',
                   'cubic_damping',
                   'quintic_stiffness',
                   'vector_jenkins',
                   'jenkins_element',
                   'vector_iwan4'
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
