"""
Nonlinear forces implemented utilizing the JAX library and automatic 
differentiation.

See Also
--------
tmdsimpy.nlforces :
    Nonlinear forces implemented without JAX.
"""

# Import nonlinear forces written with JAX
from .elastic_dry_fric_2d import ElasticDryFriction2D
from .jenkins_element import JenkinsForce
from .roughcontact.rough_contact import RoughContactFriction
from .vector_jenkins import VectorJenkins

# Explicit modifications to '__all__'

# things imported here that should be in __all__
add_to_all = ['ElasticDryFriction2D',
              'JenkinsForce',
              'RoughContactFriction',
              'VectorJenkins']

# files that have imported contents here, so should not be in __all__
remove_from_all = ['elastic_dry_fric_2d',
                   'jenkins_element',
                   'roughcontact',
                   'vector_jenkins']

# Generate a list of submodules
import os
from pathlib import Path

search_path = os.path.dirname(os.path.abspath(__file__))

__all__ = [Path(f).stem for f in os.listdir(search_path)]

# remove anything that starts with '_'
__all__ = [f for f in __all__ if f[0] != '_']

__all__ += add_to_all

__all__ = [f for f in __all__ if not f in remove_from_all]
