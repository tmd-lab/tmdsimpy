# Configure JAX to use 64 bit before JAX is used by the package.
import jax
jax.config.update("jax_enable_x64", True)

# Explicit modifications to '__all__'

# things imported here that should be in __all__
add_to_all = []

# files that have imported contents here, so should not be in __all__
remove_from_all = []

# Generate a list of submodules
import os
from pathlib import Path

search_path = os.path.dirname(os.path.abspath(__file__))

__all__ = [Path(f).stem for f in os.listdir(search_path)]

# remove anything that starts with '_'
__all__ = [f for f in __all__ if f[0] != '_']

__all__ += add_to_all

__all__ = [f for f in __all__ if not f in remove_from_all]
