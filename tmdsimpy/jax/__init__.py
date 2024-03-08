# Configure JAX to use 64 bit before JAX is used by the package.
import jax
jax.config.update("jax_enable_x64", True)
