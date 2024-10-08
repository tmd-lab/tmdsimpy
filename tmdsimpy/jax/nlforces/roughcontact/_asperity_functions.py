"""
Functions for modeling individual asperities in the Rough Contact model. 
These are used to construct friction forces for the nonlinear force defined
elsewhere. 
"""

# Standard imports
import numpy as np

# JAX imports
import jax
import jax.numpy as jnp

# Decoractions for Partial compilation
from functools import partial

###############################################################################
######## Normal Contact of Asperities                                  ########
###############################################################################

##########
# PLAN FOR NOW: 
#   -Use un as delta or the interference of the individual asperity. 

# @partial(jax.jit, static_argnums=(1,2)) # want to compile larger chunks
def _normal_el_asperity_loading(un, Re, Estar):
    """
    Elastic Loading: Hertzian Contact

    Parameters
    ----------
    un : TYPE
        DESCRIPTION.
    Re : TYPE
        DESCRIPTION.
    Estar : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """ 
    
    # Positive 
    pos_un = jnp.where(un >= 0, un, 0)
    
    # Contact Area Elastic
    a = jnp.sqrt( (2*Re) * (pos_un/2) )
    a = jnp.where(un>0, a, 0) # Want to avoid NaN gradients when not in contact
    
    fn = 4*(2*Estar)*jnp.sqrt(2*Re)/3*(pos_un/2)**(1.5)
    
    deltabar = jnp.zeros_like(un) # deltabar for the reloading solution offset
    Rebar = Re*jnp.ones_like(un) # Flattened radius of curvature
    
    return fn, a, deltabar, Rebar


# @partial(jax.jit, static_argnums=(1,2,3,4,5,6,7)) # want to compile larger chunks
def _normal_pl_asperity_loading(un, Re, Possion, Estar, Emod, Etan, delta_y, Sys):
    """
    Elastic Plastic Loading:
        H. Ghaednia, M.R.W. Brake, M. Berryhill, R.L. Jackson, 2019, Strain 
        Hardening From Elastic-Perfectly Plastic to Perfectly Elastic 
        Flattening Single Asperity Contact, Journal of Tribology.

    Parameters
    ----------
    un : TYPE
        DESCRIPTION.
    Re : TYPE
        DESCRIPTION.
    Possion : TYPE
        DESCRIPTION.
    Estar : TYPE
        DESCRIPTION.
    Emod : TYPE
        DESCRIPTION.
    Etan : TYPE
        DESCRIPTION.
    delta_y : TYPE
        DESCRIPTION.
    Sys : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    C = 1.295*jnp.exp(0.736*Possion);

    pos_un = jnp.where(un >= 0, un, 0)
    
    ###### Contact Area
    
    ae = jnp.sqrt( (2*Re)*(pos_un/2) )
    
    # this branch should not be hit when not in contact, or would need this:
    # jnp.where(un>0, ae, 0) # Want to avoid NaN gradients when not in contact
    
    B = 0.14*jnp.exp(23*Sys/(2*Estar))
    
    ap = (2*Re)*(np.pi*C*Sys/(2*2*Estar))*jnp.sqrt( pos_un/delta_y * ( (pos_un/delta_y)/1.9 )**B )

    # Assumes that Tangent Modulus Material Property is a static input
    if Etan / Emod < np.finfo(float).eps:
        # Note, the use of a standard if statement is only evaluated on initial
        # run and thus deterimes the branch for all subsequent runs. Etan is
        # marked static so that this does not cause errors
        
        Upsilon = 0
        
        # The Limit of Upsilon at Et -> 0 has derivative of 0 w.r.t. un. 
        # However, autodiff will do d/dx(0^x) and break. 
        # Therefore, Upsilon is set here
        
    else:            
        # Eqn 26 - not sure about the square root in the denominator.
        Upsilon = -2*jnp.sqrt(Emod/Sys)*(Etan/Emod)**(1 - ((pos_un-delta_y)/2)/(Re*2)) \
                / ((4 - 3*jnp.exp(-2*jnp.sqrt(Emod/Sys)*Etan/Emod))*(1 - Etan/Emod))
    
    # Eqn 25
    a = ae + (ap - ae)*jnp.exp(Upsilon)
    
    ###### Force Parameters
    H_Sys = 2.84 - 0.92*(1 - jnp.cos(np.pi*a/(2*Re)) )
    
    Fc = 4/3*(2*Re/(2*Estar))**2*(C*np.pi*Sys/2)**3
    
    ###### Forces
    
    # Elastic Forces - No idea where the pi came from in the paper, but
    # it is in consistent with Hertzian contact so is not included
    # here.
    Fe = 4*(2*Estar)*jnp.sqrt(2*Re)/3*(un/2)**(1.5)
    
    # Plastic Forces
    Fp = Fc*(jnp.exp(-0.25*(pos_un/delta_y)**(5/12))*(pos_un/delta_y)**(1.5) \
             + 4*H_Sys/C*(1-jnp.exp(-1/25*(pos_un/delta_y)**(5/9)))*(pos_un/delta_y))
    
    fn = Fp + (Fe - Fp)*(1 - jnp.exp(-3.3*Etan/Emod))
    
    ###### Return Data
    
    deltabar = jnp.zeros_like(un) # deltabar for the reloading solution offset
    Rebar = Re*jnp.ones_like(un) # Flattened radius of curvature
    
    return fn, a, deltabar, Rebar


# @partial(jax.jit, static_argnums=(4, 5))  # want to compile larger chunks
def _normal_asperity_unloading(un, deltam, Fm, Re, Estar):
    """
    
    Elastic Unloading:
        M.R.W. Brake, 2015, An analytical elastic plastic contact model 
        with strain hardening and frictional effects for normal and oblique
        impacts, International Journal of Solids and Structures

    Parameters
    ----------
    un : TYPE
        DESCRIPTION.
    deltam : TYPE
        DESCRIPTION.
    Fm : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    deltabar = deltam*(1 - Fm/(4/3*Estar*jnp.sqrt(Re)*deltam**1.5))
        
    pos_un = jnp.where((un-deltabar)>=0, (un-deltabar), 0)
    pos_deltam = jnp.where((deltam-deltabar)>=0, (deltam-deltabar), 0)
    
    Rebar = Fm**2/( (4/3*Estar)**2 * (pos_deltam)**3)
    
    a = jnp.sqrt(Rebar*pos_un)
    a = jnp.where(pos_un>0, a, 0) # Want to avoid NaN gradients when not in contact
    
    fn = 4/3*Estar*jnp.sqrt(Rebar)*(pos_un)**1.5
    
    # Fm could be 0 if never comes into contact, so this corrects the normal
    # force in those cases so it is not nan
    fn = jnp.where(pos_un>0, fn, 0) 
    
    ###### Return Data
    
    return fn, a, deltabar, Rebar


# @partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))  # want to compile larger chunks
def _normal_asperity_general(un, deltam, Fm, 
                             Re, Possion, Estar, Emod, Etan, delta_y, Sys):
    """
    Call the full set of normal functions and combine to give the correct force
    value

    Parameters
    ----------
    un : Normal interference relative to nominal gap of each asperity (array)
    deltam : Maximum previous displacement of each asperity (array)
    Fm : Maximum previous force of each asperity (array)
    Re : Effective radius of the asperities (scalar). This is half of the real 
         radius
    Possion : Poisson's ratio (scalar)
    Estar : Combined Elastic Modulus for contact (scalar)
    Emod : Elastic Modulus (scalar)
    Etan : Tangent Modulus of plasticity (scalar)
    delta_y : Yield interference between two asperities
    Sys : Yield Stress of contacts

    Returns
    -------
    fn : Normal force (array) between asperities
    a : Contact radii of asperities (array)
    deltabar : Permanent Deformation displacement of asperities (array)
    Rebar : Permanent Deformation radius change of asperities (array)

    """
    # Elastic Loading
    fn_el, a_el, deltabar_el, Rebar_el = _normal_el_asperity_loading(un, Re, 
                                                                     Estar)
    
    # Plastic Loading
    fn_pl, a_pl, deltabar_pl, Rebar_pl = _normal_pl_asperity_loading(un, Re, 
                                             Possion, Estar, Emod, Etan, 
                                             delta_y, Sys)
    
    # Elastic Unloading after Plasticity
    fn_pu, a_pu, deltabar_pu, Rebar_pu = _normal_asperity_unloading(un, deltam, 
                                                            Fm, Re, Estar)
    
    # Split Elastic v. Elastic Unloading After Plasticity
    elastic_flag = jnp.logical_and(un<1.9*delta_y, deltam<1.9*delta_y)
    
    fn       = jnp.where(elastic_flag, fn_el,       fn_pu)
    a        = jnp.where(elastic_flag, a_el,        a_pu)
    deltabar = jnp.where(elastic_flag, deltabar_el, deltabar_pu)
    Rebar    = jnp.where(elastic_flag, Rebar_el,    Rebar_pu)
    
    # Add in Plastic Loading where appropriate
    # Strictly greater because want the unloading gradient if recall at the 
    # same displacement - this gives a more consistent linear eigenanalysis
    # about the prestressed state.
    yielding_flag = jnp.logical_and(un>deltam, un>1.9*delta_y)
    
    fn       = jnp.where(yielding_flag, fn_pl,       fn)
    a        = jnp.where(yielding_flag, a_pl,        a)
    deltabar = jnp.where(yielding_flag, deltabar_pl, deltabar)
    Rebar    = jnp.where(yielding_flag, Rebar_pl,    Rebar)
    
    return fn, a, deltabar, Rebar

###############################################################################
######## Tangential Contact of Asperities                              ########
###############################################################################

def _tangential_asperity(uxy, uxy0, fxy0, fn, a, Gstar, mu):
    """
    Calculate the tangential forces

    Parameters
    ----------
    uxy : Tangential displacements shared by all asperities, size: (2,)
    uxy0 : Previous tangential displacement shared by all asperities, size(2,)
    fxy0 : previous tangential forces at each asperity in columns of x and y (Nasp, 2)
    fn : Normal forces for each asperity, (Nasp,)
    a : Contact radii for each asperity, (Nasp,)
    Gstar : Material property in Mindlin contact
    mu : Friction coefficient

    Returns
    -------
    fxy : Current frictional forces in tangential directions in columns of x and y (Nasp, 2)

    """
    # Tangential Stiffness
    kt = 8*Gstar*a
    
    # Stuck calculation
    fxy = jnp.outer(kt,(uxy - uxy0)) + fxy0
    
    fs = jnp.outer(mu*fn,np.ones(2))
    
    # Positive slip
    fxy = jnp.minimum(fxy, fs)
    
    # Negative slip
    fxy = jnp.maximum(fxy, -fs)
    
    return fxy
    

def _interp_hist_loop_body(ind, txy0_interp, 
                           quad_radii_curr, quad_radii0, txy0):
    """
    Wrapper function to allow for interpolation of traction history 
    used in a for loop.
    
    Does interpolation at a single index
    """
    
    txy0_interp = txy0_interp.at[ind, :, 0].set(
                jnp.interp(quad_radii_curr[ind, :],
                           quad_radii0[ind, :],
                           txy0[ind, :, 0],
                           left=0.0,
                           right=0.0) # right = new contact area
                )
    
    txy0_interp = txy0_interp.at[ind, :, 1].set(
                jnp.interp(quad_radii_curr[ind, :],
                           quad_radii0[ind, :],
                           txy0[ind, :, 1],
                           left=0.0,
                           right=0.0) # right = new contact area
                )
    
    return txy0_interp
    
def _tangential_asperity_mif(uxy, uxy0, txy0, fn, a, Gstar, mu, 
                             quad_radii0, quad_radii_norm, weight_radii):
    """
    Calculates tangential asperity forces for the Mindlin-Iwan Fit (MIF)
    model.

    Parameters
    ----------
    uxy : TYPE
        DESCRIPTION.
    uxy0 : TYPE
        DESCRIPTION.
    txy0 : TYPE
        DESCRIPTION.
    fn : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    Gstar : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    quad_radii0 : (Nasp,Nrad) numpy.ndarray
        Initial quadrature radii for each point in the history tractions
        Could replace this with `np.outer(a0, quad_radii_norm)`
    quad_radii_norm : (Nrad,) numpy.ndarray
        Normalized quadrature radii to be used in integrals. All radii in [0,1]
    weight_radii : (Nrad), numpy.ndarray
        Quadrature weights for integral over radial displacements. 
        These are just weights for a linear integral, this function includes
        terms to handle the fact that it is an integral in radial coordinates.

    Returns
    -------
    asp_fxy : (Nasp, 2) numpy.ndarray
        Integrated forces for each of the asperities in x,y directions
    txy : (Nasp, Nrad, 2) numpy.ndarray
        Tractions at each asperity (dim 0), radial position (dim 1), and for x
        or y direction (dim 2).
    quad_radii_curr : (Nasp, Nrad) numpy.ndarray
        Radial positions in real lengths for each asperity traction 
        calculation. With dim 0 being the asperity and dim 1 being the radial 
        location

    """
    
    # Interpolate history to new physical radii
    quad_radii_curr = jnp.outer(a, quad_radii_norm)
    
    txy0_interp = jnp.zeros_like(txy0)
    
    # Having jax expand this for loop is worse for compile and evaluation time
    loop_fun = lambda i, txy0_interp : _interp_hist_loop_body(i, txy0_interp, 
                                quad_radii_curr, quad_radii0, txy0)
    
    txy0_interp = jax.lax.fori_loop(0, a.shape[0], loop_fun, txy0_interp)
    
    # Tangential Stiffness
    Kt_tot = 8*Gstar*a
    
    # Equation 42 of Porter and Brake (2023) 
    kt_tilde = 0.8709 + (1 - 0.8709) / jnp.sqrt(1-quad_radii_norm**2) \
                + 0.0629 * quad_radii_norm \
                - 0.8915 * quad_radii_norm**2 \
                + 0.6998 * quad_radii_norm**3
                
    # mask out nan at normalized radius of 1 
    # (zero normal pressure there, so this is fine)
    kt_tilde = jnp.where(jnp.isfinite(kt_tilde), kt_tilde, 0.0)
    
    # Equation 43 of Porter and Brake (2023) 
    kt_local = jnp.outer(Kt_tot/a**2/np.pi, kt_tilde)
    
    kt_local = jnp.where(fn.reshape(-1,1)>0, kt_local, 0)
    
    # Normal Pressure Distribution, Equation 35 of Porter and Brake (2023)
    tn = jnp.outer(3*fn/(2*np.pi*a**2), jnp.sqrt(1 - quad_radii_norm**2))
    
    # Mask out nan's where it was not in contact
    tn = jnp.where(fn.reshape(-1,1)>0, tn, 0)

    #### NEED TO MASK OUT NANS    
    
    # Stuck Force Prediction, Equation 37 of Porter and Brake (2023)
    txy = jnp.atleast_3d(kt_local)*(uxy.reshape(1,1,-1)-uxy0.reshape(1,1,-1))\
            + txy0_interp
    
    # Positive slip limit
    txy = jnp.minimum(txy, jnp.atleast_3d(mu*tn))
    
    # Negative slip limit
    txy = jnp.maximum(txy, jnp.atleast_3d(-mu*tn))
    
    # Integrate over radii on each asperity, 
    # Equation 38 of Porter and Brake (2023)
    asp_fxy = 2*np.pi*(a.reshape(-1,1))**2 \
                * jnp.einsum('ijk,j', 
                             txy * jnp.atleast_3d(quad_radii_norm), 
                             weight_radii)
    
    return asp_fxy, txy, quad_radii_curr