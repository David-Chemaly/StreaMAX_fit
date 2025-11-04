import StreaMAX

import jax
import jax.numpy as jnp

from utils import params_to_stream

BAD_VAL = -1e10

def logl(params, dict_data, n_particles=10000, n_min=101):
    theta_stream, r_stream, _ = params_to_stream(params, n_particles)
    r_bin, _, count_bin = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])

    n_bad = jnp.sum(jnp.nan_to_num(count_bin, nan=0.) < n_min)
    if jnp.all(jnp.isnan(r_bin)):
        logl = BAD_VAL * len(r_bin)
    elif n_bad == 0:
        var = dict_data['r_err']**2 + params[13]**2
        logl  = -.5 * jnp.sum(  (r_bin - dict_data['r'])**2 / var  + jnp.log(2 * jnp.pi * var)  )
    else:
        logl = BAD_VAL * n_bad

    return logl