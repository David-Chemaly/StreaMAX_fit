import StreaMAX

import jax
import jax.numpy as jnp

from utils import params_to_stream

BAD_VAL = -1e15

def logl(params, dict_data, n_particles=20000, n_min=3, var_ratio_thresh=9.0, triaxial=False):
    theta_stream, r_stream, xv_stream = params_to_stream(params, n_particles, triaxial=triaxial)
    r_bin, sig_bin, count_bin = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(
        theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])

    sig_idx = 15 if triaxial else 13
    n_bad = jnp.sum(count_bin < n_min)
    if n_bad > 0:
        logl = BAD_VAL * n_bad

    else:
        var_data  = dict_data['r_err']**2
        var_model = sig_bin**2 / count_bin
        n_high_error = jnp.sum(var_model / var_data > 1/var_ratio_thresh)
        if n_high_error > 0:
            logl = BAD_VAL/1e3 * n_high_error

        else:
            var = var_data + params[sig_idx]**2
            logl = -.5 * jnp.sum((r_bin - dict_data['r'])**2 / var + jnp.log(2 * jnp.pi * var))

    return logl
