import StreaMAX

import jax
import jax.numpy as jnp

import numpy as np

from utils import params_to_stream

BAD_VAL = -1e15

def logl(params, dict_data, n_particles=20000, n_min=3, var_ratio_thresh=9.0, min_err=0.):
    theta_stream, r_stream, xv_stream = params_to_stream(params, n_particles)
    r_bin, sig_bin, count_bin = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])

    n_bad = jnp.sum(count_bin < n_min)
    if n_bad > 0:
        logl = BAD_VAL * n_bad

    else:
        var_data  = jnp.clip(dict_data['r_err'], a_min=min_err, a_max=None)**2
        var_model = sig_bin**2 / count_bin
        n_high_error = jnp.sum( var_model / var_data > 1/var_ratio_thresh )
        if n_high_error > 0:
            logl = BAD_VAL/1e3 * n_high_error

        else:
            var = var_data + params[13]**2
            logl  = -.5 * jnp.sum( (r_bin - dict_data['r'])**2 / var  + jnp.log(2 * jnp.pi * var))

            if 'vz' in dict_data:
                mask     = jnp.abs(theta_stream) < dict_data['bin_width'] / 2
                count    = jnp.maximum(jnp.sum(mask), 1)
                vz_model = jnp.sum(xv_stream[:, 5] * mask) / count
                vz_var   = dict_data['vz_err']**2
                logl    += -.5 * ((dict_data['vz'] - vz_model)**2 / vz_var + jnp.log(2 * jnp.pi * vz_var))

    return logl
