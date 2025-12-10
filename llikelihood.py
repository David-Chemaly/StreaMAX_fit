import StreaMAX

import jax
import jax.numpy as jnp

import numpy as np

from utils import params_to_stream

BAD_VAL = -1e15

def logl(params, dict_data, n_particles=10000, n_min=101, var_ratio_thresh=9.0, key=jax.random.PRNGKey(111)):
    theta_stream, r_stream, _ = params_to_stream(params, n_particles)

    arg_take = jax.random.permutation(key, theta_stream.shape[0])

    nn = int(params[14])
    theta_stream = theta_stream[arg_take][:nn]
    r_stream     = r_stream[arg_take][:nn]

    r_bin, sig_bin, count_bin = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])


    if jnp.min(count_bin) < 3:
        logl = BAD_VAL
    
    else:
        alpha = 0.01
        N_min_diff = jnp.min( (count_bin + jnp.sqrt(count_bin)) / jnp.max(count_bin))
        if N_min_diff < alpha:
            logl = BAD_VAL/1e3 * ( 1 +  (1 - N_min_diff/alpha))

        else:
            var_data  = dict_data['r_err']**2
            var_model = sig_bin**2 / count_bin
            worst_ratio = jnp.max( var_model / var_data )

            penalty = (jnp.log(worst_ratio) + jnp.log(var_ratio_thresh))**2
            if worst_ratio > 1/var_ratio_thresh:
                logl = BAD_VAL/1e6 * ( 1 +  penalty)

            else:
                var = var_data + params[13]**2
                logl  = -.5 * jnp.sum( (r_bin - dict_data['r'])**2 / var  + \
                                        jnp.log(2 * jnp.pi * var)) + 1e3 * penalty
    return logl