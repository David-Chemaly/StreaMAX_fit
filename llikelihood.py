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

    # Protect against division by zero
    safe_count = jnp.maximum(count_bin, 1)
    n_bad = jnp.sum(count_bin < n_min)

    # Compute all branches (JIT-compatible: no Python if/else)
    var_data = dict_data['r_err']**2
    var_model = sig_bin**2 / safe_count
    n_high_error = jnp.sum(var_model / var_data > 1.0 / var_ratio_thresh)

    var = var_data + params[sig_idx]**2
    normal_logl = -0.5 * jnp.sum(
        (r_bin - dict_data['r'])**2 / var + jnp.log(2 * jnp.pi * var))

    return jnp.where(
        n_bad > 0,
        BAD_VAL * n_bad,
        jnp.where(n_high_error > 0, BAD_VAL / 1e3 * n_high_error, normal_logl))
