import StreaMAX

import jax
import jax.numpy as jnp

from utils import params_to_stream

BAD_VAL = -1e18

def logl(params, dict_data, n_particles=20000, n_min=3, var_ratio_thresh=9.0, triaxial=False, max_aspect=1.5):
    # --- Penalty hierarchy (harshest first, cheapest first) ---

    # 1. Shape constraints for triaxial (harshest: 10x BAD_VAL)
    #    Enforces p >= q and max aspect ratio, before any expensive computation
    if triaxial:
        p_val, q_val = float(params[2]), float(params[3])
        if q_val > p_val:
            return BAD_VAL
        aspect = max(1.0, p_val) / min(1.0, q_val)
        if aspect > max_aspect:
            return BAD_VAL

    # 2. Generate stream and bin (expensive)
    theta_stream, r_stream, xv_stream = params_to_stream(params, n_particles, triaxial=triaxial)
    r_bin, sig_bin, count_bin = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(
        theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])

    sig_idx = 16 if triaxial else 13

    # 3. Empty bins (severe: BAD_VAL per bad bin)
    n_bad = jnp.sum(count_bin < n_min)
    if n_bad > 0:
        return BAD_VAL/1e3 * n_bad

    # 4. Model variance too high relative to data (moderate: BAD_VAL/1e3)
    var_data  = dict_data['r_err']**2
    var_model = sig_bin**2 / count_bin
    n_high_error = jnp.sum(var_model / var_data > 1/var_ratio_thresh)
    if n_high_error > 0:
        return BAD_VAL/1e6 * n_high_error

    # 5. Normal log-likelihood
    var = var_data + params[sig_idx]**2
    return -.5 * jnp.sum((r_bin - dict_data['r'])**2 / var + jnp.log(2 * jnp.pi * var))
