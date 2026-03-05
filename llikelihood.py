import StreaMAX

import jax
import jax.numpy as jnp

from utils import params_to_stream

BAD_VAL = -1e15
VAR_EPS = 1e-8


def _position_logl(params, dict_data, n_particles, n_min, var_ratio_pos, min_err, use_flattening):
    theta_stream, r_stream, xv_stream = params_to_stream(params, n_particles, use_flattening=use_flattening)
    r_bin, sig_bin, count_bin = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(
        theta_stream, r_stream, dict_data['theta'], dict_data['bin_width']
    )

    n_bad = jnp.sum(count_bin < n_min)
    if n_bad > 0:
        return BAD_VAL * n_bad, theta_stream, xv_stream

    err_floor = jnp.maximum(min_err, VAR_EPS)
    var_data = jnp.clip(dict_data['r_err'], a_min=err_floor, a_max=None) ** 2
    var_model = sig_bin ** 2 / count_bin
    n_high_error = jnp.sum(var_model / var_data > 1 / var_ratio_pos)
    if n_high_error > 0:
        return BAD_VAL / 1e3 * n_high_error, theta_stream, xv_stream

    sig_r = params[-2] if params.shape[0] in (12, 15) else params[-1]
    var = jnp.maximum(var_data + sig_r ** 2, VAR_EPS)
    logl = -0.5 * jnp.sum((r_bin - dict_data['r']) ** 2 / var + jnp.log(2 * jnp.pi * var))
    return logl, theta_stream, xv_stream


def logl_pos(params, dict_data, n_particles=20000, n_min=3, var_ratio_pos=9.0, min_err=0.0, use_flattening=True):
    logl, _, _ = _position_logl(params, dict_data, n_particles, n_min, var_ratio_pos, min_err, use_flattening)
    return jnp.where(jnp.isfinite(logl), logl, BAD_VAL)


def logl_pos_vel(
    params,
    dict_data,
    n_particles=20000,
    n_min=3,
    var_ratio_pos=9.0,
    min_err=0.0,
    use_flattening=True,
    var_ratio_vel=None,
):
    if var_ratio_vel is None:
        var_ratio_vel = var_ratio_pos

    logl, theta_stream, xv_stream = _position_logl(
        params, dict_data, n_particles, n_min, var_ratio_pos, min_err, use_flattening
    )

    if ('vz' not in dict_data) or ('vz_err' not in dict_data):
        return jnp.where(jnp.isfinite(logl), logl, BAD_VAL)

    if logl <= BAD_VAL / 10:
        return jnp.where(jnp.isfinite(logl), logl, BAD_VAL)

    theta_ref = jnp.asarray(dict_data['vz_theta']) if 'vz_theta' in dict_data else jnp.array(0.0)
    half_width = jnp.asarray(dict_data['vz_window']) if 'vz_window' in dict_data else (dict_data['bin_width'] / 2.0)
    mask = jnp.abs(theta_stream - theta_ref) < half_width
    count_vz = jnp.sum(mask)
    if count_vz < n_min:
        logl += BAD_VAL
    else:
        vz_sel = xv_stream[:, 5][mask]
        vz_model = jnp.mean(vz_sel)
        vz_model_var = jnp.var(vz_sel) / count_vz
        err_floor = jnp.maximum(min_err, VAR_EPS)
        vz_data_var = jnp.clip(dict_data['vz_err'], a_min=err_floor, a_max=None) ** 2
        if vz_model_var / vz_data_var > 1 / var_ratio_vel:
            logl += BAD_VAL / 1e3
        else:
            sig_v = params[-1]
            vz_var = jnp.maximum(vz_data_var + sig_v ** 2, VAR_EPS)
            logl += -0.5 * (
                (dict_data['vz'] - vz_model) ** 2 / vz_var + jnp.log(2 * jnp.pi * vz_var)
            )

    return jnp.where(jnp.isfinite(logl), logl, BAD_VAL)


# Backward compatibility alias for existing imports/calls.
logl = logl_pos
