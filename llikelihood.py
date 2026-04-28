import StreaMAX

import jax
import jax.numpy as jnp

import numpy as np

from utils import params_to_stream, params_to_stream_scale_free

BAD_VAL = -1e15


def _legacy_track_loglike(params, dict_data, n_particles=20000, n_min=3, var_ratio_thresh=9.0, mass_loss_mode='linear_to_zero'):
    theta_stream, r_stream, xv_stream = params_to_stream(params, n_particles, mass_loss_mode=mass_loss_mode)
    r_bin, sig_bin, count_bin = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])

    n_bad = jnp.sum(count_bin < n_min)
    if n_bad > 0:
        logl = BAD_VAL * n_bad
    else:
        var_data = dict_data['r_err']**2
        var_model = sig_bin**2 / count_bin
        n_high_error = jnp.sum(var_model / var_data > 1 / var_ratio_thresh)
        if n_high_error > 0:
            logl = BAD_VAL/1e3 * n_high_error
        else:
            var = var_data + params[13]**2
            logl = -.5 * jnp.sum((r_bin - dict_data['r'])**2 / var + jnp.log(2 * jnp.pi * var))

    return logl, theta_stream, xv_stream


def logl_track(params, dict_data, n_particles=20000, n_min=3, var_ratio_thresh=9.0, var_ratio_thresh_v=9.0, mass_loss_mode='linear_to_zero'):
    logl, _, _ = _legacy_track_loglike(params, dict_data, n_particles, n_min, var_ratio_thresh, mass_loss_mode)
    return logl

def logl(params, dict_data, n_particles=20000, n_min=3, var_ratio_thresh=9.0, var_ratio_thresh_v=9.0, mass_loss_mode='linear_to_zero'):
    logl, theta_stream, xv_stream = _legacy_track_loglike(
        params, dict_data, n_particles, n_min, var_ratio_thresh, mass_loss_mode
    )
    if logl <= BAD_VAL / 1e4:
        return logl

    theta_center = dict_data['vz_theta']
    theta_half = dict_data['vz_window']/2
    theta_vmin = theta_center - theta_half
    theta_vmax = theta_center + theta_half
    mask_v = (theta_stream >= theta_vmin) & (theta_stream <= theta_vmax)
    count_v = jnp.sum(mask_v)
    vz_sum = jnp.sum(xv_stream[:, 5] * mask_v)
    vz_mean = vz_sum / count_v
    logl += -.5 * ((dict_data['vz'] - vz_mean)**2 / dict_data['vz_err']**2)
    return logl

def logl_v(params, dict_data, n_particles=20000, n_min=3, var_ratio_thresh=9.0, var_ratio_thresh_v=9.0, mass_loss_mode='linear_to_zero'):
    theta_stream, r_stream, xv_stream = params_to_stream(params, n_particles, mass_loss_mode=mass_loss_mode)
    r_bin, sig_bin, count_bin = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])

    n_bad = jnp.sum(count_bin < n_min)

    theta_center = dict_data['vz_theta']
    theta_half = dict_data['vz_window']/2
    theta_vmin = theta_center - theta_half
    theta_vmax = theta_center + theta_half
    mask_v = (theta_stream >= theta_vmin) & (theta_stream <= theta_vmax)
    count_v = jnp.sum(mask_v)
    n_bad_v = jnp.sum(mask_v) < n_min
    
    n = n_bad + n_bad_v
    if n > 0:
        logl = BAD_VAL * n
    
    else:
        var_data  = dict_data['r_err']**2
        var_model = sig_bin**2 / count_bin
        n_high_error = jnp.sum( var_model / var_data > 1/var_ratio_thresh )

        vz_sum = jnp.sum(xv_stream[:, 5] * mask_v)
        vz_mean = vz_sum / count_v
        vz_res = (xv_stream[:, 5] - vz_mean) * mask_v
        vz_model_var = jnp.sum(vz_res**2) / (count_v**2)

        vz_data_var = dict_data['vz_err']**2
        n_high_error_v = vz_model_var / vz_data_var > 1/var_ratio_thresh_v

        n_high = n_high_error + n_high_error_v
        if n_high > 0:
            logl = BAD_VAL/1e3 * n_high
        
        else:
            var = var_data + params[13]**2 #params[10]**2
            logl  = -.5 * jnp.sum( (r_bin - dict_data['r'])**2 / var  + jnp.log(2 * jnp.pi * var))

            vz_var = vz_data_var # + params[14]**2 # + params[11]**2
            logl += -.5 * ((dict_data['vz'] - vz_mean)**2 / vz_var + jnp.log(2 * jnp.pi * vz_var))
            
    return logl


def _scale_free_track_loglike(params, dict_data, n_particles=20000, n_min=3, var_ratio_thresh=9.0, mass_loss_mode='linear_to_zero'):
    theta_stream, r_stream, xv_stream = params_to_stream_scale_free(params, n_particles, mass_loss_mode=mass_loss_mode)
    r_bin, sig_bin, count_bin = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])

    n_bad = jnp.sum(count_bin < n_min)
    if n_bad > 0:
        logl = BAD_VAL * n_bad
    else:
        var_data = dict_data['r_err']**2
        var_model = sig_bin**2 / count_bin
        n_high_error = jnp.sum(var_model / var_data > 1 / var_ratio_thresh)
        if n_high_error > 0:
            logl = BAD_VAL/1e3 * n_high_error
        else:
            var = var_data + params[13]**2
            logl = -.5 * jnp.sum((r_bin - dict_data['r'])**2 / var + jnp.log(2 * jnp.pi * var))

    return logl, theta_stream, xv_stream


def logl_scale_free_track(params, dict_data, n_particles=20000, n_min=3, var_ratio_thresh=9.0, var_ratio_thresh_v=9.0, mass_loss_mode='linear_to_zero'):
    logl, _, _ = _scale_free_track_loglike(params, dict_data, n_particles, n_min, var_ratio_thresh, mass_loss_mode)
    return jnp.nan_to_num(logl, nan=BAD_VAL, neginf=BAD_VAL, posinf=BAD_VAL)


def logl_scale_free_track_los(params, dict_data, n_particles=20000, n_min=3, var_ratio_thresh=9.0, var_ratio_thresh_v=9.0, mass_loss_mode='linear_to_zero'):
    logl, theta_stream, xv_stream = _scale_free_track_loglike(
        params, dict_data, n_particles, n_min, var_ratio_thresh, mass_loss_mode
    )
    if logl <= BAD_VAL / 1e4:
        return logl

    theta_center = dict_data['vz_theta']
    theta_half = dict_data['vz_window']/2
    theta_vmin = theta_center - theta_half
    theta_vmax = theta_center + theta_half
    mask_v = (theta_stream >= theta_vmin) & (theta_stream <= theta_vmax)
    count_v = jnp.sum(mask_v)
    n_bad_v = bool(jax.device_get(count_v < n_min))
    if n_bad_v:
        return BAD_VAL

    vz_sum = jnp.sum(xv_stream[:, 5] * mask_v)
    vz_mean = vz_sum / count_v
    vz_var = dict_data['vz_err']**2
    logl += -.5 * ((dict_data['vz'] - vz_mean)**2 / vz_var + jnp.log(2 * jnp.pi * vz_var))
    return jnp.nan_to_num(logl, nan=BAD_VAL, neginf=BAD_VAL, posinf=BAD_VAL)
