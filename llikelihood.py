import StreaMAX

import jax
import jax.numpy as jnp

import numpy as np

from utils import params_to_stream

BAD_VAL = -1e15
VAR_EPS = 1e-8

def logl(params, dict_data, n_particles=20000, n_min=3, var_ratio_pos=9.0, min_err=0., use_kinematics=False, use_flattening=True, var_ratio_vel=None):
    theta_stream, r_stream, xv_stream = params_to_stream(params, n_particles, use_flattening=use_flattening)
    r_bin, sig_bin, count_bin = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])
    if var_ratio_vel is None:
        var_ratio_vel = var_ratio_pos

    n_bad = jnp.sum(count_bin < n_min)
    if n_bad > 0:
        logl = BAD_VAL * n_bad
    
    else:
        err_floor = jnp.maximum(min_err, VAR_EPS)
        var_data  = jnp.clip(dict_data['r_err'], a_min=err_floor, a_max=None)**2
        var_model = sig_bin**2 / count_bin
        n_high_error = jnp.sum( var_model / var_data > 1/var_ratio_pos )
        if n_high_error > 0:
            logl = BAD_VAL/1e3 * n_high_error
        
        else:
            sig_r = params[-2] if use_kinematics else params[-1]
            var = jnp.maximum(var_data + sig_r**2, VAR_EPS)
            logl  = -.5 * jnp.sum( (r_bin - dict_data['r'])**2 / var  + jnp.log(2 * jnp.pi * var))

            # Optional kinematic term on line-of-sight velocity around a reference angle.
            if use_kinematics and ('vz' in dict_data) and ('vz_err' in dict_data):
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
                    vz_data_var = jnp.clip(dict_data['vz_err'], a_min=err_floor, a_max=None)**2
                    if vz_model_var / vz_data_var > 1/var_ratio_vel:
                        logl += BAD_VAL/1e3
                    else:
                        sig_v = params[-1]
                        vz_var = jnp.maximum(vz_data_var + sig_v**2, VAR_EPS)
                        logl += -.5 * ((dict_data['vz'] - vz_model)**2 / vz_var + jnp.log(2 * jnp.pi * vz_var))

    logl = jnp.where(jnp.isfinite(logl), logl, BAD_VAL)
    return logl
