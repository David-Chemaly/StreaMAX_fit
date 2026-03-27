import jax
import jax.numpy as jnp
import StreaMAX

@jax.jit
def get_q(dirx, diry, dirz, q_min=0.5, q_max=1.5):
    """
    Computes an axis ratio from the direction vector magnitude.
    Maps |dir| -> q in [q_min, q_max], monotonically.
    """
    r = jnp.sqrt(dirx**2 + diry**2 + dirz**2)
    q = jnp.exp(-r**2/2) * (jnp.sqrt(jnp.pi) * jnp.exp(r**2/2) * jax.scipy.special.erf(r/jnp.sqrt(2)) - jnp.sqrt(2)*r) / jnp.sqrt(jnp.pi)
    q = (q_max - q_min) * q + q_min
    return q

def params_to_stream(params, n_particles=10000, n_steps=99, alpha=1., unroll=False, triaxial=False):
    if triaxial:
        logM, Rs, p, q, dirx, diry, dirz, logm, rs, x0, z0, vx0, vy0, vz0, time = params[:15]
        a, b, c = 1.0, p, q
    else:
        logM, Rs, dirx, diry, dirz, logm, rs, x0, z0, vx0, vy0, vz0, time = params[:13]
        q = get_q(dirx, diry, dirz)
        a, b, c = 1.0, 1.0, q

    type_host   = 'NFW'
    params_host = {'logM': logM, 'Rs': Rs,
                    'a': a, 'b': b, 'c': c,
                    'dirx': dirx, 'diry': diry, 'dirz': dirz,
                    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0}

    type_sat   = 'Plummer'
    params_sat = {'logM': logm, 'Rs': rs,
                    'x_origin': x0, 'y_origin': 0.0, 'z_origin': z0}

    xv_f = jnp.array([x0, 0.0, z0, vx0, vy0, vz0])

    _, _, xv_stream, xhi_stream = StreaMAX.generate_stream(
        xv_f, type_host, params_host, type_sat, params_sat,
        time, alpha, n_steps, n_particles, unroll)

    _, _, theta_stream, r_stream, _ = StreaMAX.get_stream_ordered(
        xv_stream[:, 0], xv_stream[:, 1], xhi_stream)

    return theta_stream, r_stream, xv_stream
