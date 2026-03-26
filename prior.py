import jax
import jax.numpy as jnp

def prior_transform_axi(p):
    # ndim = 14
    logM, Rs, dirx, diry, dirz, \
    logm, rs, \
    x0, z0, vx0, vy0, vz0, \
    time, sig = p

    logM1 = 11 + 3 * logM
    Rs1   = 5 + 20 * Rs

    dirx1 = jax.scipy.special.ndtri(dirx)
    diry1 = jax.scipy.special.ndtri(diry)
    dirz1 = jax.scipy.special.ndtri(0.5 + dirz / 2)

    logm1 = 7 + 2 * logm
    rs1   = 1 + 2 * rs

    x1 = jax.scipy.special.ndtri(0.5 + x0 / 2) * 150
    z1 = jax.scipy.special.ndtri(0.5 + z0 / 2) * 150

    vx1 = jax.scipy.special.ndtri(vx0) * 250
    vy1 = jax.scipy.special.ndtri(0.5 + vy0 / 2) * 250
    vz1 = jax.scipy.special.ndtri(vz0) * 250

    time1 = 1 + 3 * time
    sig1  = 25 * sig

    return jnp.array([
        logM1, Rs1, dirx1, diry1, dirz1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        time1, sig1,
    ])

def prior_transform_tri(p):
    # ndim = 17
    logM, Rs, u_p, u_q, dirx, diry, dirz, \
    logm, rs, x0, z0, vx0, vy0, vz0, time, u_roll, sig = p

    logM1 = 11 + 3 * logM
    Rs1   = 5 + 20 * Rs

    # Axis ratios: independent U(0.5, 1.5), ordering enforced in likelihood
    p1 = 0.5 + 1.0 * u_p
    q1 = 0.5 + 1.0 * u_q

    # Orientation (same as axisymmetric: direction of c-axis)
    dirx1 = jax.scipy.special.ndtri(dirx)
    diry1 = jax.scipy.special.ndtri(diry)
    dirz1 = jax.scipy.special.ndtri(0.5 + dirz / 2)

    logm1 = 7 + 2 * logm
    rs1   = 1 + 2 * rs

    x1 = jax.scipy.special.ndtri(0.5 + x0 / 2) * 150
    z1 = jax.scipy.special.ndtri(0.5 + z0 / 2) * 150

    vx1 = jax.scipy.special.ndtri(vx0) * 250
    vy1 = jax.scipy.special.ndtri(0.5 + vy0 / 2) * 250
    vz1 = jax.scipy.special.ndtri(vz0) * 250

    time1 = 1 + 3 * time

    # Roll angle around c-axis: U(0, pi), pi-symmetry of ellipsoid
    roll1 = jnp.pi * u_roll

    sig1  = 25 * sig

    return jnp.array([
        logM1, Rs1, p1, q1, dirx1, diry1, dirz1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        time1, roll1, sig1,
    ])
