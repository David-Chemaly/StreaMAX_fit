import jax
import jax.numpy as jnp

def prior_transform(p):
    #ndim = 14
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
    z1 = jax.scipy.special.ndtri(z0) * 150

    vx1 = jax.scipy.special.ndtri(vx0) * 250
    vy1 = jax.scipy.special.ndtri(0.5 + vy0 / 2) * 250
    vz1 = jax.scipy.special.ndtri(vz0) * 250

    time1 = 1 + 3*time
    sig1  = 25 * sig

    
    return jnp.array([
        logM1, Rs1, dirx1, diry1, dirz1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        time1, sig1,
    ])

def prior_transform_unif(p):
    #ndim = 14
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
    z1 = jax.scipy.special.ndtri(z0) * 150

    vx1 = jax.scipy.special.ndtri(vx0) * 250
    vy1 = jax.scipy.special.ndtri(0.5 + vy0 / 2) * 250
    vz1 = jax.scipy.special.ndtri(vz0) * 250

    time1 = 1 + 3*time
    sig1  = 25 * sig

    
    return jnp.array([
        logM1, Rs1, dirx1, diry1, dirz1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        time1, sig1,
    ])

def prior_transform_v(p):
    #ndim = 15
    logM, Rs, dirx, diry, dirz, \
    logm, rs, \
    x0, z0, vx0, vy0, vz0, \
    time, sig, sig_v = p

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

    time1 = 1 + 3*time
    sig1  = 25 * sig
    sig_v1 = 100 * sig_v
    
    return jnp.array([
        logM1, Rs1, dirx1, diry1, dirz1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        time1, sig1, sig_v1
    ])


def prior_transform_scale_free_real(p):
    # ndim = 14
    (
        logM,
        logRs,
        dirx,
        diry,
        dirz,
        log_mfrac,
        logrs,
        x0,
        z0,
        theta_v,
        phi_v,
        log_alpha,
        log_tau,
        sig,
    ) = p

    logM1 = 11.0 + 3.0 * logM
    logRs1 = jnp.log10(0.3) + (jnp.log10(300.0) - jnp.log10(0.3)) * logRs
    dirx1 = jax.scipy.special.ndtri(dirx)
    diry1 = jax.scipy.special.ndtri(diry)
    dirz1 = jax.scipy.special.ndtri(0.5 + dirz / 2.0)
    log_mfrac1 = -7.0 + 5.0 * log_mfrac
    logrs1 = jnp.log10(0.03) + (jnp.log10(30.0) - jnp.log10(0.03)) * logrs

    x1 = jax.scipy.special.ndtri(0.5 + x0 / 2.0) * 150.0
    z1 = jax.scipy.special.ndtri(z0) * 150.0

    theta_v1 = jnp.arcsin(2.0 * theta_v - 1.0)
    phi_v1 = 2.0 * jnp.pi * phi_v
    log_alpha1 = -1.0 + 2.0 * log_alpha
    log_tau1 = 2.0 * log_tau
    sig1 = 25.0 * sig

    return jnp.array(
        [
            logM1,
            logRs1,
            dirx1,
            diry1,
            dirz1,
            log_mfrac1,
            logrs1,
            x1,
            z1,
            theta_v1,
            phi_v1,
            log_alpha1,
            log_tau1,
            sig1,
        ]
    )
