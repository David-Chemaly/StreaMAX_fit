import jax
import jax.numpy as jnp

def prior_transform(p):
    # Keep unit-cube values away from exact 0/1 to avoid inf from ndtri.
    p = jnp.clip(p, 1e-12, 1.0 - 1e-12)
    ndim = len(p)
    if ndim == 14:
        logM, Rs, dirx, diry, dirz, \
        logm, rs, \
        x0, z0, vx0, vy0, vz0, \
        time, sig_r = p
        use_flattening = True
        sig_v = None
    elif ndim == 15:
        logM, Rs, dirx, diry, dirz, \
        logm, rs, \
        x0, z0, vx0, vy0, vz0, \
        time, sig_r, sig_v = p
        use_flattening = True
    elif ndim == 11:
        logM, Rs, \
        logm, rs, \
        x0, z0, vx0, vy0, vz0, \
        time, sig_r = p
        use_flattening = False
        sig_v = None
    elif ndim == 12:
        logM, Rs, \
        logm, rs, \
        x0, z0, vx0, vy0, vz0, \
        time, sig_r, sig_v = p
        use_flattening = False
    else:
        raise ValueError(f"prior_transform expects ndim in {{11,12,14,15}}, got {ndim}")

    logM1 = 11 + 3 * logM
    Rs1   = 5 + 20 * Rs

    if use_flattening:
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
    sig_r1 = 25 * sig_r

    if use_flattening:
        out = [
            logM1, Rs1, dirx1, diry1, dirz1,
            logm1, rs1,
            x1, z1, vx1, vy1, vz1,
            time1, sig_r1,
        ]
    else:
        out = [
            logM1, Rs1,
            logm1, rs1,
            x1, z1, vx1, vy1, vz1,
            time1, sig_r1,
        ]
    if sig_v is not None:
        out.append(25 * sig_v)

    return jnp.array(out)
