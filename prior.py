import jax
import jax.numpy as jnp


def _clip_unit_cube(p):
    # Keep unit-cube values away from exact 0/1 to avoid inf from ndtri.
    return jnp.clip(p, 1e-12, 1.0 - 1e-12)


def _common_transform(logM, Rs, logm, rs, x0, z0, vx0, vy0, vz0, time, sig_r):
    logM1 = 11 + 3 * logM
    Rs1 = 5 + 20 * Rs

    logm1 = 7 + 2 * logm
    rs1 = 1 + 2 * rs

    x1 = jax.scipy.special.ndtri(0.5 + x0 / 2) * 150
    z1 = jax.scipy.special.ndtri(0.5 + z0 / 2) * 150

    vx1 = jax.scipy.special.ndtri(vx0) * 250
    vy1 = jax.scipy.special.ndtri(0.5 + vy0 / 2) * 250
    vz1 = jax.scipy.special.ndtri(vz0) * 250

    time1 = 1 + 3 * time
    sig_r1 = 25 * sig_r

    return logM1, Rs1, logm1, rs1, x1, z1, vx1, vy1, vz1, time1, sig_r1


def prior_transform_flat_pos(p):
    p = _clip_unit_cube(p)
    logM, Rs, dirx, diry, dirz, logm, rs, x0, z0, vx0, vy0, vz0, time, sig_r = p
    logM1, Rs1, logm1, rs1, x1, z1, vx1, vy1, vz1, time1, sig_r1 = _common_transform(
        logM, Rs, logm, rs, x0, z0, vx0, vy0, vz0, time, sig_r
    )
    dirx1 = jax.scipy.special.ndtri(dirx)
    diry1 = jax.scipy.special.ndtri(diry)
    dirz1 = jax.scipy.special.ndtri(0.5 + dirz / 2)

    return jnp.array([
        logM1, Rs1, dirx1, diry1, dirz1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        time1, sig_r1,
    ])


def prior_transform_flat_kin(p):
    p = _clip_unit_cube(p)
    logM, Rs, dirx, diry, dirz, logm, rs, x0, z0, vx0, vy0, vz0, time, sig_r, sig_v = p
    logM1, Rs1, logm1, rs1, x1, z1, vx1, vy1, vz1, time1, sig_r1 = _common_transform(
        logM, Rs, logm, rs, x0, z0, vx0, vy0, vz0, time, sig_r
    )
    dirx1 = jax.scipy.special.ndtri(dirx)
    diry1 = jax.scipy.special.ndtri(diry)
    dirz1 = jax.scipy.special.ndtri(0.5 + dirz / 2)
    sig_v1 = 25 * sig_v

    return jnp.array([
        logM1, Rs1, dirx1, diry1, dirz1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        time1, sig_r1, sig_v1,
    ])


def prior_transform_sph_pos(p):
    p = _clip_unit_cube(p)
    logM, Rs, logm, rs, x0, z0, vx0, vy0, vz0, time, sig_r = p
    logM1, Rs1, logm1, rs1, x1, z1, vx1, vy1, vz1, time1, sig_r1 = _common_transform(
        logM, Rs, logm, rs, x0, z0, vx0, vy0, vz0, time, sig_r
    )

    return jnp.array([
        logM1, Rs1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        time1, sig_r1,
    ])


def prior_transform_sph_kin(p):
    p = _clip_unit_cube(p)
    logM, Rs, logm, rs, x0, z0, vx0, vy0, vz0, time, sig_r, sig_v = p
    logM1, Rs1, logm1, rs1, x1, z1, vx1, vy1, vz1, time1, sig_r1 = _common_transform(
        logM, Rs, logm, rs, x0, z0, vx0, vy0, vz0, time, sig_r
    )
    sig_v1 = 25 * sig_v

    return jnp.array([
        logM1, Rs1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        time1, sig_r1, sig_v1,
    ])


def get_prior_transform(use_kinematics, use_flattening):
    if use_flattening and use_kinematics:
        return prior_transform_flat_kin
    if use_flattening and (not use_kinematics):
        return prior_transform_flat_pos
    if (not use_flattening) and use_kinematics:
        return prior_transform_sph_kin
    return prior_transform_sph_pos


def prior_transform(p):
    # Backward-compatible dispatch by ndim.
    funcs = {
        11: prior_transform_sph_pos,
        12: prior_transform_sph_kin,
        14: prior_transform_flat_pos,
        15: prior_transform_flat_kin,
    }
    ndim = len(p)
    if ndim not in funcs:
        raise ValueError(f"prior_transform expects ndim in {{11,12,14,15}}, got {ndim}")
    return funcs[ndim](p)
