import jax
import jax.numpy as jnp
import jax.random as random


def _log_uniform(x, low, high):
    return jnp.where((x >= low) & (x <= high), -jnp.log(high - low), -jnp.inf)

def _log_normal(x, loc, scale):
    return jax.scipy.stats.norm.logpdf(x, loc, scale)

def _log_halfnormal(x, scale):
    return jnp.where(x >= 0., jnp.log(2.) + jax.scipy.stats.norm.logpdf(x, 0., scale), -jnp.inf)


def logprior_axi(params):
    logM, Rs, dirx, diry, dirz, logm, rs, x0, z0, vx0, vy0, vz0, time, sig = params
    return (
        _log_uniform(logM, 11., 14.) + _log_uniform(Rs, 5., 25.)
        + _log_normal(dirx, 0., 1.) + _log_normal(diry, 0., 1.) + _log_halfnormal(dirz, 1.)
        + _log_uniform(logm, 7., 9.) + _log_uniform(rs, 1., 3.)
        + _log_halfnormal(x0, 150.) + _log_halfnormal(z0, 150.)
        + _log_normal(vx0, 0., 250.) + _log_halfnormal(vy0, 250.) + _log_normal(vz0, 0., 250.)
        + _log_uniform(time, 1., 4.) + _log_uniform(sig, 0., 25.)
    )


def logprior_tri(params):
    logM, Rs, p, q, dirx, diry, dirz, logm, rs, x0, z0, vx0, vy0, vz0, time, sig = params
    return (
        _log_uniform(logM, 11., 14.) + _log_uniform(Rs, 5., 25.)
        + _log_uniform(p, 0.5, 1.0)
        + jnp.where((q >= 0.5) & (q <= p), -jnp.log(jnp.maximum(p - 0.5, 1e-10)), -jnp.inf)
        + _log_normal(dirx, 0., 1.) + _log_normal(diry, 0., 1.) + _log_halfnormal(dirz, 1.)
        + _log_uniform(logm, 7., 9.) + _log_uniform(rs, 1., 3.)
        + _log_halfnormal(x0, 150.) + _log_halfnormal(z0, 150.)
        + _log_normal(vx0, 0., 250.) + _log_halfnormal(vy0, 250.) + _log_normal(vz0, 0., 250.)
        + _log_uniform(time, 1., 4.) + _log_uniform(sig, 0., 25.)
    )


def sample_prior_axi(key, n):
    keys = random.split(key, 14)
    return jnp.column_stack([
        random.uniform(keys[0], (n,), minval=11., maxval=14.),
        random.uniform(keys[1], (n,), minval=5., maxval=25.),
        random.normal(keys[2], (n,)),
        random.normal(keys[3], (n,)),
        jnp.abs(random.normal(keys[4], (n,))),
        random.uniform(keys[5], (n,), minval=7., maxval=9.),
        random.uniform(keys[6], (n,), minval=1., maxval=3.),
        jnp.abs(random.normal(keys[7], (n,))) * 150.,
        jnp.abs(random.normal(keys[8], (n,))) * 150.,
        random.normal(keys[9], (n,)) * 250.,
        jnp.abs(random.normal(keys[10], (n,))) * 250.,
        random.normal(keys[11], (n,)) * 250.,
        random.uniform(keys[12], (n,), minval=1., maxval=4.),
        random.uniform(keys[13], (n,), minval=0., maxval=25.),
    ])


def sample_prior_tri(key, n):
    keys = random.split(key, 16)
    p = random.uniform(keys[2], (n,), minval=0.5, maxval=1.0)
    q = 0.5 + random.uniform(keys[3], (n,)) * (p - 0.5)
    return jnp.column_stack([
        random.uniform(keys[0], (n,), minval=11., maxval=14.),
        random.uniform(keys[1], (n,), minval=5., maxval=25.),
        p, q,
        random.normal(keys[4], (n,)),
        random.normal(keys[5], (n,)),
        jnp.abs(random.normal(keys[6], (n,))),
        random.uniform(keys[7], (n,), minval=7., maxval=9.),
        random.uniform(keys[8], (n,), minval=1., maxval=3.),
        jnp.abs(random.normal(keys[9], (n,))) * 150.,
        jnp.abs(random.normal(keys[10], (n,))) * 150.,
        random.normal(keys[11], (n,)) * 250.,
        jnp.abs(random.normal(keys[12], (n,))) * 250.,
        random.normal(keys[13], (n,)) * 250.,
        random.uniform(keys[14], (n,), minval=1., maxval=4.),
        random.uniform(keys[15], (n,), minval=0., maxval=25.),
    ])
