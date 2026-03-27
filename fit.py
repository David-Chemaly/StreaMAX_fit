import time as timer
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import tqdm

import blackjax
from blackjax.ns.utils import finalise, sample as ns_sample, ess as ns_ess

from prior import logprior_axi, logprior_tri, sample_prior_axi, sample_prior_tri
from llikelihood import logl


def blackjax_ns_fit(dict_data, n_particles=10000, n_min=3, var_ratio=9.0,
                    num_live=500, num_inner_steps=None,
                    dlogZ_threshold=-3, triaxial=False):
    jax_data = {k: jnp.array(v) for k, v in dict_data.items()}

    if triaxial:
        logprior_fn = logprior_tri
        sample_fn = sample_prior_tri
        ndim = 16
    else:
        logprior_fn = logprior_axi
        sample_fn = sample_prior_axi
        ndim = 14

    if num_inner_steps is None:
        num_inner_steps = ndim * 5

    num_delete = num_live // 2

    def loglikelihood_fn(params):
        return logl(params, jax_data, n_particles, n_min, var_ratio, triaxial)

    # Sample initial live points from prior
    key = random.PRNGKey(42)
    key, init_key = random.split(key)
    positions = sample_fn(init_key, num_live)

    # Create sampler
    sampler = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )
    init_fn = jax.jit(sampler.init)
    step_fn = jax.jit(sampler.step)

    print(f'Nested sampling: {num_live} live points, {ndim}D, '
          f'{num_inner_steps} inner steps, num_delete={num_delete}')

    # Initialize
    print('JIT compiling init...')
    t_jit = timer.time()
    state = init_fn(positions)
    jax.block_until_ready(state)
    print(f'Init done in {timer.time() - t_jit:.1f}s')

    # Warm up step JIT
    print('JIT compiling step function (one-time cost)...')
    t_jit = timer.time()
    key, warmup_key = random.split(key)
    state, dead_info = step_fn(warmup_key, state)
    jax.block_until_ready(state)
    dead_list = [dead_info]
    print(f'JIT compilation done in {timer.time() - t_jit:.1f}s')

    # Run until convergence
    t0 = timer.time()
    n_dead = num_delete  # from warmup step
    with tqdm.tqdm(desc='Dead points', unit=' dead points') as pbar:
        while not (state.integrator.logZ_live - state.integrator.logZ < dlogZ_threshold):
            key, subkey = random.split(key)
            state, dead_info = step_fn(subkey, state)
            dead_list.append(dead_info)
            n_dead += num_delete
            pbar.update(num_delete)

            logZ = float(state.integrator.logZ)
            logZ_live = float(state.integrator.logZ_live)
            dlogZ_current = logZ_live - logZ
            pbar.set_postfix(logZ=f'{logZ:.1f}', logZ_live=f'{logZ_live:.1f}',
                             dlogZ=f'{dlogZ_current:.2f}')

    t_elapsed = timer.time() - t0
    print(f'Converged: {n_dead} dead points in {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)')

    # Finalize: combine dead particles + remaining live points
    final_info = finalise(state, dead_list)

    # Evidence
    logZ = float(jnp.logaddexp(state.integrator.logZ, state.integrator.logZ_live))

    # Resample to equal-weight posterior samples
    key, sample_key, ess_key = random.split(key, 3)
    posterior = ns_sample(sample_key, final_info, shape=10000)
    samps = np.array(posterior.position)
    logl_vals = np.array(posterior.loglikelihood)

    ess_val = float(ns_ess(ess_key, final_info))

    print(f'log Z = {logZ:.2f}, ESS = {ess_val:.0f}')

    return {
        'samps': samps,
        'logl': logl_vals,
        'log_Z': logZ,
        'ESS': ess_val,
    }
