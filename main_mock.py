import StreaMAX

import os
import jax
import jax.numpy as jnp
import pickle
import corner
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.table import Table
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from utils import *
from prior import *
from llikelihood import *
from fit import *

def get_mock_data_stream(seed, sigma=2, ndim=14, min_count=100):
    is_data = False
    rng = np.random.default_rng(int(seed))

    # MW halo parameters
    params = np.zeros(ndim)
    params[0] = 12.0  # logM
    params[1] = 16.0  # Rs
    params[2] = 1.0   # dirx
    params[3] = 1.0   # diry
    params[4] = 1.0   # dirz

    while not is_data:
        # Resample parameters
        p = rng.uniform(0, 1, size=ndim)
        params[5:] = prior_transform(p)[5:]

        theta_stream, r_stream, _, xv_sat = params_to_stream_DiskNFW(params)
        theta_sat = jnp.unwrap(jnp.arctan2(xv_sat[:, 1], xv_sat[:, 0]))
        
        theta_bin = np.linspace(-2*np.pi, 2*np.pi, 36)
        bin_width  = theta_bin[1] - theta_bin[0]
        r_bin, w_bin, count = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, theta_bin, bin_width)

        arg_take = jnp.where(count > min_count)[0]
        theta_in = theta_bin[arg_take]
        r_in     = r_bin[arg_take]

        crit1 = jnp.all(jnp.diff(arg_take) == 1) # Must be continuous and
        crit2 = len(arg_take) > 9   # Must have at least 10 bins with more than 100 particles
        crit3 = jnp.nansum(r_in[:-1]*jnp.tanh(jnp.diff(theta_in))) > 100 # Must have length of at least 100kpc
        crit4 = jnp.min(r_stream) > 2  # Must be further than 2kpc minimum
        crit5 = jnp.max(r_stream) < 200  # Must be less than 200kpc
        crit6 = jnp.all(jnp.diff(theta_sat) > 0)  # Must be monotonic

        if crit1 and crit2 and crit3 and crit4 and crit5 and crit6: 
            is_data = True

    r_sig = r_in * sigma / 100
    r_obs = rng.normal(r_in, r_sig)

    dict_stream = {
        'theta': theta_in,
        'bin_width': bin_width,
        'r': r_obs,
        'r_err': r_sig,
        'count': count,

        'params': params,
        'theta_stream': theta_stream,
        'r_stream': r_stream,
        'x_stream': r_stream * jnp.cos(theta_stream),
        'y_stream': r_stream * jnp.sin(theta_stream),
        'x_sat': xv_sat[:, 0],
        'y_sat': xv_sat[:, 1],
    }

    return dict_stream

def plot_mock_data_stream(path, dict_stream):
    # Plot the stream
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(dict_stream['r']*np.cos(dict_stream['theta']), dict_stream['r']*np.sin(dict_stream['theta']), '-o', c='black')
    
    theta_stream, r_stream, _, _ = params_to_stream_DiskNFW(dict_stream['params'], disk_mass=0.)
    r_bin, _, _ = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_stream['theta'], dict_stream['bin_width'])
    x_bin = r_bin * np.cos(dict_stream['theta'])
    y_bin = r_bin * np.sin(dict_stream['theta'])
    plt.plot(x_bin, y_bin, '-o', c='red')
    
    plt.xlabel('X (kpc)')
    plt.ylabel('Y (kpc)')
    plt.axis('equal')


    plt.subplot(1, 2, 2)
    plt.plot(dict_stream['theta'], dict_stream['r'], '-o', c='black', label='with disk')
    plt.plot(dict_stream['theta'], r_bin, '-o', c='red', label='no disk')
    plt.legend()
    plt.xlabel('Angle (rad)')
    plt.ylabel('Radius (kpc)')

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'stream.pdf'))
    plt.close()

if __name__ == "__main__":
    N = 10
    seeds = np.arange(N)+1

    ndim  = 14
    n_min = 9
    nlive = 2000
    var_ratio = 1e-10
    n_particles = 10000

    sigma = 2

    for seed in tqdm(seeds, leave=True):
        path = f'/data/dc824-2/MockStreamsWrong/seed{seed}'

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

            # Generate mock data
            dict_data = get_mock_data_stream(seed, sigma, ndim, min_count=100)

            with open(os.path.join(path, 'dict_stream.pkl'), 'wb') as f:
                pickle.dump(dict_data, f)

            # Plot mock data stream
            plot_mock_data_stream(path, dict_data)

            # Fit the mock data
            print(f'Fitting {seed} with nlive={nlive}')
            dict_results = dynesty_fit(dict_data, logl, prior_transform, ndim, n_particles=n_particles, n_min=n_min, var_ratio=var_ratio, nlive=nlive)
            with open(f'{path}/dict_results.pkl', 'wb') as f:
                pickle.dump(dict_results, f)

            # Plot and Save corner plot
            labels = ['logM', 'Rs', 'dirx', 'diry', 'dirz', 'logm', 'rs', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time', 'sig']
            figure = corner.corner(dict_results['samps'], 
                        labels=labels,
                        color='blue',
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, 
                        title_kwargs={"fontsize": 16},
                        truths=[dict_data['params'][0], dict_data['params'][1], dict_data['params'][2], dict_data['params'][3], dict_data['params'][4],
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        truth_color='red',
                        )
            figure.savefig(f'{path}/corner_plot.pdf')
            plt.close(figure)

            # Plot and Save Best fit on Data
            best_params = dict_results['samps'][np.argmax(dict_results['logl'])]
            theta_stream, r_stream, xv_stream = params_to_stream(best_params, n_particles)
            r_bin, _, _ = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])
            x_bin = r_bin * np.cos(dict_data['theta'])
            y_bin = r_bin * np.sin(dict_data['theta'])

            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.scatter(xv_stream[:, 0], xv_stream[:, 1], c='blue', s=1, label='Best fit')
            plt.scatter(x_bin, y_bin, color='red', s=10, label='Data')

            plt.subplot(1, 2, 2)
            q_samps = get_q(dict_results['samps'][:, 2], dict_results['samps'][:, 3], dict_results['samps'][:, 4])
            plt.hist(q_samps, bins=30, density=True, alpha=0.7, color='blue', range=(0.5, 1.5))
            plt.axvline(np.median(q_samps), color='blue', linestyle='--', lw=2)
            plt.axvline(np.percentile(q_samps, 16), color='blue', linestyle=':', lw=2)
            plt.axvline(np.percentile(q_samps, 84), color='blue', linestyle=':', lw=2)
            plt.axvline(1.0, color='k', linestyle='-', lw=2)
            plt.xlabel('Halo Flattening')
            plt.ylabel('Density')

            plt.tight_layout()
            plt.savefig(f'{path}/best_fit.pdf')
            plt.close()