import StreaMAX

import os
import json
import argparse
import jax
import jax.numpy as jnp
import pickle
import corner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from utils import params_to_stream, get_q
from fit import blackjax_ns_fit

LABELS_AXI = ['logM', 'Rs', 'dirx', 'diry', 'dirz',
              'logm', 'rs', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time', 'sig']
LABELS_TRI = ['logM', 'Rs', 'p', 'q', 'dirx', 'diry', 'dirz',
              'logm', 'rs', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time', 'sig']


def load_stream_data(csv_path, n_bins=36, min_count=3):
    df = pd.read_csv(csv_path)
    x, y = df['x'].values, df['y'].values
    chi = df['phase_chi'].values
    accreted = df['AccretedFlag'].values

    arg_sort = np.argsort(chi)
    x_sorted = x[arg_sort]
    y_sorted = y[arg_sort]
    accreted_sorted = accreted[arg_sort]

    r = np.sqrt(x_sorted**2 + y_sorted**2)
    theta = np.unwrap(np.arctan2(y_sorted, x_sorted))

    theta_prog = theta[accreted_sorted == 1].mean()
    theta -= theta_prog

    theta_edges = np.linspace(-2 * np.pi, 2 * np.pi, n_bins + 1)
    bin_width = theta_edges[1] - theta_edges[0]
    binned = np.digitize(theta, bins=theta_edges)

    theta_data, r_data, r_sig, counts = [], [], [], []
    for i in range(1, n_bins + 1):
        mask = binned == i
        n = mask.sum()
        if n >= min_count:
            theta_data.append(0.5 * (theta_edges[i - 1] + theta_edges[i]))
            r_data.append(np.mean(r[mask]))
            r_sig.append(np.std(r[mask]))
            counts.append(n)

    theta_data = np.array(theta_data)
    r_data = np.array(r_data)
    r_sig = np.array(r_sig)
    counts = np.array(counts)
    r_err = r_sig / np.sqrt(counts)

    return {
        'theta': theta_data, 'r': r_data, 'r_err': r_err,
        'bin_width': bin_width, 'delta_theta': theta_prog,
        'r_sig': r_sig, 'counts': counts,
    }


def plot_corner(dict_results, path, triaxial=False):
    labels = LABELS_TRI if triaxial else LABELS_AXI
    figure = corner.corner(dict_results['samps'],
                labels=labels, color='blue',
                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                title_kwargs={"fontsize": 12})
    figure.savefig(os.path.join(path, 'corner_plot.pdf'), bbox_inches='tight')
    plt.close(figure)


def plot_best_fit(dict_data, dict_results, path, n_particles, triaxial=False):
    best_params = dict_results['samps'][np.argmax(dict_results['logl'])]
    theta_stream, r_stream, xv_stream = params_to_stream(best_params, n_particles, triaxial=triaxial)

    r_bin, _, _ = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(
        theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])

    dt = dict_data['delta_theta']
    c, s = np.cos(dt), np.sin(dt)
    x0 = np.array(xv_stream[:, 0])
    y0 = np.array(xv_stream[:, 1])
    x_stream = x0 * c - y0 * s
    y_stream = x0 * s + y0 * c

    theta_abs = dict_data['theta'] + dt
    x_data = dict_data['r'] * np.cos(theta_abs)
    y_data = dict_data['r'] * np.sin(theta_abs)

    r_bin_np = np.array(r_bin)
    x_bin = r_bin_np * np.cos(theta_abs)
    y_bin = r_bin_np * np.sin(theta_abs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].scatter(x_stream, y_stream, c='blue', s=1, alpha=0.3, label='Best fit stream')
    axes[0].scatter(x_bin, y_bin, c='lime', s=40, zorder=5, label='Model track')
    axes[0].scatter(x_data, y_data, c='red', s=30, zorder=6, label='Data')
    axes[0].set_xlabel('X (kpc)')
    axes[0].set_ylabel('Y (kpc)')
    axes[0].set_aspect('equal')
    axes[0].legend(fontsize=12)

    axes[1].errorbar(dict_data['theta'], dict_data['r'], yerr=dict_data['r_err'],
                     fmt='o', color='red', capsize=3, label='Data')
    axes[1].scatter(dict_data['theta'], r_bin_np, c='lime', s=40, zorder=5, label='Model track')
    axes[1].scatter(np.array(theta_stream), np.array(r_stream), c='blue', s=1, alpha=0.1)
    axes[1].set_xlabel(r'$\theta$ (rad)')
    axes[1].set_ylabel('r (kpc)')
    axes[1].legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(path, 'best_fit.pdf'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_flattening(dict_results, path, triaxial=False):
    samps = dict_results['samps']

    if triaxial:
        p_samps = samps[:, 2]
        q_samps = samps[:, 3]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(p_samps, bins=30, density=True, alpha=0.7, color='blue', range=(0.5, 1.0))
        axes[0].axvline(np.median(p_samps), color='blue', ls='--', lw=2)
        axes[0].axvline(np.percentile(p_samps, 16), color='blue', ls=':', lw=2)
        axes[0].axvline(np.percentile(p_samps, 84), color='blue', ls=':', lw=2)
        axes[0].axvline(1.0, color='k', lw=2)
        axes[0].set_xlabel('p = b/a')
        axes[0].set_ylabel('Density')

        axes[1].hist(q_samps, bins=30, density=True, alpha=0.7, color='blue', range=(0.5, 1.0))
        axes[1].axvline(np.median(q_samps), color='blue', ls='--', lw=2)
        axes[1].axvline(np.percentile(q_samps, 16), color='blue', ls=':', lw=2)
        axes[1].axvline(np.percentile(q_samps, 84), color='blue', ls=':', lw=2)
        axes[1].axvline(1.0, color='k', lw=2)
        axes[1].set_xlabel('q = c/a')
        axes[1].set_ylabel('Density')

        axes[2].scatter(p_samps, q_samps, s=1, alpha=0.2, color='blue')
        axes[2].plot([0.5, 1.0], [0.5, 1.0], 'k--', lw=1)
        axes[2].set_xlabel('p = b/a')
        axes[2].set_ylabel('q = c/a')
        axes[2].set_xlim(0.5, 1.0)
        axes[2].set_ylim(0.5, 1.0)
        axes[2].set_aspect('equal')

    else:
        q_samps = get_q(samps[:, 2], samps[:, 3], samps[:, 4])

        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes.hist(q_samps, bins=30, density=True, alpha=0.7, color='blue', range=(0.5, 1.5))
        axes.axvline(np.median(q_samps), color='blue', ls='--', lw=2)
        axes.axvline(np.percentile(q_samps, 16), color='blue', ls=':', lw=2)
        axes.axvline(np.percentile(q_samps, 84), color='blue', ls=':', lw=2)
        axes.axvline(1.0, color='k', lw=2)
        axes.set_xlabel('Halo Flattening q')
        axes.set_ylabel('Density')

    fig.tight_layout()
    fig.savefig(os.path.join(path, 'flattening.pdf'), bbox_inches='tight', dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit a stellar stream with StreaMAX (BlackJAX NS)')
    parser.add_argument('csv', help='Path to the stream CSV file')
    parser.add_argument('-o', '--output', required=True, help='Output directory for results and plots')
    parser.add_argument('--triaxial', action='store_true', help='Use triaxial model (default: axisymmetric)')
    parser.add_argument('--num-live', type=int, default=500, help='Number of live points (default: 500)')
    parser.add_argument('--max-iterations', type=int, default=50000, help='Maximum NS iterations (default: 50000)')
    parser.add_argument('--dlogZ', type=float, default=0.01, help='Convergence threshold (default: 0.01)')
    parser.add_argument('--n-particles', type=int, default=10000, help='Stream particles per likelihood call (default: 10000)')
    parser.add_argument('--n-min', type=int, default=3, help='Minimum particles per model bin (default: 3)')
    parser.add_argument('--var-ratio', type=float, default=9.0, help='Variance ratio threshold (default: 9.0)')
    parser.add_argument('--n-bins', type=int, default=36, help='Number of angular bins for data (default: 36)')
    parser.add_argument('--min-count', type=int, default=3, help='Minimum particle count per data bin (default: 3)')
    args = parser.parse_args()

    # Build run subdirectory from hyperparameters
    run_name = f'nlive{args.num_live}_npart{args.n_particles}_nmin{args.n_min}_vr{args.var_ratio}'
    run_dir = os.path.join(args.output, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save hyperparameters
    hyperparams = {
        'num_live': args.num_live,
        'max_iterations': args.max_iterations,
        'dlogZ': args.dlogZ,
        'n_particles': args.n_particles,
        'n_min': args.n_min,
        'var_ratio': args.var_ratio,
        'n_bins': args.n_bins,
        'min_count': args.min_count,
    }
    with open(os.path.join(run_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)

    # Load data
    dict_data = load_stream_data(args.csv, n_bins=args.n_bins, min_count=args.min_count)
    print(f'Loaded {len(dict_data["theta"])} data points from {args.csv}')

    mode = 'triaxial' if args.triaxial else 'axisymmetric'
    print(f'Fitting {mode} model with BlackJAX NS on {jax.devices()[0]}')
    print(f'Run directory: {run_dir}')

    # Fit
    dict_results = blackjax_ns_fit(
        dict_data, n_particles=args.n_particles, n_min=args.n_min,
        var_ratio=args.var_ratio, num_live=args.num_live,
        max_iterations=args.max_iterations, dlogZ_threshold=args.dlogZ,
        triaxial=args.triaxial)

    # Save results
    with open(os.path.join(run_dir, 'dict_results.pkl'), 'wb') as f:
        pickle.dump(dict_results, f)
    with open(os.path.join(run_dir, 'dict_data.pkl'), 'wb') as f:
        pickle.dump(dict_data, f)

    print(f'Results saved to {run_dir}')
    print(f'log Z = {dict_results["log_Z"]:.2f}')
    print(f'ESS = {dict_results["ESS"]:.0f}')

    # Plots
    plot_corner(dict_results, run_dir, triaxial=args.triaxial)
    plot_best_fit(dict_data, dict_results, run_dir, args.n_particles, triaxial=args.triaxial)
    plot_flattening(dict_results, run_dir, triaxial=args.triaxial)
    print('Plots saved')
