import argparse
import json
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
from scipy.stats import gaussian_kde, norm
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({'font.size': 18})

from utils import *
from prior import *
from llikelihood import *
from fit import *

KM_S_TO_KPC_GYR = 1.0227121650537077
PATH_DATA_DEFAULT = Path('/data/dc824-2/SGA_Streams')
PATH_OUT_DEFAULT = Path('/data/dc824-2/SGA_Streams_Kinematics')
DEFAULT_STREAMS = ['UGC01424_factor3.0_pixscale0.6']
LIKELIHOOD_MODES = ('track', 'track_los')
LEGACY_LABELS = (
    'logM', 'Rs', 'dirx', 'diry', 'dirz', 'logm', 'rs',
    'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time', 'sig',
)
SCALE_FREE_LABELS = (
    'logM', 'logRs', 'dirx', 'diry', 'dirz', 'log_mfrac', 'logrs',
    'x0', 'z0', 'theta_v', 'phi_v', 'log_alpha', 'log_tau', 'sig',
)


def kde_eval(samples, xgrid, bw_method=None):
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 2:
        return np.full_like(xgrid, np.nan, dtype=float)
    kde = gaussian_kde(samples, bw_method=bw_method)
    return kde(xgrid)

def uniform_pdf(x, a, b):
    y = np.zeros_like(x, dtype=float)
    m = (x >= a) & (x <= b)
    y[m] = 1.0 / (b - a)
    return y

def gaussian_pdf(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)

def nice_name(name):
    return name.split("_factor")[0]


def fit_results_path(path_out, name, parameterization, mode, ndim, n_particles, n_min, var_ratio, nlive, v_host=None, v_err_host=None):
        tag = (
            f'{parameterization}_{mode}_ndim{ndim}_Nparticles{n_particles}'
            f'_Nmin{n_min}_VarRatio{var_ratio}_nlive{nlive}'
            f'_vhost{v_host}_vhosterr{v_err_host}'
        )
        output_dir = Path(path_out) / name / tag
        return output_dir, output_dir / 'dict_results.pkl'


def get_host_logmass_truths(name, cfg, strings_catalogue_by_name=None, catalogue_index=None):
        # New estimate: logMtotal directly from STRRINGS csv
        log_mtotal_new = float(cfg.get('logMtotal', np.nan))

        # Old estimate: M_stream / M_host ratio from STRRINGS_catalogue
        log_mtotal_old = np.nan
        if strings_catalogue_by_name is not None:
            cat_row = strings_catalogue_by_name.get(name)
            if cat_row is not None:
                m_stream = pd.to_numeric(cat_row.get('M_stream', np.nan), errors='coerce')
                mratio = pd.to_numeric(cat_row.get('M_stream/M_host', np.nan), errors='coerce')
                if np.isfinite(m_stream) and np.isfinite(mratio) and mratio > 0:
                    m_stellar = m_stream / mratio
                    log_mtotal_old = float(np.log10(halo_mass_from_stellar_mass(m_stellar)))
        if not np.isfinite(log_mtotal_old) and catalogue_index is not None:
            m_stream = pd.to_numeric(catalogue_index.get('M_stream', np.nan), errors='coerce')
            mratio = pd.to_numeric(catalogue_index.get('M_stream/M_host', np.nan), errors='coerce')
            if np.isfinite(m_stream) and np.isfinite(mratio) and mratio > 0:
                m_stellar = m_stream / mratio
                log_mtotal_old = float(np.log10(halo_mass_from_stellar_mass(m_stellar)))

        return log_mtotal_new, log_mtotal_old

def extra_processing(name, dict_data):
        if name == 'NGC1084_GROUP_factor2.5_pixscale0.6':
            arg_region = np.array([ 4, 6, 9, 14, 18, 22, 25, 28, 30, 33, 35, 37, 40, 42])
            for ar in range(len(arg_region)-1):
                dict_data['theta'][arg_region[ar]:arg_region[ar+1]] = dict_data['theta'][arg_region[ar]]
                dict_data['r'][arg_region[ar]:arg_region[ar+1]] = dict_data['r'][arg_region[ar]]
                dict_data['r_err'][arg_region[ar]:arg_region[ar+1]] = dict_data['r_err'][arg_region[ar]]
            keep_indices = [arg_region[ar] for ar in range(len(arg_region)-1)]
            all_indices = np.arange(len(dict_data['theta']))
            region_indices = np.concatenate([np.arange(arg_region[ar], arg_region[ar+1]) for ar in range(len(arg_region)-1)])
            other_indices = np.setdiff1d(all_indices, region_indices)
            final_indices = np.sort(np.concatenate([keep_indices, other_indices]))
            dict_data['theta'] = dict_data['theta'][final_indices]
            dict_data['r'] = dict_data['r'][final_indices]
            dict_data['r_err'] = dict_data['r_err'][final_indices]
        elif name == 'NGC1121_factor6.5_pixscale0.6':
            arg_region = np.array([ 2, 6, 10, 15, 17, 19, 21, 23, 25, 27, 30, 35, 40, 42, 45, 47, 49, 52, 55, 57, 60, 64, 67, 69])
            for ar in range(len(arg_region)-1):
                dict_data['theta'][arg_region[ar]:arg_region[ar+1]] = dict_data['theta'][arg_region[ar]]
                dict_data['r'][arg_region[ar]:arg_region[ar+1]] = dict_data['r'][arg_region[ar]]
                dict_data['r_err'][arg_region[ar]:arg_region[ar+1]] = dict_data['r_err'][arg_region[ar]]
            keep_indices = [arg_region[ar] for ar in range(len(arg_region)-1)]
            all_indices = np.arange(len(dict_data['theta']))
            region_indices = np.concatenate([np.arange(arg_region[ar], arg_region[ar+1]) for ar in range(len(arg_region)-1)])
            other_indices = np.setdiff1d(all_indices, region_indices)
            final_indices = np.sort(np.concatenate([keep_indices, other_indices]))
            dict_data['theta'] = dict_data['theta'][final_indices]
            dict_data['r'] = dict_data['r'][final_indices]
            dict_data['r_err'] = dict_data['r_err'][final_indices]
        elif name == 'PGC000902_factor4.0_pixscale0.6':
            arg_take = np.arange(len(dict_data['theta']))[16:]
            dict_data['theta'] = dict_data['theta'][arg_take]
            dict_data['r'] = dict_data['r'][arg_take]
            dict_data['r_err'] = dict_data['r_err'][arg_take]
        elif name == 'PGC039258_factor2.5_pixscale0.6':
            arg_region = np.array([ 2, 4, 7, 10, 12, 14, 16, 18, 21])
            for ar in range(len(arg_region)-1):
                dict_data['theta'][arg_region[ar]:arg_region[ar+1]] = dict_data['theta'][arg_region[ar]]
                dict_data['r'][arg_region[ar]:arg_region[ar+1]] = dict_data['r'][arg_region[ar]]
                dict_data['r_err'][arg_region[ar]:arg_region[ar+1]] = dict_data['r_err'][arg_region[ar]]
            keep_indices = [arg_region[ar] for ar in range(len(arg_region)-1)]
            all_indices = np.arange(len(dict_data['theta']))
            region_indices = np.concatenate([np.arange(arg_region[ar], arg_region[ar+1]) for ar in range(len(arg_region)-1)])
            other_indices = np.setdiff1d(all_indices, region_indices)
            final_indices = np.sort(np.concatenate([keep_indices, other_indices]))
            dict_data['theta'] = dict_data['theta'][final_indices]
            dict_data['r'] = dict_data['r'][final_indices]
            dict_data['r_err'] = dict_data['r_err'][final_indices]
        elif name == 'PGC1092512_factor2.5_pixscale0.6':
            arg_take = np.where((dict_data['theta'] > 10))[0]
            dict_data['theta'] = dict_data['theta'][arg_take]
            dict_data['r'] = dict_data['r'][arg_take]
            dict_data['r_err'] = dict_data['r_err'][arg_take]
        elif name == 'PGC938075_factor4.5_pixscale0.6':
            dict_data['r_err'] *= 100
            dict_data['r'] *= 100
        else:
            print('No extra processing for this stream')

        return dict_data

def parse_args():
    parser = argparse.ArgumentParser(description="Fit real streams with legacy or scale-free parameterizations.")
    parser.add_argument('--names', nargs='+', default=DEFAULT_STREAMS, help="Stream names to fit, or 'all' for every name in names.txt.")
    parser.add_argument('--parameterization', choices=['legacy', 'scale_free'], default='legacy')
    parser.add_argument('--mode', choices=['track', 'track_los', 'both'], default='track_los')
    parser.add_argument('--path-data', type=Path, default=PATH_DATA_DEFAULT)
    parser.add_argument('--path-out', type=Path, default=PATH_OUT_DEFAULT)
    parser.add_argument('--nlive', type=int, default=2000)
    parser.add_argument('--n-min', type=int, default=3)
    parser.add_argument('--var-ratio-v', type=float, default=9.0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def get_fit_spec(parameterization, mode):
    if parameterization == 'legacy':
        return {
            'ndim': len(LEGACY_LABELS),
            'labels': list(LEGACY_LABELS),
            'prior_fn': prior_transform,
            'logl_fn': logl_track if mode == 'track' else logl,
            'stream_fn': params_to_stream,
        }
    return {
        'ndim': len(SCALE_FREE_LABELS),
        'labels': list(SCALE_FREE_LABELS),
        'prior_fn': prior_transform_scale_free_real,
        'logl_fn': logl_scale_free_track if mode == 'track' else logl_scale_free_track_los,
        'stream_fn': params_to_stream_scale_free,
    }


def extract_q_samples(parameterization, samples):
    samples = np.asarray(samples)
    if parameterization in ('legacy', 'scale_free'):
        return np.asarray(get_q(samples[:, 2], samples[:, 3], samples[:, 4]))
    return np.asarray(samples[:, 2], dtype=float)


def save_corner_plot(output_dir, dict_results, labels, M_halo_new, M_halo_old):
    ndim = len(labels)
    figure = corner.corner(
        dict_results['samps'],
        labels=labels,
        color='blue',
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 16},
    )
    if np.isfinite(M_halo_new):
        ax_logM_hist = figure.axes[0]
        ax_logM_hist.axvline(M_halo_new, color='red', linestyle='--', lw=2)
        for i in range(1, ndim):
            ax_2d = figure.axes[i * ndim]
            ax_2d.axvline(M_halo_new, color='red', linestyle='--', lw=1, alpha=0.5)
    if np.isfinite(M_halo_old):
        ax_logM_hist = figure.axes[0]
        ax_logM_hist.axvline(M_halo_old, color='green', linestyle='--', lw=2)
        for i in range(1, ndim):
            ax_2d = figure.axes[i * ndim]
            ax_2d.axvline(M_halo_old, color='green', linestyle='--', lw=1, alpha=0.5)
    figure.savefig(output_dir / 'corner_plot.pdf')
    plt.close(figure)


def save_q_posterior(output_dir, q_samps):
    plt.figure(figsize=(8, 6))
    plt.hist(q_samps, bins=30, density=True, alpha=0.7, color='blue', range=(0.5, 1.5))
    plt.axvline(np.median(q_samps), color='blue', linestyle='--', lw=2)
    plt.axvline(np.percentile(q_samps, 16), color='blue', linestyle=':', lw=2)
    plt.axvline(np.percentile(q_samps, 84), color='blue', linestyle=':', lw=2)
    plt.axvline(1.0, color='k', linestyle='-', lw=2)
    plt.xlabel('Halo Flattening')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(output_dir / 'q_posterior.pdf')
    plt.close()


def save_best_fit_image(output_dir, path_data, name, dict_data, best_params, n_particles, stream_fn):
    theta_stream, r_stream, xv_stream = stream_fn(best_params, n_particles)
    r_bin, _, _ = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])
    xv_stream = np.asarray(xv_stream)
    r_bin = np.asarray(r_bin)
    if name == 'PGC938075_factor4.5_pixscale0.6':
        r_bin /= 100
        xv_stream /= 100
    x_bin = r_bin * np.cos(dict_data['theta'] + dict_data['delta_theta'])
    y_bin = r_bin * np.sin(dict_data['theta'] + dict_data['delta_theta'])

    x0 = xv_stream[:, 0]
    y0 = xv_stream[:, 1]
    dt = dict_data['delta_theta']
    c, s = np.cos(dt), np.sin(dt)
    x_stream = x0 * c - y0 * s
    y_stream = x0 * s + y0 * c

    sga = Table.read(path_data / 'SGA-2020.fits', hdu=1)
    residual, mask, z_redshift, pixel_to_kpc, PA = get_residuals_and_mask(path_data, sga, name)
    center_x, center_y = residual.shape[1]//2, residual.shape[0]//2

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(residual, origin='lower', cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(residual, origin='lower', cmap='gray')
    plt.scatter(x_stream / pixel_to_kpc + center_x, y_stream / pixel_to_kpc + center_y, c='blue', s=1, label='Best fit')
    plt.scatter(x_bin / pixel_to_kpc + center_x, y_bin / pixel_to_kpc + center_y, c='lime')
    plt.scatter(dict_data['x']/pixel_to_kpc + center_x, dict_data['y']/pixel_to_kpc + center_y, color='red', s=10, label='Data')
    plt.xlim(0, residual.shape[1])
    plt.ylim(0, residual.shape[0])
    plt.axis('off')
    plt.savefig(output_dir / 'image_best_fit.pdf')
    plt.close()


def save_mode_comparison(stream_root, name, parameterization, results_by_mode, M_halo_new, M_halo_old):
    if not {'track', 'track_los'}.issubset(results_by_mode):
        return

    track_results = results_by_mode['track']
    los_results = results_by_mode['track_los']
    logM_track = np.asarray(track_results['samps'][:, 0])
    logM_los = np.asarray(los_results['samps'][:, 0])
    q_track = extract_q_samples(parameterization, track_results['samps'])
    q_los = extract_q_samples(parameterization, los_results['samps'])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    fig.suptitle(f"{nice_name(name)} [{parameterization}]", y=1.02, fontsize=16)

    ax = axes[0]
    xM = np.linspace(11.0, 14.0, 600)
    ax.plot(xM, uniform_pdf(xM, 11.0, 14.0), color='0.6', lw=1.8, label='Prior')
    ax.plot(xM, kde_eval(logM_los, xM), color='tab:blue', lw=2.2, label='Track + LOS')
    ax.plot(xM, kde_eval(logM_track, xM), color='tab:orange', lw=2.2, label='Track only')
    if np.isfinite(M_halo_new):
        ax.axvline(M_halo_new, color='red', ls='--', lw=2.0, label='Literature (new)')
    if np.isfinite(M_halo_old):
        ax.axvline(M_halo_old, color='green', ls='--', lw=2.0, label='Catalogue (old)')
    ax.set_xlim(11.0, 14.0)
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo}/M_\odot)$')
    ax.set_ylabel('Density')
    ax.set_yticks([])

    ax = axes[1]
    xq = np.linspace(0.5, 1.5, 600)
    ax.plot(xq, uniform_pdf(xq, 0.5, 1.5), color='0.6', lw=1.8, label='Prior')
    ax.plot(xq, kde_eval(q_los, xq), color='tab:blue', lw=2.2, label='Track + LOS')
    ax.plot(xq, kde_eval(q_track, xq), color='tab:orange', lw=2.2, label='Track only')
    ax.set_xlim(0.5, 1.5)
    ax.set_xlabel(r'$q$')
    ax.set_yticks([])
    ax.legend(loc='upper right', frameon=False)

    for ax in axes:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    plt.tight_layout()
    plt.savefig(stream_root / f'{parameterization}_posterior_comparison_kde.pdf', bbox_inches='tight')
    plt.savefig(stream_root / f'{parameterization}_posterior_comparison_kde.png', bbox_inches='tight')
    plt.close()


def attach_los_data(dict_data, cfg):
    v = cfg.get('v', np.nan)
    v_err = cfg.get('v_err', np.nan)
    v_host = cfg.get('v_host', np.nan)
    v_err_host = cfg.get('v_err_host', np.nan)
    if not (np.isfinite(v) and np.isfinite(v_err) and v_err > 0):
        return False, v_host, v_err_host

    if np.isfinite(v_host) and np.isfinite(v_err_host) and v_err_host > 0:
        vz = float(v) - float(v_host)
        vz_err = np.sqrt(float(v_err)**2 + float(v_err_host)**2)
    else:
        vz = float(v)
        vz_err = float(v_err)
    dict_data['vz'] = vz * KM_S_TO_KPC_GYR
    dict_data['vz_err'] = vz_err * KM_S_TO_KPC_GYR
    dict_data['vz_theta'] = 0.0
    dict_data['vz_window'] = dict_data['bin_width']
    return True, v_host, v_err_host


def run_stream(name, args, strings_by_name, catalogue_by_name):
    if name in ['NGC1084_GROUP_factor2.5_pixscale0.6', 'NGC1121_factor6.5_pixscale0.6', 'PGC021008_factor2.5_pixscale0.6']:
        print(f"Skipping {name} due to known issues.")
        return

    track_path = args.path_data / name / 'dict_track.pkl'
    if not track_path.exists():
        print(f"Skipping {name}: missing {track_path}")
        return

    with open(track_path, 'rb') as f:
        dict_data = pickle.load(f)
    dict_data = extra_processing(name, dict_data)
    dict_data['delta_theta'] = np.median(dict_data['theta'])
    dict_data['theta'] -= dict_data['delta_theta']
    dict_data['bin_width'] = np.diff(dict_data['theta']).min()

    cfg = strings_by_name.get(name, {})
    var_ratio_i = float(cfg.get('Var ratio', np.nan))
    if not np.isfinite(var_ratio_i) or var_ratio_i <= 0:
        print(f"Skipping {name}: missing/invalid Var ratio in STRRINGS.csv")
        return

    n_particles_i = cfg.get('N', np.nan)
    if not (np.isfinite(n_particles_i) and n_particles_i > 0):
        print(f"Skipping {name}: missing/invalid N in STRRINGS.csv")
        return
    n_particles_i = int(n_particles_i)

    has_los, v_host, v_err_host = attach_los_data(dict_data, cfg)
    requested_modes = list(LIKELIHOOD_MODES) if args.mode == 'both' else [args.mode]
    modes = []
    for mode in requested_modes:
        if mode == 'track_los' and not has_los:
            print(f"Skipping {name} mode=track_los: missing/invalid numeric v or v_err in STRRINGS.csv")
            continue
        modes.append(mode)
    if not modes:
        return

    stream_root = args.path_out / name
    stream_root.mkdir(parents=True, exist_ok=True)

    M_halo_new, M_halo_old = get_host_logmass_truths(
        name,
        cfg,
        strings_catalogue_by_name=catalogue_by_name,
        catalogue_index=None,
    )

    results_by_mode = {}
    for mode in modes:
        fit_spec = get_fit_spec(args.parameterization, mode)
        output_dir, results_path = fit_results_path(
            args.path_out,
            name,
            args.parameterization,
            mode,
            fit_spec['ndim'],
            n_particles_i,
            args.n_min,
            var_ratio_i,
            args.nlive,
            v_host=v_host,
            v_err_host=v_err_host,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if results_path.exists() and not args.overwrite:
            print(f"Skipping fit for {name} [{args.parameterization}/{mode}]: existing output at {output_dir}")
            with open(results_path, 'rb') as f:
                dict_results = pickle.load(f)
        else:
            fit_config = {
                'name': name,
                'parameterization': args.parameterization,
                'mode': mode,
                'labels': fit_spec['labels'],
                'nlive': int(args.nlive),
                'n_particles': int(n_particles_i),
                'n_min': int(args.n_min),
                'var_ratio': float(var_ratio_i),
                'var_ratio_v': float(args.var_ratio_v),
                'v_host': None if not np.isfinite(v_host) else float(v_host),
                'v_err_host': None if not np.isfinite(v_err_host) else float(v_err_host),
            }
            with open(output_dir / 'fit_config.json', 'w', encoding='ascii') as f:
                json.dump(fit_config, f, indent=2)

            print(f"Fitting {name} [{args.parameterization}/{mode}] with nlive={args.nlive}")
            dict_results = dynesty_fit(
                dict_data,
                fit_spec['logl_fn'],
                fit_spec['prior_fn'],
                fit_spec['ndim'],
                n_particles=n_particles_i,
                n_min=args.n_min,
                var_ratio=var_ratio_i,
                var_ratio_v=args.var_ratio_v,
                nlive=args.nlive,
            )
            dict_results['labels'] = fit_spec['labels']
            dict_results['parameterization'] = args.parameterization
            dict_results['mode'] = mode
            with open(results_path, 'wb') as f:
                pickle.dump(dict_results, f)

        save_corner_plot(output_dir, dict_results, fit_spec['labels'], M_halo_new, M_halo_old)
        q_samps = extract_q_samples(args.parameterization, dict_results['samps'])
        save_q_posterior(output_dir, q_samps)
        best_params = dict_results['samps'][np.argmax(dict_results['logl'])]
        save_best_fit_image(
            output_dir,
            args.path_data,
            name,
            dict_data,
            best_params,
            n_particles_i,
            fit_spec['stream_fn'],
        )
        results_by_mode[mode] = dict_results

    save_mode_comparison(stream_root, name, args.parameterization, results_by_mode, M_halo_new, M_halo_old)


def main():
    args = parse_args()
    args.path_data = Path(args.path_data)
    args.path_out = Path(args.path_out)
    args.path_out.mkdir(parents=True, exist_ok=True)

    names_all = np.loadtxt(args.path_data / 'names.txt', dtype=str)
    selected_names = list(names_all) if args.names == ['all'] else list(args.names)

    strings_df = pd.read_csv('STRRINGS.csv')
    strings_df = strings_df.rename(columns={c: c.strip() if isinstance(c, str) else c for c in strings_df.columns})
    strings_df['Name'] = strings_df['Name'].astype(str).str.strip()
    for col in ['sigma_ratio', 'N', 'Var ratio', 'v', 'v_err', 'v_host', 'v_err_host', 'logMstar', 'logMstar_err', 'logMtotal', 'logMtotal_err']:
        if col in strings_df.columns:
            strings_df[col] = pd.to_numeric(strings_df[col], errors='coerce')
    strings_by_name = strings_df.set_index('Name').to_dict(orient='index')

    STRRINGS_catalogue = pd.read_csv(args.path_data / 'STRRINGS_catalogue.csv')
    if 'Name' in STRRINGS_catalogue.columns:
        STRRINGS_catalogue['Name'] = STRRINGS_catalogue['Name'].astype(str).str.strip()
        catalogue_by_name = STRRINGS_catalogue.set_index('Name').to_dict(orient='index')
    else:
        catalogue_by_name = {}

    for name in tqdm(selected_names, leave=True):
        run_stream(name, args, strings_by_name, catalogue_by_name)


if __name__ == "__main__":
    main()
