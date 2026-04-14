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
plt.rcParams.update({'font.size': 18})

from utils import *
from prior import *
from llikelihood import *
from fit import *

KM_S_TO_KPC_GYR = 1.0227121650537077


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


def fit_results_path(path_out, name, ndim, n_particles, n_min, var_ratio, nlive, v_host=None, v_err_host=None):
        output_dir = (
            f'{path_out}/{name}/Plots_ndim{ndim}_Nparticles{n_particles}'
            f'_Nmin{n_min}_VarRatio{var_ratio}_nlive{nlive}'
            f'_vhost{v_host}_vhosterr{v_err_host}'
        )
        return output_dir, f'{output_dir}/dict_results.pkl'


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

if __name__ == "__main__":
    ndim  = 14
    n_min = 3
    nlive = 2000
    var_ratio_v = 9.0
    logl_fn = logl
    prior_transform_fn = prior_transform

    PATH_DATA = f'/data/dc824-2/SGA_Streams'
    PATH_OUT  = f'/data/dc824-2/SGA_Streams_Kinematics'
    names = np.loadtxt(f'{PATH_DATA}/names.txt', dtype=str)
    STRRINGS_catalogue = pd.read_csv(f'{PATH_DATA}/STRRINGS_catalogue.csv')
    strings_df = pd.read_csv('STRRINGS.csv')
    strings_df = strings_df.rename(columns={c: c.strip() if isinstance(c, str) else c for c in strings_df.columns})
    strings_df['Name'] = strings_df['Name'].astype(str).str.strip()
    for col in ['sigma_ratio', 'N', 'Var ratio', 'v', 'v_err', 'v_host', 'v_err_host', 'logMstar', 'logMstar_err', 'logMtotal', 'logMtotal_err']:
        if col in strings_df.columns:
            strings_df[col] = pd.to_numeric(strings_df[col], errors='coerce')
    strings_by_name = strings_df.set_index('Name').to_dict(orient='index')
    if 'Name' in STRRINGS_catalogue.columns:
        STRRINGS_catalogue['Name'] = STRRINGS_catalogue['Name'].astype(str).str.strip()
        catalogue_by_name = STRRINGS_catalogue.set_index('Name').to_dict(orient='index')
    else:
        catalogue_by_name = {}

    index = -1
    for name in tqdm(names, leave=True):

        if name not in ['NGC5387_factor2.5_pixscale0.6', 'PGC1001085_factor3.5_pixscale0.6']:
            continue

        if name in ['NGC1084_GROUP_factor2.5_pixscale0.6', 'NGC1121_factor6.5_pixscale0.6', 'PGC021008_factor2.5_pixscale0.6']:
            print(f"Skipping {name} due to known issues.")
            continue

        index += 1

        with open(f"{PATH_DATA}/{name}/dict_track.pkl", "rb") as f:
            dict_data = pickle.load(f)
        dict_data = extra_processing(name, dict_data)

        # This sets the progenitor in the middle of the stream
        dict_data['delta_theta'] = np.median(dict_data['theta'])
        dict_data['theta'] -= dict_data['delta_theta']
        dict_data['bin_width'] = np.diff(dict_data['theta']).min()
        cfg = strings_by_name.get(name, {})
        var_ratio_i = cfg.get('Var ratio', np.nan)
        var_ratio_i = float(var_ratio_i)

        n_particles_i = cfg.get('N', np.nan)
        if not (np.isfinite(n_particles_i) and n_particles_i > 0):
            print(f"Skipping {name}: missing/invalid N in STRRINGS.csv")
            continue
        n_particles_i = int(n_particles_i)

        # Add velocity data for logl_v (km/s -> kpc/Gyr).
        v = cfg.get('v', np.nan)
        v_err = cfg.get('v_err', np.nan)
        v_host = cfg.get('v_host', np.nan)
        v_err_host = cfg.get('v_err_host', np.nan)
        if not (np.isfinite(v) and np.isfinite(v_err) and v_err > 0):
            print(f"Skipping {name}: missing/invalid numeric v or v_err in STRRINGS.csv")
            continue
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

        new_PATH_DATA, results_path = fit_results_path(
            PATH_OUT,
            name,
            ndim,
            n_particles_i,
            n_min,
            var_ratio_i,
            nlive,
            v_host=v_host,
            v_err_host=v_err_host,
        )
        if os.path.exists(new_PATH_DATA):
            print(f"Skipping {name}: existing output folder found at {new_PATH_DATA}")
            continue

        os.makedirs(new_PATH_DATA, exist_ok=True)

        M_halo_new, M_halo_old = get_host_logmass_truths(
            name,
            cfg,
            strings_catalogue_by_name=catalogue_by_name,
            catalogue_index=STRRINGS_catalogue.iloc[index] if index < len(STRRINGS_catalogue) else None,
        )

        print(f'Fitting {name} with nlive={nlive} and fixed progenitor at center')
        dict_results = dynesty_fit(dict_data, logl_fn, prior_transform_fn, ndim, n_particles=n_particles_i, n_min=n_min, var_ratio=var_ratio_i, var_ratio_v=var_ratio_v, nlive=nlive)
        with open(results_path, 'wb') as f:
            pickle.dump(dict_results, f)

        # Plot and Save corner plot
        labels = ['logM', 'Rs', 'dirx', 'diry', 'dirz', 'logm', 'rs', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time', 'sig']
        figure = corner.corner(dict_results['samps'],
                    labels=labels,
                    color='blue',
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_kwargs={"fontsize": 16},
                    truths=[M_halo_new, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, dict_data['vz'], np.nan, np.nan],
                    truth_color='red',
                    )
        # Overlay old catalogue-derived M_halo as a second truth on the logM panels
        if np.isfinite(M_halo_old):
            ax_logM_hist = figure.axes[0]
            ax_logM_hist.axvline(M_halo_old, color='green', linestyle='--', lw=2)
            for i in range(1, ndim):
                ax_2d = figure.axes[i * ndim]
                ax_2d.axvline(M_halo_old, color='green', linestyle='--', lw=1, alpha=0.5)
        figure.savefig(f'{new_PATH_DATA}/corner_plot.pdf')
        plt.close(figure)

        # Plot and Save flattening
        q_samps = get_q(dict_results['samps'][:, 2], dict_results['samps'][:, 3], dict_results['samps'][:, 4])
        plt.figure(figsize=(8, 6))
        plt.hist(q_samps, bins=30, density=True, alpha=0.7, color='blue', range=(0.5, 1.5))
        plt.axvline(np.median(q_samps), color='blue', linestyle='--', lw=2)
        plt.axvline(np.percentile(q_samps, 16), color='blue', linestyle=':', lw=2)
        plt.axvline(np.percentile(q_samps, 84), color='blue', linestyle=':', lw=2)
        plt.axvline(1.0, color='k', linestyle='-', lw=2)
        plt.xlabel('Halo Flattening')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(f'{new_PATH_DATA}/q_posterior.pdf')
        plt.close()

        # Plot and Save Best fit on Data
        best_params = dict_results['samps'][np.argmax(dict_results['logl'])]
        theta_stream, r_stream, xv_stream = params_to_stream(best_params, n_particles_i)
        r_bin, _, _ = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])
        if name == 'PGC938075_factor4.5_pixscale0.6':
            r_bin /= 100
            xv_stream /= 100
        x_bin = r_bin * np.cos(dict_data['theta'] + dict_data['delta_theta'])
        y_bin = r_bin * np.sin(dict_data['theta'] + dict_data['delta_theta'])

        # rotate Cartesian positions by +delta_theta
        x0 = xv_stream[:, 0]
        y0 = xv_stream[:, 1]
        dt = dict_data['delta_theta']
        c, s = np.cos(dt), np.sin(dt)
        x_stream = x0 * c - y0 * s
        y_stream = x0 * s + y0 * c

        sga = Table.read(f'{PATH_DATA}/SGA-2020.fits', hdu=1)
        residual, mask, z_redshift, pixel_to_kpc, PA = get_residuals_and_mask(PATH_DATA, sga, name)
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
        plt.savefig(f'{new_PATH_DATA}/image_best_fit.pdf')
        plt.close()

        # Plot and Save posterior comparison (Track + 1 LOS vs Track only)
        OG_results_path = (
            f'{PATH_DATA}/{name}/Plots_fixedProg_Sig_ndim{ndim}_Nparticles{n_particles_i}'
            f'_Nmin3_VarRatio{var_ratio_i}_nlive2000/dict_results.pkl'
        )
        if os.path.exists(OG_results_path):
            with open(OG_results_path, 'rb') as f:
                OG_dict_results = pickle.load(f)

            v_true = float(v) - float(v_host) if np.isfinite(v_host) else float(v)
            v_sig_true = np.sqrt(float(v_err)**2 + float(v_err_host)**2) if np.isfinite(v_err_host) else float(v_err)

            logM   = dict_results['samps'][:, 0]
            vz_samps = dict_results['samps'][:, 11]
            q_samps_kde = get_q(*dict_results['samps'][:, 2:5].T)

            OG_logM = OG_dict_results['samps'][:, 0]
            OG_vz   = OG_dict_results['samps'][:, 11]
            OG_q    = get_q(*OG_dict_results['samps'][:, 2:5].T)

            fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
            fig.suptitle(nice_name(name), y=1.02, fontsize=16)

            c_los   = "tab:blue"
            c_track = "tab:orange"
            c_prior = "0.6"
            c_meas  = "red"

            # logM panel
            ax = axes[0]
            xM = np.linspace(11.0, 14.0, 600)
            ax.plot(xM, uniform_pdf(xM, 11.0, 14.0), color=c_prior, lw=1.8, label="Prior", zorder=2)
            ax.plot(xM, kde_eval(logM, xM), color=c_los, lw=2.2, label="Track + 1 LOS", zorder=3)
            ax.plot(xM, kde_eval(OG_logM, xM), color=c_track, lw=2.2, label="Track only", zorder=3)
            if np.isfinite(M_halo_new):
                ax.axvline(M_halo_new, color='red', ls='--', lw=2.0, zorder=1, label="Literature (new)")
            if np.isfinite(M_halo_old):
                ax.axvline(M_halo_old, color='green', ls='--', lw=2.0, zorder=1, label="Catalogue (old)")
            ax.set_xlim(11.0, 14.0)
            ax.set_xlabel(r'$\log_{10}(M_{\rm halo}/M_\odot)$')
            ax.set_ylabel("Density")
            ax.set_yticks([])

            # vz panel
            ax = axes[1]
            vmin = min(np.nanpercentile(vz_samps, 0.5), np.nanpercentile(OG_vz, 0.5), -900, v_true - 4*v_sig_true)
            vmax = max(np.nanpercentile(vz_samps, 99.5), np.nanpercentile(OG_vz, 99.5),  900, v_true + 4*v_sig_true)
            xV = np.linspace(vmin, vmax, 800)
            ax.plot(xV, gaussian_pdf(xV, 0.0, 250.0), color=c_prior, lw=1.8, label="Prior")
            ax.plot(xV, kde_eval(vz_samps, xV), color=c_los, lw=2.2, label="Track + 1 LOS")
            ax.plot(xV, kde_eval(OG_vz, xV), color=c_track, lw=2.2, label="Track only")
            if np.isfinite(v_true):
                ax.axvline(v_true, color=c_meas, ls="--", lw=2.0, label="Measured LOS")
                if np.isfinite(v_sig_true):
                    ax.axvline(v_true - v_sig_true, color=c_meas, ls=":", lw=1.6)
                    ax.axvline(v_true + v_sig_true, color=c_meas, ls=":", lw=1.6)
            ax.set_xlim(vmin, vmax)
            ax.set_xlabel(r'$v_z\ [{\rm km\,s^{-1}}]$')
            ax.set_yticks([])

            # q panel
            ax = axes[2]
            xq = np.linspace(0.5, 1.5, 600)
            ax.plot(xq, uniform_pdf(xq, 0.5, 1.5), color=c_prior, lw=1.8, label="Prior")
            ax.plot(xq, kde_eval(q_samps_kde, xq), color=c_los, lw=2.2, label="Track + 1 LOS")
            ax.plot(xq, kde_eval(OG_q, xq), color=c_track, lw=2.2, label="Track only")
            ax.set_xlim(0.5, 1.5)
            ax.set_xlabel(r'$q$')
            ax.set_yticks([])
            ax.legend(loc="upper right", frameon=False)

            for ax in axes:
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)

            plt.tight_layout()
            plt.savefig(f'{new_PATH_DATA}/posterior_comparison_kde.pdf', bbox_inches="tight")
            plt.savefig(f'{new_PATH_DATA}/posterior_comparison_kde.png', bbox_inches="tight")
            plt.close()
        else:
            print(f"Skipping posterior comparison for {name}: missing OG results")
