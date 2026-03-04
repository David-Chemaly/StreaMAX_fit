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

KM_S_TO_KPC_GYR = 1.0227121650537077

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
    use_kinematics = True
    use_flattening = False
    if use_flattening and use_kinematics:
        ndim = 15
    elif use_flattening and (not use_kinematics):
        ndim = 14
    elif (not use_flattening) and use_kinematics:
        ndim = 12
    else:
        ndim = 11
    n_min = 3
    nlive = 2000
    var_ratio_vel = 9.0

    PATH_INPUT = f'/data/dc824-2/SGA_Streams'
    PATH_OUTPUT = f'/data/dc824-2/SGA_Streams_Kinematics'
    names = np.loadtxt(f'{PATH_INPUT}/names.txt', dtype=str)
    STRRINGS_catalogue = pd.read_csv(f'{PATH_INPUT}/STRRINGS_catalogue.csv')
    strings_df = pd.read_excel('STRRINGS.xlsx')
    strings_df = strings_df.rename(columns={c: c.strip() if isinstance(c, str) else c for c in strings_df.columns})
    needed_cols = ["Name", "sigma_ratio", "N", "Var ratio", "v", "v_err", "v_host", "v_err_host"]
    for col in needed_cols:
        if col not in strings_df.columns:
            print(f"Warning: column '{col}' not found in STRRINGS.xlsx")
    strings_df["Name"] = strings_df["Name"].astype(str).str.strip()
    strings_xlsx = strings_df.set_index("Name").to_dict(orient="index")

    index = -1
    for name in tqdm(names, leave=True):
        index += 1
        with open(f"{PATH_INPUT}/{name}/dict_track.pkl", "rb") as f:
            dict_data = pickle.load(f)
        dict_data = extra_processing(name, dict_data)

        # This sets the progenitor in the middle of the stream
        dict_data['delta_theta'] = np.median(dict_data['theta'])
        dict_data['theta'] -= dict_data['delta_theta']
        dict_data['bin_width'] = np.diff(dict_data['theta']).min()

        stream_cfg = strings_xlsx.get(name, {})
        var_ratio_i = stream_cfg.get('Var_ratio', np.nan)
        if not np.isfinite(var_ratio_i):
            print(f"Skipping {name}: missing/invalid Var ratio in STRRINGS.xlsx")
            continue
        var_ratio_i = float(var_ratio_i)
        var_ratio_vel_i = var_ratio_vel

        N_i = stream_cfg.get('N', np.nan)
        if not (np.isfinite(N_i) and N_i > 0):
            print(f"Skipping {name}: missing/invalid N in STRRINGS.xlsx")
            continue
        n_particles = int(N_i)

        if use_kinematics:
            sigma_ratio_i = stream_cfg.get('sigma_ratio', np.nan)
            vz = stream_cfg.get('v', np.nan)
            vz_err = stream_cfg.get('v_err', np.nan)
            v_host = stream_cfg.get('v_host', np.nan)
            v_err_host = stream_cfg.get('v_err_host', np.nan)
            if not np.isfinite(sigma_ratio_i):
                print(f"Skipping {name}: missing sigma_ratio in STRRINGS.xlsx")
                continue
            if sigma_ratio_i == 999:
                print(f"Skipping {name}: sigma_ratio == 999")
                continue
            if not (np.isfinite(vz) and np.isfinite(vz_err) and vz_err > 0):
                print(f"Skipping {name}: missing/invalid numeric v or v_err in STRRINGS.xlsx")
                continue
            if not (np.isfinite(v_host) and np.isfinite(v_err_host) and v_err_host > 0):
                print(f"Skipping {name}: missing/invalid numeric v_host or v_err_host in STRRINGS.xlsx")
                continue
            # Excel values are in km/s. Use peculiar velocity and combine independent errors.
            vz_pec = float(vz) - float(v_host)
            vz_pec_err = np.sqrt(float(vz_err)**2 + float(v_err_host)**2)
            dict_data['vz'] = vz_pec * KM_S_TO_KPC_GYR
            dict_data['vz_err'] = vz_pec_err * KM_S_TO_KPC_GYR
            dict_data['vz_theta'] = 0.0
            dict_data['vz_window'] = float(dict_data['bin_width']) / 2.0

        stream_output_dir = f'{PATH_OUTPUT}/{name}'
        new_PATH_DATA = f'{stream_output_dir}/Plots_fixedProg_Sig_Transform_ndim{ndim}_Nparticles{n_particles}_Nmin{n_min}_VarRatioP{var_ratio_i}_VarRatioV{var_ratio_vel_i}_nlive{nlive}'
        if not os.path.exists(new_PATH_DATA):         
            os.makedirs(new_PATH_DATA, exist_ok=True)
            
            M_stellar = STRRINGS_catalogue.iloc[index]['M_stream']/STRRINGS_catalogue.iloc[index]['M_stream/M_host']
            M_halo = np.log10(halo_mass_from_stellar_mass(M_stellar))

            print(f'Fitting {name} with nlive={nlive} and fixed progenitor at center (kinematics={use_kinematics}, flattening={use_flattening})')
            dict_results = dynesty_fit(dict_data, logl, prior_transform, ndim, n_particles=n_particles, n_min=n_min, var_ratio=var_ratio_i, nlive=nlive, use_kinematics=use_kinematics, use_flattening=use_flattening, var_ratio_vel=var_ratio_vel_i)
            with open(f'{new_PATH_DATA}/dict_results.pkl', 'wb') as f:
                pickle.dump(dict_results, f)

            # Plot and Save corner plot
            if use_flattening:
                labels = ['logM', 'Rs', 'dirx', 'diry', 'dirz', 'logm', 'rs', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time', 'sig_r']
                truths = [M_halo, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            else:
                labels = ['logM', 'Rs', 'logm', 'rs', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time', 'sig_r']
                truths = [M_halo, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            if use_kinematics:
                labels.append('sig_v')
                truths.append(np.nan)
            figure = corner.corner(dict_results['samps'], 
                        labels=labels,
                        color='blue',
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, 
                        title_kwargs={"fontsize": 16},
                        truths=truths,
                        truth_color='red',
                        )
            figure.savefig(f'{new_PATH_DATA}/corner_plot.pdf')
            plt.close(figure)

            # Plot and Save flattening
            plt.figure(figsize=(8, 6))
            if use_flattening:
                q_samps = get_q(dict_results['samps'][:, 2], dict_results['samps'][:, 3], dict_results['samps'][:, 4])
            else:
                q_samps = np.ones(len(dict_results['samps']))
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
            theta_stream, r_stream, xv_stream = params_to_stream(best_params, n_particles, use_flattening=use_flattening)
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

            sga = Table.read(f'{PATH_INPUT}/SGA-2020.fits', hdu=1)
            residual, mask, z_redshift, pixel_to_kpc, PA = get_residuals_and_mask(PATH_INPUT, sga, name)
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
