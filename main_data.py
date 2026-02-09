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

def extra_processing(name, dict_data):
        if name == 'NGC1084_GROUP_factor2.5_pixscale0.6':
            arg_region = np.array([ 4, 9, 14, 22, 28, 33, 35, 40, 42])
            for ar in range(len(arg_region)-1):
                dict_data['theta'][arg_region[ar]:arg_region[ar+1]] = np.median(dict_data['theta'][arg_region[ar]:arg_region[ar+1]])
                dict_data['r'][arg_region[ar]:arg_region[ar+1]] = np.mean(dict_data['r'][arg_region[ar]:arg_region[ar+1]])
                dict_data['r_err'][arg_region[ar]:arg_region[ar+1]] = np.sqrt(np.mean(dict_data['r_err'][arg_region[ar]:arg_region[ar+1]]**2))
            keep_indices = [arg_region[ar] for ar in range(len(arg_region)-1)]
            all_indices = np.arange(len(dict_data['theta']))
            region_indices = np.concatenate([np.arange(arg_region[ar], arg_region[ar+1]) for ar in range(len(arg_region)-1)])
            other_indices = np.setdiff1d(all_indices, region_indices)
            final_indices = np.sort(np.concatenate([keep_indices, other_indices]))
            dict_data['theta'] = dict_data['theta'][final_indices]
            dict_data['r'] = dict_data['r'][final_indices]
            dict_data['r_err'] = dict_data['r_err'][final_indices]
        elif name == 'NGC1121_factor6.5_pixscale0.6':
            arg_region = np.array([ 2, 8, 13, 17, 21, 25, 30, 35, 40, 44, 49, 55, 60, 64, 69])
            for ar in range(len(arg_region)-1):
                dict_data['theta'][arg_region[ar]:arg_region[ar+1]] = np.median(dict_data['theta'][arg_region[ar]:arg_region[ar+1]])
                dict_data['r'][arg_region[ar]:arg_region[ar+1]] = np.mean(dict_data['r'][arg_region[ar]:arg_region[ar+1]])
                dict_data['r_err'][arg_region[ar]:arg_region[ar+1]] = np.sqrt(np.mean(dict_data['r_err'][arg_region[ar]:arg_region[ar+1]]**2))
            keep_indices = [arg_region[ar] for ar in range(len(arg_region)-1)]
            all_indices = np.arange(len(dict_data['theta']))
            region_indices = np.concatenate([np.arange(arg_region[ar], arg_region[ar+1]) for ar in range(len(arg_region)-1)])
            other_indices = np.setdiff1d(all_indices, region_indices)
            final_indices = np.sort(np.concatenate([keep_indices, other_indices]))
            dict_data['theta'] = dict_data['theta'][final_indices]
            dict_data['r'] = dict_data['r'][final_indices]
            dict_data['r_err'] = dict_data['r_err'][final_indices]
        elif name == 'PGC000902_factor4.0_pixscale0.6':
            arg_take = np.where((dict_data['theta'] > 8.5))[0]
            dict_data['theta'] = dict_data['theta'][arg_take]
            dict_data['r'] = dict_data['r'][arg_take]
            dict_data['r_err'] = dict_data['r_err'][arg_take]
        elif name == 'PGC039258_factor2.5_pixscale0.6':
            arg_region = np.array([ 2, 7, 12, 16, 21])
            for ar in range(len(arg_region)-1):
                dict_data['theta'][arg_region[ar]:arg_region[ar+1]] = np.median(dict_data['theta'][arg_region[ar]:arg_region[ar+1]])
                dict_data['r'][arg_region[ar]:arg_region[ar+1]] = np.mean(dict_data['r'][arg_region[ar]:arg_region[ar+1]])
                dict_data['r_err'][arg_region[ar]:arg_region[ar+1]] = np.sqrt(np.mean(dict_data['r_err'][arg_region[ar]:arg_region[ar+1]]**2))
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
    var_ratio = 9.0
    n_particles_per_point = 1500
    n_particles_min = 10000

    PATH_DATA = f'/data/dc824-2/SGA_Streams'
    names = np.loadtxt(f'{PATH_DATA}/names.txt', dtype=str)
    STRRINGS_catalogue = pd.read_csv(f'{PATH_DATA}/STRRINGS_catalogue.csv')

    list_undone_names = ['NGC1084_GROUP_factor2.5_pixscale0.6', 'NGC1121_factor6.5_pixscale0.6', 'PGC000902_factor4.0_pixscale0.6',
                            'PGC039258_factor2.5_pixscale0.6', 'PGC1092512_factor2.5_pixscale0.6', 'PGC938075_factor4.5_pixscale0.6']

    index = -1
    for name in tqdm(names, leave=True):
        index += 1

        if name in list_undone_names:
            with open(f"{PATH_DATA}/{name}/dict_track.pkl", "rb") as f:
                dict_data = pickle.load(f)
            dict_data = extra_processing(name, dict_data)

            # This sets the progenitor in the middle of the stream
            dict_data['delta_theta'] = np.median(dict_data['theta'])
            dict_data['theta'] -= dict_data['delta_theta']
            dict_data['bin_width'] = np.diff(dict_data['theta']).min()

            n_particles = jnp.maximum(n_particles_min, n_particles_per_point * len(dict_data['theta'])).item()

            new_PATH_DATA = f'{PATH_DATA}/{name}/Plots_fixedProg_Sig_Transform_ndim{ndim}_Nparticles{n_particles}_Nmin{n_min}_VarRatio{var_ratio}_nlive{nlive}'
            if not os.path.exists(new_PATH_DATA):         
                os.makedirs(new_PATH_DATA, exist_ok=True)
                
                M_stellar = STRRINGS_catalogue.iloc[index]['M_stream']/STRRINGS_catalogue.iloc[index]['M_stream/M_host']
                M_halo = np.log10(halo_mass_from_stellar_mass(M_stellar))

                print(f'Fitting {name} with nlive={nlive} and fixed progenitor at center')
                dict_results = dynesty_fit(dict_data, logl, prior_transform, ndim, n_particles=n_particles, n_min=n_min, var_ratio=var_ratio, nlive=nlive)
                with open(f'{new_PATH_DATA}/dict_results.pkl', 'wb') as f:
                    pickle.dump(dict_results, f)

                # Plot and Save corner plot
                labels = ['logM', 'Rs', 'dirx', 'diry', 'dirz', 'logm', 'rs', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time', 'sig']
                figure = corner.corner(dict_results['samps'], 
                            labels=labels,
                            color='blue',
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, 
                            title_kwargs={"fontsize": 16},
                            truths=[M_halo, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                            truth_color='red',
                            )
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
                theta_stream, r_stream, xv_stream = params_to_stream(best_params, n_particles)
                r_bin, _, _ = jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(theta_stream, r_stream, dict_data['theta'], dict_data['bin_width'])
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
                if name == 'PGC938075_factor4.5_pixscale0.6':
                    pixel_to_kpc *= 100
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