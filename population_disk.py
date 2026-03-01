import os
import corner
import pickle
import numpy as np
from tqdm import tqdm
from astropy.table import Table
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import linregress

from utils import *
from population import dynesty_fit

def nfw_mass_enclosed(r, M0, rs):
    """
    NFW enclosed mass with 2-parameter normalization:
      M0 = 4*pi*rho_s*rs^3   (Msun)
      rs = scale radius      (same length units as r)
    """
    r = np.asarray(r, dtype=float)
    x = r / rs
    return M0 * (np.log(1.0 + x) - x / (1.0 + x))

def mn_mass_enclosed_midplane(r, Md, a, b):
    """
    Miyamoto-Nagai 'equivalent enclosed mass' assuming z=0 (mid-plane), so r = R.
      Md: disk mass
      a : scale length
      b : scale height
    """
    r = np.asarray(r, dtype=float)
    c = a + b
    return Md * (r**3) / (r**2 + c**2)**1.5

def disk_to_halo_mass_ratio_r(r, Md, a, b, M0_nfw, rs_nfw):
    """
    Ratio M_disk(<r) / M_NFW(<r) using mid-plane convention for the disk.
    """
    Mdisk = mn_mass_enclosed_midplane(r, Md, a, b)
    Mhalo = nfw_mass_enclosed(r, M0_nfw, rs_nfw)
    return Mdisk / Mhalo

if __name__ == "__main__":
    ndim  = 14
    n_min = 3
    nlive = 2000
    alpha = 0.01
    var_ratio = 9.0
    n_particles_per_point = 2000
    n_particles_min = 10000

    PATH_DATA = f'/data/dc824-2/SGA_Streams'
    names = np.loadtxt(f'{PATH_DATA}/names.txt', dtype=str)
    STRRINGS_catalogue = pd.read_csv(f'{PATH_DATA}/STRRINGS_catalogue.csv')
    df = pd.read_excel(f'STRRINGS.xlsx')


    PATH_DATA = f'/data/dc824-2/MockStreamsDiskEdgeOn'
    N = 80

    MSE_list = []
    MAE_list = []
    area_list = []
    M_ratio_list = []
    mod_list = []
    median_list = []
    mean_list = []
    mean_r_list = []
    q_all = []
    M_enc_list =[]
    for seed in tqdm(range(N), leave=True):
        try:
            with open(f"{PATH_DATA}/seed{seed+1}/dict_stream.pkl", "rb") as f:
                dict_data = pickle.load(f)
            with open(f"{PATH_DATA}/seed{seed+1}/dict_results.pkl", "rb") as f:
                dict_samps = pickle.load(f)

            q_samps = get_q(*dict_samps['samps'][:, 2:5].T)
            q_all.append(q_samps)
            M_ratio = 100 * (10**dict_data['params_disk'][0]/ 10**dict_data['params'][0])
            M_enc = disk_to_halo_mass_ratio_r(dict_data['r'].mean(),
                                            10**dict_data['params_disk'][0], dict_data['params_disk'][1], dict_data['params_disk'][2], 10**dict_data['params'][0], dict_data['params'][1])
            M_enc_list.append(M_enc)
            MSE = np.mean((q_samps-1)**2)
            MAE = np.mean(np.abs(q_samps-1))
            count, edges = np.histogram(q_samps, bins=30, range=(0.5, 1.5))
            area = 100*(count[14]+count[15])/np.sum(count)
            mod = np.argmax(count) / 30 * (1.5-0.5) + 0.5
            median = np.median(q_samps)
            mean = np.mean(q_samps)

            MSE_list.append(MSE)
            MAE_list.append(MAE)
            area_list.append(area)
            M_ratio_list.append(M_ratio)
            mod_list.append(mod)
            median_list.append(median)
            mean_list.append(mean)
            mean_r_list.append(dict_data['r'].mean())

        except:
            print(f"Error processing seed {seed+1}. Skipping.")
    MSE_list = np.array(MSE_list)
    MAE_list = np.array(MAE_list)
    area_list = np.array(area_list)
    M_ratio_list = np.array(M_ratio_list)
    mod_list = np.array(mod_list)
    median_list = np.array(median_list)
    mean_list = np.array(mean_list)
    mean_r_list = np.array(mean_r_list)
    M_enc_list = np.array(M_enc_list)

    q_all_big = {}

    q_fits = []
    for i in np.argsort(M_ratio_list):
        q_fits.append(q_all[i])
    q_all_big[0] = q_fits

    # q_fits = []
    # for i in np.argsort(M_ratio_list)[25:50]:
    #     q_fits.append(q_all[i])
    # q_all_big[1] = q_fits

    # q_fits = []
    # for i in np.argsort(M_ratio_list)[50:]:
    #     q_fits.append(q_all[i])
    # q_all_big[2] = q_fits

    nlive = 500
    ndim  = 2

    for i in range(len(q_all_big)):
        print(f'[{i}] Fitting population with {len(q_all_big[i])} streams using a gaussian distribution')
        dns_results = dynesty_fit(q_all_big[i], ndim=ndim, nlive=nlive, pop_type='gaussian')
        with open(os.path.join(PATH_DATA, f'dict_pop_disk_nlive{nlive}_N{len(q_all_big[i])}_{i}.pkl'), 'wb') as f:
            pickle.dump(dns_results, f)

        figure = corner.corner(dns_results['samps'],
                labels=[r'$\mu_q$', r'$\sigma_q$'],
                color='blue',
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 16},
                truth_color='red')

        # Mark the spherical case (mu=1) with a black vertical line
        axes = np.array(figure.get_axes()).reshape(ndim, ndim)
        axes[0, 0].axvline(1., color='red', linestyle='--', lw=1.5)  # 1D mu histogram
        axes[1, 0].axvline(1., color='red', linestyle='--', lw=1.5)  # 2D contour panel

        figure.savefig(os.path.join(PATH_DATA, f'corner_pop_nlive{nlive}_N{len(q_all_big[i])}_{i}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
        plt.close(figure)