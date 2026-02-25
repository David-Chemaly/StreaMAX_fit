import os
import corner
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 18})
from scipy.stats import norm

import jax
import jax.numpy as jnp

import dynesty
import dynesty.utils as dyut

from utils import get_q

BAD_VAL = -1e15

### Uniform ###
def uniform(x, a, delta):
    prob = np.where((x >= a-delta) & (x <= a+delta), 1./delta, 0.)
    return prob

def prior_transform_uniform(p):
    a0, delta0 = p

    a1 = 2*a0
    delta1 = 2*delta0

    return [a1, delta1]

### Gaussian ###
def gaussian(x, mu, sigma):
    return 1/jnp.sqrt(2*jnp.pi*sigma**2) * jnp.exp(-0.5*(x-mu)**2/sigma**2)

def prior_transform_gaussian(p):
    mu0, sigma0 = p

    mu1    = 2*mu0
    sigma1 = 2*sigma0

    return [mu1, sigma1]

### Binomial ###
def binomial(x, mu1, mu2, sigma1, sigma2, prob=0.5):
    return prob * (1/np.sqrt(2*np.pi*sigma1**2) * np.exp(-0.5*(x-mu1)**2/sigma1**2)) + \
           (1-prob) * (1/np.sqrt(2*np.pi*sigma2**2) * np.exp(-0.5*(x-mu2)**2/sigma2**2))

def prior_transform_binomial(p):
    mu1, mu2, sigma1, sigma2, prob = p

    mu1    = mu1
    sigma1 = 2*sigma1
    mu2    = mu2+1
    sigma2 = 2*sigma2
    prob   = prob

    return [mu1, mu2, sigma1, sigma2, prob]

### Likelihood ###
def log_likelihood(theta, dict_data, pop_type='uniform'):
    if pop_type == 'uniform':
        pop_dist = uniform
    elif pop_type == 'gaussian':
        pop_dist = gaussian
    elif pop_type == 'binomial':
        pop_dist = binomial

    # Log-likelihood
    log_likelihood = 0
    for i in range(len(dict_data)):
        likelihood      = np.mean(pop_dist(dict_data[i], *theta))
        if likelihood <= 0:
            return BAD_VAL
        log_likelihood += np.log(likelihood)

    return log_likelihood

### Dynesty Fit ###
def dynesty_fit(dict_data, ndim=2, nlive=500, pop_type='uniform'):
    if pop_type == 'uniform':
        prior_transform = prior_transform_uniform
    elif pop_type == 'gaussian':
        prior_transform = prior_transform_gaussian
    elif pop_type == 'binomial':
        prior_transform = prior_transform_binomial

    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood,
                                prior_transform,
                                ndim,
                                logl_args=(dict_data, pop_type),
                                nlive=nlive,
                                sample='unif',  
                                pool=poo,
                                queue_size=nthreads * 2)
        dns.run_nested(n_effective=10000)

    res   = dns.results
    inds  = np.arange(len(res.samples))
    inds  = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl  = res.logl[inds]

    dns_results = {
                    'dns': dns,
                    'samps': samps,
                    'logl': logl,
                    'logz': res.logz,
                    'logzerr': res.logzerr,
                }

    return dns_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=['best', 'yes', 'both', '70'], default='best',
                        help='best = Elisabeth==best only; '
                             'yes  = best + yes; '
                             'both = all streams'
                             '70  = sigma ratio<=70; ')
    parser.add_argument('--ess-only', action='store_true',
                        help='skip the population fit and just produce the ESS plot')
    args = parser.parse_args()

    fit_dist = 'gaussian'
    ndim  = 2
    nlive = 500

    PATH_DATA = '/data/dc824-2/SGA_Streams/for_pop'
    STRINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'STRRINGS.xlsx')

    rm_names = ['NGC5387_factor2.5_pixscale0.6', 'PGC021008_factor2.5_pixscale0.6', 'PGC430221_factor4.0_pixscale0.6']

    df = pd.read_excel(STRINGS_PATH)
    if args.filter == 'best':
        names = df[df['Elisabeth'] == 'best']['Name'].tolist()
    elif args.filter == 'yes':
        names = df[df['Elisabeth'].isin(['best', 'yes'])]['Name'].tolist()
    elif args.filter == 'both':  # both
        names = df['Name'].tolist()
    elif args.filter == '70':
        names = df[df['sigma_ratio'] <= 70]['Name'].tolist()

    q_fits     = []
    names_used = []
    for name in names:
        if name not in rm_names:
            path_dict = f'{PATH_DATA}/{name}.pkl'
            if os.path.exists(path_dict):
                with open(path_dict, "rb") as f:
                    dict_results = pickle.load(f)
                q_fits.append(get_q(*dict_results['samps'][:, 2:5].T))
                names_used.append(name.split('_factor')[0])

    if not args.ess_only:
        print(f'[{args.filter}] Fitting population with {len(q_fits)} streams using a {fit_dist} distribution')
        dns_results = dynesty_fit(q_fits, ndim=ndim, nlive=nlive, pop_type=fit_dist)
        with open(os.path.join(PATH_DATA, f'dict_pop_{fit_dist}_nlive{nlive}_N{len(q_fits)}_{args.filter}.pkl'), 'wb') as f:
            pickle.dump(dns_results, f)

        figure = corner.corner(dns_results['samps'],
                labels=[r'$\mu$', r'$\sigma$'],
                color='blue',
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 16},
                truth_color='red')

        # Mark the spherical case (mu=1) with a black vertical line
        axes = np.array(figure.get_axes()).reshape(ndim, ndim)
        axes[0, 0].axvline(1., color='black', lw=1.5)  # 1D mu histogram
        axes[1, 0].axvline(1., color='black', lw=1.5)  # 2D contour panel

        figure.savefig(os.path.join(PATH_DATA, f'corner_pop_{fit_dist}_nlive{nlive}_N{len(q_fits)}_{args.filter}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
        plt.close(figure)

    # --- Effective sample size diagnostic ---
    # Kish ESS = (sum w)^2 / sum(w^2), evaluated at the 16th, 50th and 84th
    # percentile of each population parameter.
    # If --ess-only, try to load a previous fit; fall back to (mu=1, sigma=0.1).
    if args.ess_only:
        pop_pkl = os.path.join(PATH_DATA, f'dict_pop_{fit_dist}_nlive{nlive}_N{len(q_fits)}_{args.filter}.pkl')
        if os.path.exists(pop_pkl):
            with open(pop_pkl, 'rb') as f:
                dns_results = pickle.load(f)
            print(f'ESS evaluated using existing fit: {pop_pkl}')
        else:
            dns_results = None
            print('No existing fit found — ESS evaluated at default theta (mu=1, sigma=0.1)')

    if fit_dist == 'gaussian':
        pop_dist_fn = gaussian
    elif fit_dist == 'uniform':
        pop_dist_fn = uniform
    elif fit_dist == 'binomial':
        pop_dist_fn = binomial

    if dns_results is not None:
        thetas = np.percentile(dns_results['samps'], [16, 50, 84], axis=0)
    else:
        thetas = np.tile([1.0, 0.1], (3, 1))
    labels_pct = ['16th percentile', '50th percentile', '84th percentile']
    colors_pct = ['#4393c3', '#2166ac', '#053061']

    def kish_ess(q, theta):
        w = np.clip(np.array(pop_dist_fn(q, *theta)), 0, None)
        return (w.sum()**2) / (w**2).sum() if (w**2).sum() > 0 else 0.

    ess_matrix   = np.array([[kish_ess(q, theta) for q in q_fits] for theta in thetas])
    N_per_galaxy = np.array([len(q) for q in q_fits])
    pct_matrix   = ess_matrix / N_per_galaxy[np.newaxis, :] * 100  # ESS/N_i per galaxy

    n       = len(names_used)
    bar_h   = 0.25
    offsets = np.array([-bar_h, 0, bar_h])

    fig_ess, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, max(4, 0.45 * n)), sharey=True)

    for k, (label, color, offset) in enumerate(zip(labels_pct, colors_pct, offsets)):
        ypos = np.arange(n) + offset
        ax_l.barh(ypos, np.clip(ess_matrix[k], 1, None), height=bar_h, color=color, label=label)
        ax_r.barh(ypos, pct_matrix[k],                   height=bar_h, color=color)

    # Left: raw ESS, log scale
    ax_l.set_xscale('log')
    ax_l.axvline(1000, color='black', lw=1.2, ls='--', label='ESS = 1000')
    ax_l.set_xlabel('Effective sample size  (log scale)')
    ax_l.set_yticks(np.arange(n))
    ax_l.set_yticklabels(names_used, fontsize=10)
    ax_l.legend(fontsize=9)

    # Right: ESS / N_i per galaxy, linear scale
    ax_r.axvline(5, color='black', lw=1.2, ls='--', label='5%')
    ax_r.set_xlabel('ESS / N  (%,  per galaxy)')
    ax_r.legend(fontsize=9)

    fig_ess.suptitle(f'Importance sampling ESS per stream ({args.filter})', y=1.01)
    fig_ess.tight_layout()
    fig_ess.savefig(os.path.join(PATH_DATA, f'ess_{fit_dist}_nlive{nlive}_N{len(q_fits)}_{args.filter}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig_ess)