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
    fit_dist = 'gaussian'
    ndim = 2
    nlive = 500

    PATH_DATA = '/data/dc824-2/SGA_Streams/for_pop'
    STRINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'STRRINGS.xlsx')

    rm_names = ['NGC5387_factor2.5_pixscale0.6', 'PGC021008_factor2.5_pixscale0.6']

    # Load STRRINGS.xlsx and build two name lists: yes-only and yes+no
    df = pd.read_excel(STRINGS_PATH)
    names_yes  = df[df['Elisabeth'] == 'yes']['Name'].tolist()
    names_both = df['Name'].tolist()

    for label, names in [('yes', names_yes), ('both', names_both)]:
        q_fits = []
        for name in names:
            if name not in rm_names:
                path_dict = f'{PATH_DATA}/{name}.pkl'
                if os.path.exists(path_dict):
                    with open(path_dict, "rb") as f:
                        dict_results = pickle.load(f)
                    q_fits.append(get_q(*dict_results['samps'][:, 2:5].T))

        print(f'[{label}] Fitting population with {len(q_fits)} streams using a {fit_dist} distribution')
        dns_results = dynesty_fit(q_fits, ndim=ndim, nlive=nlive, pop_type=fit_dist)
        with open(os.path.join(PATH_DATA, f'dict_pop_{fit_dist}_nlive{nlive}_N{len(q_fits)}_{label}.pkl'), 'wb') as f:
            pickle.dump(dns_results, f)

        # Plots the corner plots
        figure = corner.corner(dns_results['samps'],
                color='blue',
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 16},
                truth_color='red')
        figure.savefig(os.path.join(PATH_DATA, f'corner_pop_{fit_dist}_nlive{nlive}_N{len(q_fits)}_{label}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
        plt.close(figure)