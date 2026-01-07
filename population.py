import numpy as np
import emcee
import multiprocessing

import pickle
import numpy as np
from tqdm import tqdm

import os 
import pickle
import scipy
import corner 
import matplotlib.pyplot as plt

import multiprocessing as mp

import dynesty
import dynesty.utils as dyut

from orbit_fit import model_log_likelihood, orbit_prior_transform # import only for loading samples

def combine_data(path_save, N):
    dict_flat, dict_params = {}, {}
    all_true_params = []
    max_length = 0
    N_real = 0
    for i in tqdm( range(N), leave=True):

        try:
            with open(f'{path_save}/xx_{i+1:03d}/dict_result.pkl', 'rb') as file:
                dns = pickle.load(file)

            true_params = np.loadtxt(f'{path_save}/xx_{i+1:03d}/params_data.txt')
            all_true_params.append(true_params[2])

            posterior_samples = dns['samps'][:,2] # CAREFUL HERE - 1 is the index of the parameter we are interested in

            dict_params[i] = true_params

            dict_flat[i] = posterior_samples

            max_length = np.max([max_length, len(posterior_samples)])
            N_real += 1
        except:
            print(f'Failed for i={i}')

    samples_array = np.zeros([N_real, max_length])
    for i in range(N_real):
        try:
            samples_array[i, :len(dict_flat[i])] = dict_flat[i]
        except:
            print(f'Failed for i={i}')
    
    with open(f'{path_save}/dict_q_N{N_real}.pkl', 'wb') as f:
        pickle.dump(dict_flat, f)

    return dict_flat, N_real

def MCMC_fit(path_save, N, ndim=2, nwalkers=10, nsteps=1000):

    dict_data, N_real = combine_data(path_save, N)

    # Initial positions of the walkers
    p0 = 2*np.random.uniform(size=(nwalkers, ndim))

    # Set up the multiprocessing pool
    multiprocessing.set_start_method("spawn", force=True)
    with multiprocessing.Pool() as pool:
        # Set up the MCMC sampler with the pool for parallel processing
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(dict_data, ), pool=pool)
        
        # Run MCMC
        sampler.run_mcmc(p0, nsteps, progress=True)

    # Get the chain of samples after burn-in (e.g., first 1000 steps)
    samples = sampler.get_chain(discard=200, thin=15, flat=True)

    return samples, N_real

def gaussian(x, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(x-mu)**2/sigma**2)

# Define the log-posterior function
def log_posterior(theta, dict_data):
    mu, sigma = theta
    
    # Prior: assuming broad uniform priors
    if mu < 0:
        return -np.inf
    elif mu > 2:
        return -np.inf
    elif sigma < 0: 
        return -np.inf
    elif sigma > 2:
        return -np.inf
    
    else:
        # Log-prior (uniform priors, log of 1 is 0, so we can ignore it)
        log_prior = 0
        
        log_likelihood = 0
        for i in dict_data.keys():
        # likelihood      = np.sum( gaussian(dict_data, mu, sigma) * np.int8(dict_data!=0), axis=1 ) / NN
            likelihood       = np.mean( gaussian(dict_data[i], mu, sigma) )
            log_likelihood  += np.log(likelihood)

        return log_prior + log_likelihood

def prior_transform(p):
    #ndim = 12
    mu0, sigma0 = p

    mu1 = 2*mu0
    sigma1 = 2*sigma0

    return [mu1, sigma1]

def log_likelihood(theta, dict_data):
    mu, sigma = theta

    # Log-likelihood
    log_likelihood = 0
    for i in dict_data.keys():
        likelihood      = np.mean(gaussian(dict_data[i], mu, sigma))
        log_likelihood += np.log(likelihood)

    return log_likelihood

def dynesty_fit(dict_data, ndim=2, nlive=500):
    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood,
                                prior_transform,
                                ndim,
                                logl_args=(dict_data, ),
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

if __name__ == '__main__':
    # from jax_dynesty import *
    q_true = 1.0
    q_sig = 0.1
    seed = 3
    ndim = 14
    nlive = 4000
    sigma = 1

    PATH_SAVE = f'/data/dc824-2/S2S/q{q_true}_qsig{q_sig}_seed{seed}_ndim{ndim}_nlive{nlive}_sigma{sigma}'

    # i = 30
    # n = i+1
    # # MCMC fit
    # samples, N_real = MCMC_fit(PATH_SAVE, n, ndim=2, nwalkers=10, nsteps=1000)

    # for n in [25, 50, 100]:

    n = 75
    # # Dynesty fit
    dict_data, N_real = combine_data(PATH_SAVE, n)
    dict_result = dynesty_fit(dict_data)
    samples = dict_result['samps']

    np.save(PATH_SAVE+f'/population_samples_N{N_real}_dynesty.npy', samples)

    fig = corner.corner(samples, 
            color='blue',
            quantiles=[0.16, 0.5, 0.84], 
            show_titles=True, 
            title_kwargs={"fontsize": 16},
            truths=[q_true, q_sig], 
            truth_color='red',
            labels=[r"q$_{mean}$", r"q$_{sigma}$"])
    fig.savefig(f'{PATH_SAVE}/population_corner_N{N_real}_dynesty.pdf')

    with open(f'{PATH_SAVE}/dict_q_N{N_real}.pkl', 'rb') as file:
        dict_q= pickle.load(file)

    average_q = 0
    all_q = []
    for i in dict_q.keys():
        average_q += dict_q[i].mean()
        all_q.extend(dict_q[i])
    average_q /= len(dict_q)

    plt.figure()
    plt.hist(all_q, bins=30, color='blue', histtype='bar', ec='black')
    plt.axvline(average_q, color='red', lw=2, label='Mean', linestyle='--')
    plt.axvline(q_true, color='red', lw=2, label='True')
    plt.xlabel('q')
    plt.ylabel('Count')
    plt.legend(loc='best')
    plt.savefig(f'{PATH_SAVE}/q_hist_N{N_real}.pdf')

    