import os
import numpy as np
import multiprocessing as mp

import dynesty
import dynesty.utils as dyut

def dynesty_fit(dict_data, logl_fn, prior_fn, ndim, n_particles=10000, n_min=101, var_ratio=9.0, min_err=0., nlive=2000):
    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(logl_fn,
                                prior_fn,
                                ndim,
                                logl_args=(dict_data, n_particles, n_min, var_ratio, min_err),
                                nlive=nlive,
                                sample='rslice',
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