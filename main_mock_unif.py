"""Axisymmetric NFW projected-stream mock with scale-free priors.

This is the stream-track analogue of the StreaMASS orbit test.  The mock and
the fit use the same axisymmetric NFW model.  To make the spatial track
invariant under the mass scaling M->lambda*M, velocity and time are sampled
through

    log_alpha  = log10(|v| / v_circ)
    log_tau    = log10(t * v_circ / r0)
    log_mfrac  = log10(m_sat / M_host)

instead of sampling vx, vy, vz, t, and the satellite mass in absolute units.
The host and satellite scale radii are sampled as logRs and logrs.  With
track-only data, logM should marginalise to its uniform prior; adding a
line-of-sight velocity datum breaks the degeneracy and pins logM.
"""

import argparse
import json
import os
import pickle
from functools import partial
from pathlib import Path

import dynesty
import dynesty.utils as dyut
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import StreaMAX
from tqdm import tqdm

matplotlib.use("Agg")
import corner
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 14})

BAD_VAL = -1e15
EPSILON = 1e-6
DEFAULT_OUTPUT_ROOT = Path("/data/dc824-2/StreamUnif")
RS_MIN_KPC = 5.0
RS_MAX_KPC = 30.0
RS_SAT_MIN_KPC = 0.5
RS_SAT_MAX_KPC = 3.0
LOG_RS_MIN = np.log10(RS_MIN_KPC)
LOG_RS_MAX = np.log10(RS_MAX_KPC)
LOG_RS_SAT_MIN = np.log10(RS_SAT_MIN_KPC)
LOG_RS_SAT_MAX = np.log10(RS_SAT_MAX_KPC)
LOG_ALPHA_MIN = -1.0
LOG_ALPHA_MAX = 1.0
LOG_TAU_MIN = 0.0
LOG_TAU_MAX = 2.0
LOG_MFRAC_MIN = -7.0
LOG_MFRAC_MAX = -1.0
Z0_MIN_KPC = 1e-3
Z0_MAX_KPC = 50.0
LIKELIHOOD_MODES = ("track", "track_los")

LABELS = (
    "logM",
    "logRs",
    "q",
    "theta_q",
    "phi_q",
    "log_mfrac",
    "logrs",
    "x0",
    "z0",
    "theta_v",
    "phi_v",
    "log_alpha",
    "log_tau",
    "sig",
)
NDIM = len(LABELS)

DEFAULT_FREE_PARAMS = (
    "logM",
    "logRs",
    "q",
    "log_mfrac",
    "logrs",
    "log_alpha",
    "log_tau",
)
DEFAULT_FIXED_PARAMS = tuple(
    label for label in LABELS if label not in DEFAULT_FREE_PARAMS
)
FIXABLE_PARAMS = DEFAULT_FIXED_PARAMS


def effective_stream_particle_count(n_particles, n_steps):
    release_count = n_steps + 1
    block = int(np.lcm(2, release_count))
    return int(np.ceil(int(n_particles) / block) * block)


def prior_transform_unif(u):
    (
        u_logM,
        u_logRs,
        u_q,
        u_theta_q,
        u_phi_q,
        u_mfrac,
        u_logrs,
        u_x0,
        u_z0,
        u_theta_v,
        u_phi_v,
        u_alpha,
        u_tau,
        u_sig,
    ) = u

    logM = 10.0 + 4.0 * u_logM
    logRs = LOG_RS_MIN + (LOG_RS_MAX - LOG_RS_MIN) * u_logRs
    q = 0.5 + u_q
    theta_q = jnp.arcsin(2.0 * u_theta_q - 1.0)
    phi_q = 2.0 * jnp.pi * u_phi_q
    log_mfrac = LOG_MFRAC_MIN + (LOG_MFRAC_MAX - LOG_MFRAC_MIN) * u_mfrac
    logrs = LOG_RS_SAT_MIN + (LOG_RS_SAT_MAX - LOG_RS_SAT_MIN) * u_logrs
    x0 = 10.0 + 70.0 * u_x0
    z0 = Z0_MIN_KPC + (Z0_MAX_KPC - Z0_MIN_KPC) * u_z0
    theta_v = jnp.arcsin(2.0 * u_theta_v - 1.0)
    phi_v = 2.0 * jnp.pi * u_phi_v

    log_alpha = LOG_ALPHA_MIN + (LOG_ALPHA_MAX - LOG_ALPHA_MIN) * u_alpha
    log_tau = LOG_TAU_MIN + (LOG_TAU_MAX - LOG_TAU_MIN) * u_tau
    sig = 10.0 * u_sig

    return jnp.asarray(
        [
            logM,
            logRs,
            q,
            theta_q,
            phi_q,
            log_mfrac,
            logrs,
            x0,
            z0,
            theta_v,
            phi_v,
            log_alpha,
            log_tau,
            sig,
        ],
        dtype=jnp.float32,
    )


def stream_free_to_physical(params):
    params = np.asarray(params, dtype=float)
    (
        logM,
        logRs,
        q,
        theta_q,
        phi_q,
        log_mfrac,
        logrs,
        x0,
        z0,
        theta_v,
        phi_v,
        log_alpha,
        log_tau,
        sig,
    ) = params
    Rs = 10.0**logRs
    rs = 10.0**logrs

    dirx = np.cos(theta_q) * np.cos(phi_q)
    diry = np.cos(theta_q) * np.sin(phi_q)
    dirz = np.sin(theta_q)

    logm = logM + log_mfrac

    v_circ = oriented_nfw_circular_speed(logM, Rs, x0, 0.0, z0, q, dirx, diry, dirz)
    r0 = np.sqrt(x0**2 + z0**2)
    speed = (10.0**log_alpha) * v_circ
    time = (10.0**log_tau) * r0 / max(v_circ, EPSILON)

    vx0 = speed * np.cos(theta_v) * np.cos(phi_v)
    vy0 = speed * np.cos(theta_v) * np.sin(phi_v)
    vz0 = speed * np.sin(theta_v)

    return {
        "logM": logM,
        "logRs": logRs,
        "Rs": Rs,
        "q": q,
        "theta_q": theta_q,
        "phi_q": phi_q,
        "dirx": dirx,
        "diry": diry,
        "dirz": dirz,
        "log_mfrac": log_mfrac,
        "logm": logm,
        "logrs": logrs,
        "rs": rs,
        "x0": x0,
        "y0": 0.0,
        "z0": z0,
        "vx0": vx0,
        "vy0": vy0,
        "vz0": vz0,
        "time": time,
        "theta_v": theta_v,
        "phi_v": phi_v,
        "log_alpha": log_alpha,
        "log_tau": log_tau,
        "sig": sig,
        "v_circ": v_circ,
        "speed": speed,
    }


def oriented_nfw_circular_speed(logM, Rs, x0, y0, z0, q, dirx, diry, dirz):
    params_host = StreaMAX.prepare_params(
        {
            "logM": logM,
            "Rs": Rs,
            "a": 1.0,
            "b": 1.0,
            "c": q,
            "dirx": dirx,
            "diry": diry,
            "dirz": dirz,
            "x_origin": 0.0,
            "y_origin": 0.0,
            "z_origin": 0.0,
        }
    )
    acc = np.asarray(StreaMAX.NFW_acceleration(x0, y0, z0, params_host))
    radius = np.hypot(x0, y0)
    if radius < EPSILON:
        return EPSILON

    e_radius = np.asarray([x0 / radius, y0 / radius, 0.0])
    a_radius = float(np.dot(acc, e_radius))
    return float(np.sqrt(max(-radius * a_radius, EPSILON)))


def params_to_stream_unif(params, n_particles=10000, n_steps=99, alpha=1.0, unroll=False):
    phys = stream_free_to_physical(params)
    n_particles = effective_stream_particle_count(n_particles, n_steps)

    params_host = {
        "logM": phys["logM"],
        "Rs": phys["Rs"],
        "a": 1.0,
        "b": 1.0,
        "c": phys["q"],
        "dirx": phys["dirx"],
        "diry": phys["diry"],
        "dirz": phys["dirz"],
        "x_origin": 0.0,
        "y_origin": 0.0,
        "z_origin": 0.0,
    }
    params_sat = {
        "logM": phys["logm"],
        "Rs": phys["rs"],
        "x_origin": phys["x0"],
        "y_origin": 0.0,
        "z_origin": phys["z0"],
    }
    xv_f = jnp.asarray(
        [phys["x0"], 0.0, phys["z0"], phys["vx0"], phys["vy0"], phys["vz0"]],
        dtype=jnp.float32,
    )

    _, xv_sat, xv_stream, xhi_stream = StreaMAX.generate_stream(
        xv_f,
        "NFW",
        params_host,
        "Plummer",
        params_sat,
        phys["time"],
        alpha,
        n_steps,
        n_particles,
        unroll,
    )
    _, _, theta_stream, r_stream, _ = StreaMAX.get_stream_ordered(
        xv_stream[:, 0],
        xv_stream[:, 1],
        xhi_stream,
    )
    return theta_stream, r_stream, xv_stream, xv_sat


def get_projected_track(theta_stream, r_stream, theta_bins, bin_width):
    return jax.vmap(StreaMAX.get_track_2D, in_axes=(None, None, 0, None))(
        theta_stream,
        r_stream,
        theta_bins,
        bin_width,
    )


def make_mock_data_stream(
    params,
    seed,
    sigma_pct=2.0,
    vz_err=20.0,
    vz_window_factor=1.0,
    n_particles=10000,
    n_track_bins=36,
    min_count=100,
    min_bins=10,
    r_max=80.0,
    max_bin_width=20.0,
):
    rng = np.random.default_rng(int(seed))
    theta_stream, r_stream, xv_stream, xv_sat = params_to_stream_unif(
        params,
        n_particles=n_particles,
    )

    theta_bins = jnp.linspace(-2.0 * jnp.pi, 2.0 * jnp.pi, n_track_bins + 1)
    bin_width = theta_bins[1] - theta_bins[0]
    r_bin, w_bin, count = get_projected_track(theta_stream, r_stream, theta_bins, bin_width)

    arg_take = np.asarray(jax.device_get(jnp.where(count > min_count)[0]))
    if len(arg_take) < min_bins:
        return None
    if not np.all(np.diff(arg_take) == 1):
        return None

    r_bin_np = np.asarray(jax.device_get(r_bin))
    w_bin_np = np.asarray(jax.device_get(w_bin))
    count_np = np.asarray(jax.device_get(count))
    if np.max(r_bin_np[arg_take]) > r_max:
        return None
    if not np.all(w_bin_np[arg_take] < max_bin_width):
        return None

    theta_sat = np.unwrap(
        np.arctan2(
            np.asarray(jax.device_get(xv_sat[:, 1])),
            np.asarray(jax.device_get(xv_sat[:, 0])),
        )
    )
    if not np.all(np.diff(theta_sat) > 0):
        return None

    theta_in = np.asarray(theta_bins)[arg_take]
    r_true = r_bin_np[arg_take]
    r_err = r_true * sigma_pct / 100.0
    r_obs = rng.normal(r_true, r_err)

    vz_window = float(bin_width) * vz_window_factor
    mask_v = (theta_stream >= -0.5 * vz_window) & (theta_stream <= 0.5 * vz_window)
    count_v = int(jax.device_get(jnp.sum(mask_v)))
    if count_v < min_count:
        return None
    vz_true = float(jax.device_get(jnp.sum(xv_stream[:, 5] * mask_v) / count_v))
    vz_obs = float(rng.normal(vz_true, vz_err))

    return {
        "seed": int(seed),
        "n_particles": int(n_particles),
        "n_particles_effective": effective_stream_particle_count(n_particles, 99),
        "n_track_bins": int(n_track_bins),
        "min_count": int(min_count),
        "sigma_pct": float(sigma_pct),
        "theta": jnp.asarray(theta_in, dtype=jnp.float32),
        "bin_width": float(bin_width),
        "r": jnp.asarray(r_obs, dtype=jnp.float32),
        "r_true": jnp.asarray(r_true, dtype=jnp.float32),
        "r_err": jnp.asarray(r_err, dtype=jnp.float32),
        "vz": vz_obs,
        "vz_true": vz_true,
        "vz_err": float(vz_err),
        "vz_theta": 0.0,
        "vz_window": vz_window,
        "vz_count": count_v,
        "w": w_bin_np[arg_take],
        "count": count_np[arg_take],
        "params": np.asarray(params, dtype=float),
        "physical_params": stream_free_to_physical(params),
        "theta_stream": np.asarray(jax.device_get(theta_stream)),
        "r_stream": np.asarray(jax.device_get(r_stream)),
        "x_stream": np.asarray(jax.device_get(xv_stream[:, 0])),
        "y_stream": np.asarray(jax.device_get(xv_stream[:, 1])),
        "z_stream": np.asarray(jax.device_get(xv_stream[:, 2])),
        "x_sat": np.asarray(jax.device_get(xv_sat[:, 0])),
        "y_sat": np.asarray(jax.device_get(xv_sat[:, 1])),
        "z_sat": np.asarray(jax.device_get(xv_sat[:, 2])),
    }


def find_mock_data_stream(seed, args):
    rng = np.random.default_rng(int(seed))
    for trial in range(args.truth_trials):
        params = np.array(prior_transform_unif(rng.uniform(0.0, 1.0, size=NDIM)), copy=True)
        params[-1] = 0.0
        dict_data = make_mock_data_stream(
            params,
            seed=seed,
            sigma_pct=args.sigma_pct,
            vz_err=args.vz_err,
            vz_window_factor=args.vz_window_factor,
            n_particles=args.n_particles,
            n_track_bins=args.n_track_bins,
            min_count=args.min_count,
            min_bins=args.min_bins,
            r_max=args.r_max,
            max_bin_width=args.max_bin_width,
        )
        if dict_data is not None:
            dict_data["truth_trials"] = trial + 1
            return dict_data

    raise RuntimeError(f"Could not find a usable projected stream for seed={seed}.")


def _track_loglike(
    params,
    dict_data,
    n_particles=10000,
    n_min=3,
    var_ratio_thresh=9.0,
):
    theta_stream, r_stream, xv_stream, _ = params_to_stream_unif(params, n_particles=n_particles)
    r_bin, sig_bin, count_bin = get_projected_track(
        theta_stream,
        r_stream,
        dict_data["theta"],
        dict_data["bin_width"],
    )

    n_bad = int(jax.device_get(jnp.sum(count_bin < n_min)))
    if n_bad > 0:
        return BAD_VAL * n_bad, theta_stream, xv_stream

    var_data = dict_data["r_err"] ** 2
    var_model = sig_bin**2 / count_bin
    n_high_error = int(
        jax.device_get(jnp.sum(var_model / var_data > 1.0 / var_ratio_thresh))
    )
    if n_high_error > 0:
        return BAD_VAL / 1e3 * n_high_error, theta_stream, xv_stream

    var = var_data + params[-1] ** 2
    log_like = -0.5 * jnp.sum(
        (r_bin - dict_data["r"]) ** 2 / var + jnp.log(2.0 * jnp.pi * var)
    )
    return log_like, theta_stream, xv_stream


def logl_unif(
    params,
    dict_data,
    n_particles=10000,
    n_min=3,
    var_ratio_thresh=9.0,
):
    log_like, _, _ = _track_loglike(params, dict_data, n_particles, n_min, var_ratio_thresh)
    return jnp.nan_to_num(log_like, nan=BAD_VAL, neginf=BAD_VAL, posinf=BAD_VAL)


def logl_unif_los(
    params,
    dict_data,
    n_particles=10000,
    n_min=3,
    var_ratio_thresh=9.0,
):
    log_like, theta_stream, xv_stream = _track_loglike(
        params,
        dict_data,
        n_particles,
        n_min,
        var_ratio_thresh,
    )
    if log_like <= BAD_VAL / 1e4:
        return log_like

    theta_center = dict_data["vz_theta"]
    theta_half = dict_data["vz_window"] / 2.0
    mask_v = (theta_stream >= theta_center - theta_half) & (theta_stream <= theta_center + theta_half)
    count_v = jnp.sum(mask_v)
    n_bad_v = int(jax.device_get(count_v < n_min))
    if n_bad_v > 0:
        return BAD_VAL * n_bad_v

    vz_mean = jnp.sum(xv_stream[:, 5] * mask_v) / count_v
    vz_var = dict_data["vz_err"] ** 2
    log_like += -0.5 * (
        (dict_data["vz"] - vz_mean) ** 2 / vz_var + jnp.log(2.0 * jnp.pi * vz_var)
    )
    return jnp.nan_to_num(log_like, nan=BAD_VAL, neginf=BAD_VAL, posinf=BAD_VAL)


def _expand_params(params_reduced, truth, free_indices, fixed_indices):
    params_full = np.zeros(NDIM, dtype=np.float64)
    params_full[free_indices] = np.asarray(params_reduced)
    params_full[fixed_indices] = np.asarray(truth)[fixed_indices]
    return params_full


def _reduced_prior_transform(u_reduced, truth, free_indices, fixed_indices):
    u_full = np.full(NDIM, 0.5)
    u_full[free_indices] = np.asarray(u_reduced)
    params = np.asarray(prior_transform_unif(u_full), dtype=np.float64)
    params[fixed_indices] = np.asarray(truth)[fixed_indices]
    return params[free_indices]


def _reduced_logl_track(params_reduced, dict_data, n_particles, n_min, var_ratio,
                        truth, free_indices, fixed_indices):
    params_full = _expand_params(params_reduced, truth, free_indices, fixed_indices)
    return logl_unif(params_full, dict_data, n_particles, n_min, var_ratio)


def _reduced_logl_track_los(params_reduced, dict_data, n_particles, n_min, var_ratio,
                            truth, free_indices, fixed_indices):
    params_full = _expand_params(params_reduced, truth, free_indices, fixed_indices)
    return logl_unif_los(params_full, dict_data, n_particles, n_min, var_ratio)


def resolve_fixed_indices(fixed_names):
    fixed_indices = np.array([LABELS.index(n) for n in fixed_names], dtype=np.int64)
    free_indices = np.array([i for i in range(NDIM) if i not in fixed_indices], dtype=np.int64)
    return free_indices, fixed_indices


def dynesty_fit_unif(
    dict_data,
    logl_fn,
    n_particles=10000,
    n_min=3,
    var_ratio=1e-10,
    nlive=2000,
    n_effective=10000,
    dlogz_init=0.01,
    nthreads=None,
    fixed_names=(),
    checkpoint_path=None,
    checkpoint_every=0,
):
    nthreads = os.cpu_count() if nthreads is None else int(nthreads)

    truth = np.asarray(dict_data["params"], dtype=np.float64)
    free_indices, fixed_indices = resolve_fixed_indices(fixed_names)
    ndim_fit = int(len(free_indices))

    if len(fixed_indices) == 0:
        logl_for_sampler = logl_fn
        prior_for_sampler = prior_transform_unif
        logl_args = (dict_data, n_particles, n_min, var_ratio)
    else:
        if logl_fn is logl_unif:
            reduced_logl_impl = _reduced_logl_track
        elif logl_fn is logl_unif_los:
            reduced_logl_impl = _reduced_logl_track_los
        else:
            raise ValueError(f"Unsupported logl_fn for reduced fit: {logl_fn}")
        logl_for_sampler = partial(
            reduced_logl_impl,
            truth=truth,
            free_indices=free_indices,
            fixed_indices=fixed_indices,
        )
        prior_for_sampler = partial(
            _reduced_prior_transform,
            truth=truth,
            free_indices=free_indices,
            fixed_indices=fixed_indices,
        )
        logl_args = (dict_data, n_particles, n_min, var_ratio)

    run_kwargs = {"n_effective": n_effective, "dlogz_init": dlogz_init}
    if checkpoint_path is not None and checkpoint_every and checkpoint_every > 0:
        run_kwargs["checkpoint_file"] = str(checkpoint_path)
        run_kwargs["checkpoint_every"] = float(checkpoint_every)

    if nthreads > 1:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        with mp.Pool(nthreads) as pool:
            sampler = dynesty.DynamicNestedSampler(
                logl_for_sampler,
                prior_for_sampler,
                ndim_fit,
                logl_args=logl_args,
                nlive=nlive,
                sample="rslice",
                pool=pool,
                queue_size=2 * nthreads,
            )
            sampler.run_nested(**run_kwargs)
    else:
        sampler = dynesty.DynamicNestedSampler(
            logl_for_sampler,
            prior_for_sampler,
            ndim_fit,
            logl_args=logl_args,
            nlive=nlive,
            sample="rslice",
        )
        sampler.run_nested(**run_kwargs)

    res = sampler.results
    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))

    raw_samps = res.samples[inds]
    if len(fixed_indices) == 0:
        full_samps = raw_samps
    else:
        full_samps = np.zeros((len(raw_samps), NDIM), dtype=np.float64)
        full_samps[:, free_indices] = raw_samps
        full_samps[:, fixed_indices] = truth[fixed_indices]

    return {
        "dns": sampler,
        "samps": full_samps,
        "logl": res.logl[inds],
        "logz": res.logz,
        "logzerr": res.logzerr,
        "nthreads": nthreads,
        "fixed_names": tuple(fixed_names),
        "free_indices": tuple(int(i) for i in free_indices),
        "fixed_indices": tuple(int(i) for i in fixed_indices),
        "ndim_fit": ndim_fit,
    }


def plot_mock_data_stream(path, dict_stream):
    theta = np.asarray(dict_stream["theta"])
    r_obs = np.asarray(dict_stream["r"])
    r_err = np.asarray(dict_stream["r_err"])
    r_true = np.asarray(dict_stream["r_true"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(
        dict_stream["x_stream"],
        dict_stream["y_stream"],
        c="0.65",
        s=1,
        alpha=0.35,
        label="truth stream",
    )
    ax.errorbar(
        r_obs * np.cos(theta),
        r_obs * np.sin(theta),
        fmt="o",
        color="tab:red",
        ms=4,
        label="mock track",
    )
    ax.plot(dict_stream["x_sat"], dict_stream["y_sat"], color="tab:blue", lw=1.0, label="satellite")
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    ax.set_aspect("equal")
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(theta, r_true, "-", color="black", lw=2, label="truth track")
    ax.errorbar(theta, r_obs, yerr=r_err, fmt="o", color="tab:red", ms=4, label="mock data")
    ax.set_xlabel("Angle (rad)")
    ax.set_ylabel("Radius (kpc)")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path / "stream.pdf")
    plt.close(fig)


def plot_best_fit(path, dict_data, best_params, n_particles):
    theta_best, r_best, xv_best, _ = params_to_stream_unif(best_params, n_particles=n_particles)
    r_best_bin, _, _ = get_projected_track(
        theta_best,
        r_best,
        dict_data["theta"],
        dict_data["bin_width"],
    )

    truth_params = dict_data["params"]
    theta_truth, r_truth, xv_truth, _ = params_to_stream_unif(truth_params, n_particles=n_particles)
    r_truth_bin, _, _ = get_projected_track(
        theta_truth,
        r_truth,
        dict_data["theta"],
        dict_data["bin_width"],
    )

    xv_best = np.asarray(jax.device_get(xv_best))
    xv_truth = np.asarray(jax.device_get(xv_truth))
    r_best_bin = np.asarray(jax.device_get(r_best_bin))
    r_truth_bin = np.asarray(jax.device_get(r_truth_bin))
    theta = np.asarray(dict_data["theta"])
    r_obs = np.asarray(dict_data["r"])
    r_err = np.asarray(dict_data["r_err"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    projections = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]

    for ax, (i, j, xlabel, ylabel) in zip(axes.flat[:3], projections, strict=True):
        ax.scatter(xv_truth[:, i], xv_truth[:, j], s=1, color="black", alpha=0.25, label="truth")
        ax.scatter(xv_best[:, i], xv_best[:, j], s=1, color="tab:blue", alpha=0.25, label="best fit")
        if i == 0 and j == 1:
            ax.errorbar(
                r_obs * np.cos(theta),
                r_obs * np.sin(theta),
                fmt="o",
                color="tab:red",
                ms=4,
                label="mock track",
            )
        ax.set_xlabel(f"{xlabel} (kpc)")
        ax.set_ylabel(f"{ylabel} (kpc)")
        ax.set_aspect("equal")

    ax = axes.flat[3]
    ax.errorbar(theta, r_obs, yerr=r_err, fmt="o", color="tab:red", ms=4, label="mock data")
    ax.plot(theta, r_truth_bin, color="black", lw=2, label="truth")
    ax.plot(theta, r_best_bin, color="tab:blue", lw=2, label="best fit")
    ax.set_xlabel("Angle (rad)")
    ax.set_ylabel("Radius (kpc)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles_r, labels_r = ax.get_legend_handles_labels()
    fig.legend(handles + handles_r, labels + labels_r, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path / "best_fit.pdf")
    plt.close(fig)


def save_corner_plots(path, dict_data, dict_results):
    samples = dict_results["samps"]
    truths = dict_data["params"]
    fixed_indices = set(int(i) for i in dict_results.get("fixed_indices") or ())

    free_cols = [i for i in range(NDIM) if i not in fixed_indices]
    if len(free_cols) >= 2:
        figure = corner.corner(
            samples[:, free_cols],
            labels=[LABELS[i] for i in free_cols],
            truths=[truths[i] for i in free_cols],
            truth_color="red",
            color="blue",
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 9},
        )
        figure.savefig(path / "corner_plot.pdf")
        plt.close(figure)

    derived = derived_sample_columns(samples)
    truth_phys = stream_free_to_physical(dict_data["params"])

    subset_indices = [LABELS.index(label) for label in DEFAULT_FREE_PARAMS]
    subset_cols = []
    subset_labels = []
    subset_truth = []
    for idx in subset_indices:
        if idx in fixed_indices:
            continue
        subset_cols.append(samples[:, idx])
        subset_labels.append(LABELS[idx])
        subset_truth.append(truths[idx])
    subset_cols += [derived[:, 0], derived[:, 1], derived[:, 4]]
    subset_labels += ["|v|", "time", "vz0"]
    subset_truth += [truth_phys["speed"], truth_phys["time"], truth_phys["vz0"]]

    subset = np.column_stack(subset_cols)
    if subset.shape[1] >= 2:
        figure = corner.corner(
            subset,
            labels=subset_labels,
            truths=subset_truth,
            truth_color="red",
            color="blue",
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 10},
        )
        figure.savefig(path / "corner_mass_velocity_time.pdf")
        plt.close(figure)


def save_q_posterior(path, dict_data, dict_results):
    q_samps = dict_results["samps"][:, 2]
    q_true = dict_data["params"][2]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(q_samps, bins=30, density=True, alpha=0.7, color="blue", range=(0.5, 1.5))
    ax.axvline(np.median(q_samps), color="blue", linestyle="--", lw=2)
    ax.axvline(np.percentile(q_samps, 16), color="blue", linestyle=":", lw=2)
    ax.axvline(np.percentile(q_samps, 84), color="blue", linestyle=":", lw=2)
    ax.axvline(q_true, color="red", linestyle="-", lw=2)
    ax.set_xlim(0.5, 1.5)
    ax.set_xlabel("Halo Flattening")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(path / "q_posterior.pdf")
    plt.close(fig)


def save_mass_posterior(path, dict_data, dict_results):
    logm_samps = dict_results["samps"][:, 0]
    logm_true = dict_data["params"][0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(logm_samps, bins=30, density=True, alpha=0.7, color="blue", range=(10.0, 14.0))
    ax.axhline(1.0 / 4.0, color="0.4", linestyle="--", lw=1.5, label="uniform prior")
    ax.axvline(logm_true, color="red", linestyle="-", lw=2, label="truth")
    ax.set_xlim(10.0, 14.0)
    ax.set_xlabel("logM")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path / "mass_posterior.pdf")
    plt.close(fig)


def derived_sample_columns(samples):
    derived = []
    for sample in samples:
        phys = stream_free_to_physical(sample)
        derived.append([phys["speed"], phys["time"], phys["vx0"], phys["vy0"], phys["vz0"]])
    return np.asarray(derived)


def save_diagnostics(path, dict_data, dict_results, n_effective_target):
    """Post-fit diagnostics to flag under-sampling or a loose evidence cut."""
    dns = dict_results.get("dns")
    if dns is None:
        return
    res = dns.results

    samps_raw = np.asarray(res.samples)
    logl_raw = np.asarray(res.logl)
    logwt = np.asarray(res.logwt)
    logz = np.asarray(res.logz)
    logzerr = np.asarray(res.logzerr)

    w = np.exp(logwt - logz[-1])
    w_norm = w / np.sum(w)
    ess_weighted = float(1.0 / np.sum(w_norm ** 2))

    samps_eq = np.asarray(dict_results["samps"])
    logM_eq = samps_eq[:, 0]

    bad_thresh = BAD_VAL / 1e3
    bad_frac = float(np.mean(logl_raw < bad_thresh))

    # logM is LABELS[0]; if it was fixed, it isn't in samps_raw. free_indices
    # is sorted, so logM sits at raw column 0 whenever it is free.
    free_indices = dict_results.get("free_indices")
    logM_is_free = (free_indices is None) or (0 in free_indices)
    unique_logM_raw = (
        int(len(np.unique(np.round(samps_raw[:, 0], 4)))) if logM_is_free else None
    )
    unique_logM_eq = int(len(np.unique(np.round(logM_eq, 4))))

    rng = np.random.default_rng(0)
    idx = np.arange(len(logM_eq))
    rng.shuffle(idx)
    half = len(idx) // 2
    a, b = logM_eq[idx[:half]], logM_eq[idx[half : 2 * half]]
    median_a, median_b = float(np.median(a)), float(np.median(b))
    std_a, std_b = float(np.std(a)), float(np.std(b))

    # Pass/warn heuristics.
    ess_ok = ess_weighted >= 0.8 * n_effective_target
    logzerr_ok = float(logzerr[-1]) < 0.5
    diversity_ok = (unique_logM_eq / max(len(logM_eq), 1)) > 0.25
    split_ok = abs(median_a - median_b) < 0.1 * max(std_a, std_b, 1e-3)

    def mark(ok):
        return "PASS" if ok else "WARN"

    with open(path / "diagnostics.txt", "w", encoding="ascii") as f:
        f.write("=== Convergence diagnostics ===\n")
        f.write(f"n_iter               = {int(res.niter)}\n")
        f.write(f"n_samples_raw        = {len(samps_raw)}\n")
        f.write(f"n_samples_equal      = {len(samps_eq)}\n")
        f.write(
            f"ess_weighted         = {ess_weighted:.1f}  "
            f"(target n_effective={n_effective_target})  [{mark(ess_ok)}]\n"
        )
        f.write(
            f"logz                 = {float(logz[-1]):+.4f} +/- "
            f"{float(logzerr[-1]):.4f}  [{mark(logzerr_ok)}]\n"
        )
        f.write(f"bad_val_fraction     = {bad_frac:.4f}\n")
        if unique_logM_raw is not None:
            f.write(f"unique_logM_raw      = {unique_logM_raw}\n")
        f.write(
            f"unique_logM_equal    = {unique_logM_eq}  "
            f"(of {len(logM_eq)})  [{mark(diversity_ok)}]\n\n"
        )
        f.write("=== logM split-half stability ===\n")
        f.write(f"half_A: median={median_a:+.4f}  std={std_a:.4f}  n={len(a)}\n")
        f.write(f"half_B: median={median_b:+.4f}  std={std_b:.4f}  n={len(b)}\n")
        f.write(
            f"|delta_median|       = {abs(median_a - median_b):.4f}  "
            f"[{mark(split_ok)}]\n\n"
        )
        f.write("Interpretation:\n")
        f.write("  ess_weighted < 0.8 * n_effective  -> raise --n-effective or --nlive\n")
        f.write("  logzerr >= 0.5                    -> lower --dlogz-init or raise --nlive\n")
        f.write("  unique_logM_equal/N_eq <= 0.25    -> likelihood plateaus or sampler stuck\n")
        f.write("  |delta_median| >= 0.1*std         -> posterior not well-mixed across halves\n")


def save_summary(path, mode, dict_data, dict_results):
    samples = dict_results["samps"]
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    truth_phys = stream_free_to_physical(dict_data["params"])
    derived = derived_sample_columns(samples)
    derived_q16, derived_q50, derived_q84 = np.percentile(derived, [16, 50, 84], axis=0)

    fixed_names = dict_results.get("fixed_names", ())
    ndim_fit = dict_results.get("ndim_fit", NDIM)
    with open(path / "summary.txt", "w", encoding="ascii") as f:
        f.write(f"seed = {dict_data['seed']}\n")
        f.write(f"mode = {mode}\n")
        f.write(f"nthreads = {dict_results.get('nthreads', 'unknown')}\n")
        f.write(f"ndim_fit = {ndim_fit} / {NDIM}\n")
        f.write(f"fixed = {','.join(fixed_names) if fixed_names else 'none'}\n")
        f.write(f"truth_trials = {dict_data.get('truth_trials', 'unknown')}\n")
        f.write(f"sigma_pct = {dict_data['sigma_pct']:.6f}\n")
        f.write(f"n_particles = {dict_data['n_particles']}\n")
        f.write(f"n_particles_effective = {dict_data['n_particles_effective']}\n")
        f.write(f"truth_speed = {truth_phys['speed']:.6f}\n")
        f.write(f"truth_time = {truth_phys['time']:.6f}\n")
        f.write(f"truth_v_circ = {truth_phys['v_circ']:.6f}\n\n")
        f.write("Mock LOS velocity datum:\n")
        f.write(f"  vz = {dict_data['vz']:.6f} kpc/Gyr\n")
        f.write(f"  vz_true = {dict_data['vz_true']:.6f} kpc/Gyr\n")
        f.write(f"  vz_err = {dict_data['vz_err']:.6f} kpc/Gyr\n")
        f.write(f"  vz_theta = {dict_data['vz_theta']:.6f} rad\n")
        f.write(f"  vz_window = {dict_data['vz_window']:.6f} rad\n")
        f.write(f"  vz_count = {dict_data['vz_count']}\n\n")
        f.write("Scale-free prior bounds:\n")
        f.write(f"  logRs = [{LOG_RS_MIN:.6g}, {LOG_RS_MAX:.6g}]\n")
        f.write(f"  logrs = [{LOG_RS_SAT_MIN:.6g}, {LOG_RS_SAT_MAX:.6g}]\n")
        f.write(f"  Rs = 10**logRs = [{RS_MIN_KPC:.6g}, {RS_MAX_KPC:.6g}] kpc\n")
        f.write(f"  rs = 10**logrs = [{RS_SAT_MIN_KPC:.6g}, {RS_SAT_MAX_KPC:.6g}] kpc\n")
        f.write(f"  log_alpha = [{LOG_ALPHA_MIN:.6g}, {LOG_ALPHA_MAX:.6g}]\n")
        f.write(f"  log_tau = [{LOG_TAU_MIN:.6g}, {LOG_TAU_MAX:.6g}]\n")
        f.write(f"  log_mfrac = [{LOG_MFRAC_MIN:.6g}, {LOG_MFRAC_MAX:.6g}]\n")
        f.write(f"  z0 = [{Z0_MIN_KPC:.6g}, {Z0_MAX_KPC:.6g}] kpc\n")
        f.write("  speed = 10**log_alpha * v_circ\n")
        f.write("  time = 10**log_tau * r0 / v_circ\n")
        f.write("  logm = logM + log_mfrac\n\n")
        for idx, label in enumerate(LABELS):
            f.write(
                f"{label:>10s}: truth={dict_data['params'][idx]: .6f} "
                f"median={q50[idx]: .6f} "
                f"-{q50[idx] - q16[idx]:.6f} +{q84[idx] - q50[idx]:.6f}\n"
            )
        for idx, label in enumerate(["|v|", "time", "vx0", "vy0", "vz0"]):
            truth_val = [truth_phys["speed"], truth_phys["time"], truth_phys["vx0"], truth_phys["vy0"], truth_phys["vz0"]][idx]
            f.write(
                f"{label:>10s}: truth={truth_val: .6f} "
                f"median={derived_q50[idx]: .6f} "
                f"-{derived_q50[idx] - derived_q16[idx]:.6f} +{derived_q84[idx] - derived_q50[idx]:.6f}\n"
            )


def fit_mode_path(seed_path, mode, fixed_names=()):
    if not fixed_names:
        return seed_path / mode
    suffix = "fix-" + "-".join(sorted(fixed_names))
    return seed_path / f"{mode}__{suffix}"


def run_fit_mode(seed_path, mode, dict_data, args):
    fixed_names = tuple(args.fix_params or ())
    mode_path = fit_mode_path(seed_path, mode, fixed_names)
    if mode_path.exists() and not args.overwrite:
        print(f"[seed={dict_data['seed']}] Skipping {mode}: existing output folder found at {mode_path}")
        if (mode_path / "dict_results.pkl").exists():
            with open(mode_path / "dict_results.pkl", "rb") as f:
                return pickle.load(f)
        return None

    mode_path.mkdir(parents=True, exist_ok=True)
    with open(mode_path / "dict_stream.pkl", "wb") as f:
        pickle.dump(dict_data, f)

    logl_fn = logl_unif if mode == "track" else logl_unif_los
    checkpoint_path = mode_path / "dynesty.save" if args.checkpoint_every > 0 else None
    fit_config = {
        "mode": mode,
        "seed": int(dict_data["seed"]),
        "fixed_names": list(fixed_names),
        "labels": list(LABELS),
        "radius_parameterization": "sampled_logRs_logrs",
        "n_particles": int(args.n_particles),
        "n_min": int(args.n_min),
        "var_ratio": float(args.var_ratio),
        "nlive": int(args.nlive),
        "n_effective": int(args.n_effective),
        "dlogz_init": float(args.dlogz_init),
        "checkpoint_every": float(args.checkpoint_every),
    }
    with open(mode_path / "fit_config.json", "w", encoding="ascii") as f:
        json.dump(fit_config, f, indent=2)

    print(
        f"[seed={dict_data['seed']}] Fitting mode={mode}, "
        f"nlive={args.nlive}, n_particles={args.n_particles}, "
        f"fixed={fixed_names or 'none'}, "
        f"checkpoint={'on' if checkpoint_path else 'off'}"
    )
    dict_results = dynesty_fit_unif(
        dict_data,
        logl_fn,
        n_particles=args.n_particles,
        n_min=args.n_min,
        var_ratio=args.var_ratio,
        nlive=args.nlive,
        n_effective=args.n_effective,
        dlogz_init=args.dlogz_init,
        nthreads=args.nthreads,
        fixed_names=fixed_names,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )
    with open(mode_path / "dict_results.pkl", "wb") as f:
        pickle.dump(dict_results, f)

    best_params = dict_results["samps"][np.argmax(dict_results["logl"])]
    plot_best_fit(mode_path, dict_data, best_params, args.n_particles)
    save_corner_plots(mode_path, dict_data, dict_results)
    save_q_posterior(mode_path, dict_data, dict_results)
    save_mass_posterior(mode_path, dict_data, dict_results)
    save_summary(mode_path, mode, dict_data, dict_results)
    save_diagnostics(mode_path, dict_data, dict_results, args.n_effective)
    return dict_results


def load_checkpoint_samples(mode_path):
    """Restore a partial dynesty run and return a dict_results-shaped dict.

    Raises FileNotFoundError if the checkpoint or fit_config is missing.
    """
    mode_path = Path(mode_path)
    config_path = mode_path / "fit_config.json"
    checkpoint_file = mode_path / "dynesty.save"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing fit_config.json at {config_path}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Missing dynesty.save at {checkpoint_file}")

    with open(config_path, "r", encoding="ascii") as f:
        config = json.load(f)
    with open(mode_path / "dict_stream.pkl", "rb") as f:
        dict_data = pickle.load(f)

    fixed_names = tuple(config.get("fixed_names") or ())
    free_indices, fixed_indices = resolve_fixed_indices(fixed_names)
    truth = np.asarray(dict_data["params"], dtype=np.float64)

    sampler = dynesty.DynamicNestedSampler.restore(str(checkpoint_file))
    res = sampler.results
    if len(res.samples) == 0:
        raise RuntimeError("Checkpoint has no samples yet; try again later.")

    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    raw_samps = res.samples[inds]

    if len(fixed_indices) == 0:
        full_samps = raw_samps
    else:
        full_samps = np.zeros((len(raw_samps), NDIM), dtype=np.float64)
        full_samps[:, free_indices] = raw_samps
        full_samps[:, fixed_indices] = truth[fixed_indices]

    return dict_data, {
        "dns": sampler,
        "samps": full_samps,
        "logl": res.logl[inds],
        "logz": res.logz,
        "logzerr": res.logzerr,
        "nthreads": "peek",
        "fixed_names": fixed_names,
        "free_indices": tuple(int(i) for i in free_indices),
        "fixed_indices": tuple(int(i) for i in fixed_indices),
        "ndim_fit": int(len(free_indices)),
    }


def peek_fit(mode_path):
    """Read a partial checkpoint and write peek/ corner + mass + summary plots."""
    mode_path = Path(mode_path)
    dict_data, dict_results = load_checkpoint_samples(mode_path)
    peek_dir = mode_path / "peek"
    peek_dir.mkdir(exist_ok=True)
    save_corner_plots(peek_dir, dict_data, dict_results)
    save_q_posterior(peek_dir, dict_data, dict_results)
    save_mass_posterior(peek_dir, dict_data, dict_results)
    save_summary(peek_dir, "peek", dict_data, dict_results)
    n_iter = int(dict_results["dns"].results.niter)
    n_eq = len(dict_results["samps"])
    logz_last = float(dict_results["logz"][-1])
    logzerr_last = float(dict_results["logzerr"][-1])
    print(
        f"[peek] {mode_path}: niter={n_iter}, n_equal={n_eq}, "
        f"logz={logz_last:+.3f}+/-{logzerr_last:.3f}"
    )


def replot_fit(mode_path, mode=None, n_particles=10000, n_effective=10000):
    """Re-run the plotting/summary pipeline from saved pickles.

    Use this when a fit completed (or crashed during plotting) and
    dict_results.pkl + dict_stream.pkl exist on disk.
    """
    mode_path = Path(mode_path)
    with open(mode_path / "dict_stream.pkl", "rb") as f:
        dict_data = pickle.load(f)
    with open(mode_path / "dict_results.pkl", "rb") as f:
        dict_results = pickle.load(f)

    if mode is None:
        mode = "track_los" if mode_path.name.startswith("track_los") else "track"

    best_params = dict_results["samps"][np.argmax(dict_results["logl"])]
    plot_best_fit(mode_path, dict_data, best_params, n_particles)
    save_corner_plots(mode_path, dict_data, dict_results)
    save_q_posterior(mode_path, dict_data, dict_results)
    save_mass_posterior(mode_path, dict_data, dict_results)
    save_summary(mode_path, mode, dict_data, dict_results)
    save_diagnostics(mode_path, dict_data, dict_results, n_effective)
    print(f"[replot] Wrote plots + summary into {mode_path}")


def save_mode_comparison(seed_path, dict_data, results_by_mode):
    if not {"track", "track_los"}.issubset(results_by_mode):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"track": "tab:orange", "track_los": "tab:blue"}
    labels = {"track": "Track only", "track_los": "Track + LOS"}

    ax = axes[0]
    for mode, result in results_by_mode.items():
        ax.hist(
            result["samps"][:, 0],
            bins=35,
            range=(10.0, 14.0),
            density=True,
            histtype="step",
            lw=2,
            color=colors[mode],
            label=labels[mode],
        )
    ax.axhline(0.25, color="0.5", ls="--", lw=1.5, label="Uniform prior")
    ax.axvline(dict_data["params"][0], color="red", lw=2, label="Truth")
    ax.set_xlabel("logM")
    ax.set_ylabel("Density")
    ax.set_xlim(10.0, 14.0)
    ax.legend(loc="best")

    ax = axes[1]
    for mode, result in results_by_mode.items():
        ax.hist(
            result["samps"][:, 2],
            bins=35,
            range=(0.5, 1.5),
            density=True,
            histtype="step",
            lw=2,
            color=colors[mode],
            label=labels[mode],
        )
    ax.axvline(dict_data["params"][2], color="red", lw=2, label="Truth")
    ax.set_xlabel("q")
    ax.set_ylabel("Density")
    ax.set_xlim(0.5, 1.5)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(seed_path / "posterior_comparison.pdf")
    plt.close(fig)


def has_los_mock_data(dict_data):
    return all(key in dict_data for key in ("vz", "vz_true", "vz_err", "vz_theta", "vz_window", "vz_count"))


def run_seed(seed, args):
    seed_path = Path(args.output_root) / f"seed{seed}"
    seed_path.mkdir(parents=True, exist_ok=True)
    print(f"[seed={seed}] Saving outputs to {seed_path}")

    stream_path = seed_path / "dict_stream.pkl"
    dict_data = None
    if stream_path.exists() and not args.overwrite:
        with open(stream_path, "rb") as f:
            dict_data = pickle.load(f)
        if has_los_mock_data(dict_data):
            print(f"[seed={seed}] Loaded existing mock stream from {stream_path}")
            if not (seed_path / "stream.pdf").exists():
                plot_mock_data_stream(seed_path, dict_data)
        else:
            print(f"[seed={seed}] Existing mock stream lacks LOS data; regenerating.")

    if dict_data is None or not has_los_mock_data(dict_data) or args.overwrite:
        dict_data = find_mock_data_stream(seed, args)
        with open(stream_path, "wb") as f:
            pickle.dump(dict_data, f)
        plot_mock_data_stream(seed_path, dict_data)

    modes = list(LIKELIHOOD_MODES) if args.mode == "both" else [args.mode]
    results_by_mode = {}
    for mode in modes:
        result = run_fit_mode(seed_path, mode, dict_data, args)
        if result is not None:
            results_by_mode[mode] = result

    if args.mode == "both":
        save_mode_comparison(seed_path, dict_data, results_by_mode)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode", choices=["track", "track_los", "both"], default="track")

    parser.add_argument("--n-particles", type=int, default=10000)
    parser.add_argument("--n-track-bins", type=int, default=36)
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--min-bins", type=int, default=10)
    parser.add_argument("--sigma-pct", type=float, default=2.0)
    parser.add_argument("--vz-err", type=float, default=20.0)
    parser.add_argument("--vz-window-factor", type=float, default=1.0)
    parser.add_argument("--truth-trials", type=int, default=512)
    parser.add_argument("--r-max", type=float, default=80.0)
    parser.add_argument("--max-bin-width", type=float, default=20.0)

    parser.add_argument("--nlive", type=int, default=2000)
    parser.add_argument("--n-effective", type=int, default=10000)
    parser.add_argument("--n-min", type=int, default=9)
    parser.add_argument("--var-ratio", type=float, default=1e-10)
    parser.add_argument("--dlogz-init", type=float, default=0.01)
    parser.add_argument("--nthreads", type=int, default=os.cpu_count())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fix-params",
        nargs="*",
        default=list(DEFAULT_FIXED_PARAMS),
        help=(
            "Parameter names to fix at their mock truth values during the fit. "
            f"Defaults to: {', '.join(DEFAULT_FIXED_PARAMS)}. "
            f"Default free parameters are: {', '.join(DEFAULT_FREE_PARAMS)}. "
            "Pass --fix-params with no names to fit all parameters. "
            f"Recommended-safe set: {', '.join(FIXABLE_PARAMS)}."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=600.0,
        help="Dynesty checkpoint interval in seconds (0 disables).",
    )
    parser.add_argument(
        "--peek",
        type=Path,
        default=None,
        help=(
            "Path to a mode output folder with a dynesty.save file. "
            "Restores the partial run, writes peek/ plots and summary, then exits."
        ),
    )
    parser.add_argument(
        "--replot",
        type=Path,
        default=None,
        help=(
            "Path to a finished mode output folder. Re-reads dict_results.pkl "
            "and dict_stream.pkl and regenerates plots + summary + diagnostics."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.peek is not None:
        peek_fit(args.peek)
        return

    if args.replot is not None:
        replot_fit(
            args.replot,
            n_particles=args.n_particles,
            n_effective=args.n_effective,
        )
        return

    args.output_root = Path(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    unknown = [p for p in (args.fix_params or ()) if p not in LABELS]
    if unknown:
        raise SystemExit(
            f"--fix-params contains unknown names: {unknown}. Valid names: {list(LABELS)}"
        )

    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    for seed in tqdm(seeds, leave=True):
        try:
            run_seed(seed, args)
        except RuntimeError as err:
            print(f"[seed={seed}] FAILED: {err}")


if __name__ == "__main__":
    main()
