import jax
import jax.numpy as jnp
import StreaMAX

from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import numpy as np

@jax.jit
def get_q(dirx, diry, dirz, q_min=0.5, q_max=1.5):
    """
    Computes the axis ratio q from the direction vector components. Uniform [0.5, 1.5].
    """
    r  = jnp.sqrt(dirx**2 + diry**2 + dirz**2) 
    q  = jnp.exp(-r**2/2) * (jnp.sqrt(jnp.pi) * jnp.exp(r**2/2) * jax.scipy.special.erf(r/jnp.sqrt(2)) - jnp.sqrt(2)*r)/jnp.sqrt(jnp.pi)
    q  = (q_max-q_min)*q + q_min

    return q

def halo_mass_from_stellar_mass(M_star, 
                                N=0.0351, log10_M1=11.59, beta=1.376, gamma=0.608,
                                mmin=1e9, mmax=3e16, tol=1e-6, max_iter=200):
    """
    Return halo mass M_h [Msun] for a given stellar mass M_star [Msun]
    using the Moster+2013 z=0 SHMR (median relation).
    """
    def mstar_from_mh(Mh):
        x = Mh / (10**log10_M1)
        return 2*N*Mh / (x**(-beta) + x**gamma)

    a, b = mmin, mmax
    for _ in range(max_iter):
        mid = 10**((jnp.log10(a)+jnp.log10(b))/2)
        if mstar_from_mh(mid) > M_star:
            b = mid
        else:
            a = mid
        if abs(jnp.log10(b) - jnp.log10(a)) < tol:
            return 10**((jnp.log10(a)+jnp.log10(b))/2)
    return 10**((jnp.log10(a)+jnp.log10(b))/2)

def params_to_stream(params, n_particles=10000, n_steps=99, alpha=1., unroll=True):
    # Flattened NFW halo
    type_host   = 'NFW'
    params_host = {'logM': params[0], 'Rs': params[1], 
                    'a': 1.0, 'b': 1.0, 'c': get_q(params[2], params[3], params[4]),
                    'dirx': params[2], 'diry': params[3], 'dirz': params[4],
                    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0}

    # Plummer Sattelite
    type_sat   = 'Plummer'
    params_sat = {'logM': params[5], 'Rs': params[6],
                    'x_origin': params[7], 'y_origin': 0.0, 'z_origin': params[8]}

    # Initial conditions
    xv_f = jnp.array([params[7], 0.0, params[8],  # Position in kpc
                    params[9], params[10], params[11]])   # Velocity in kpc/Gyr

    # Integration time
    time  = params[12] # Gyr

    _, _, xv_stream, xhi_stream = StreaMAX.generate_stream(xv_f, 
                                                            type_host, params_host, 
                                                            type_sat, params_sat, 
                                                            time, alpha, n_steps,
                                                            n_particles, 
                                                            unroll)
    _, _, theta_stream, r_stream, _ = StreaMAX.get_stream_ordered(xv_stream[:, 0], xv_stream[:, 1], xhi_stream)

    return theta_stream, r_stream, xv_stream

def params_to_stream_DiskNFW(params, disk_mass=10.0, n_particles=10000, n_steps=99, alpha=1., unroll=True):
    # Disk + NFW halo
    type_host  = 'DiskNFW'
    params_host = {'NFW_params': {'logM': params[0], 'Rs': params[1],
                                    'a': 1.0, 'b': 1.0, 'c': get_q(params[2], params[3], params[4]),
                                    'dirx': params[2], 'diry': params[3], 'dirz': params[4],
                                    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0},

                # MW disk
                'MN_params': {'logM': disk_mass, 'Rs': 3., 'Hs': 0.3,
                                'dirx': 1.0, 'diry': 1.0, 'dirz': 1.0,
                                'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0},
                }

    # Plummer Sattelite
    type_sat   = 'Plummer'
    params_sat = {'logM': params[5], 'Rs': params[6],
                    'x_origin': params[7], 'y_origin': 0.0, 'z_origin': params[8]}

    # Initial conditions
    xv_f = jnp.array([params[7], 0.0, params[8],  # Position in kpc
                    params[9], params[10], params[11]])   # Velocity in kpc/Gyr

    # Integration time
    time  = params[12] # Gyr

    _, xv_sat, xv_stream, xhi_stream = StreaMAX.generate_stream(xv_f, 
                                                            type_host, params_host, 
                                                            type_sat, params_sat, 
                                                            time, alpha, n_steps,
                                                            n_particles, 
                                                            unroll)
    _, _, theta_stream, r_stream, _ = StreaMAX.get_stream_ordered(xv_stream[:, 0], xv_stream[:, 1], xhi_stream)

    return theta_stream, r_stream, xv_stream, xv_sat

def get_residuals_and_mask(path, sga, name, vminperc=35, vmaxperc=90):
    # Load Residuals
    with fits.open(f"{path}/{name}/data.fits") as hdul:
        header = hdul[0].header
        data = hdul[0].data
    with fits.open(f"{path}/{name}/model.fits") as hdul:
        model = hdul[0].data
    residual = data - model
    residual = np.median(residual, axis=0)
    mm = np.nanpercentile(residual, [vminperc, vmaxperc])
    residual = np.nan_to_num(np.clip(residual, mm[0], mm[1]), 0.0)

    # Load Mask
    with fits.open(f"{path}/{name}/mask.fits") as hdul:
        mask = hdul[0].data
    mask = mask/mask.max() # This assumes that only one mask is present

    # Get Redshift and pixel scale
    sga_name = name.split('_')[0]
    PA = sga[sga['GALAXY'] == sga_name]['PA'].data[0]
    z_redshift = sga[sga['GALAXY'] == sga_name]['Z_LEDA'].data[0]
    pixel_to_deg = abs(header['PC1_1'])
    pixel_to_kpc = pixel_to_deg * np.pi / 180 * cosmo.comoving_transverse_distance(z_redshift).value * 1000

    return residual, mask, z_redshift, pixel_to_kpc, PA