# -*- coding: utf-8 -*-
"""
Several functions adapted from gabefreedman/etudes.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import constants as const
from matplotlib.patches import Ellipse


class Pulsar:
    """Pulsar object containing TOAs, uncertainties, sky position, and design matrix.

    Attributes:
        name (str): Name of the pulsar.
        toas (jax.numpy.ndarray): Times of arrival in seconds.
        toaerrs (jax.numpy.ndarray): Uncertainties on the TOAs in seconds.
        ra (float): Right Ascension in radians.
        dec (float): Declination in radians.
        _pos (jax.numpy.ndarray): Unit vector pointing to the pulsar in Cartesian coordinates.
        design_matrix (jax.numpy.ndarray): Design matrix for the timing model.
    """

    def __init__(self, name, toas, ra, dec, toaerrs=None, design_matrix=None, pdist=None):
        """
        Initialize a Pulsar object.

        Parameters:
        - name (str): Name of the pulsar.
        - toas (array-like): Times of arrival (TOAs) in seconds.
        - toaerrs (array-like): Errors on the TOAs in seconds.
        - ra (float): Right Ascension in radians.
        - dec (float): Declination in radians.
        - design_matrix (array-like): Design matrix for the pulsar.
        - pdist (array-like): Gaussian mean and std of pulsar distance measurement in kpc.
        """
        self.name = name
        self.toas = jnp.array(toas)
        self.toaerrs = jnp.array(toaerrs)
        self.ra = ra
        self.dec = dec
        self._pos = jnp.array([
            jnp.cos(dec) * jnp.cos(ra),
            jnp.cos(dec) * jnp.sin(ra),
            jnp.sin(dec),
        ])
        self.design_matrix = jnp.array(design_matrix)
        self.pdist = jnp.array(pdist)

    def __repr__(self):
        return (
            f"Pulsar(name={self.name!r}, ra={self.ra:.6f}, dec={self.dec:.6f}, "
            f"toas_shape={self.toas.shape}, design_matrix_shape={self.design_matrix.shape})"
        )


def create_gw_antenna_pattern(pos, gwtheta, gwphi):
    """Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).
    :param pos: Unit vector from Earth to pulsar
    :type pos: array-like
    :param gwtheta: GW-origin polar angle in radians
    :type gwtheta: float
    :param gwphi: GW-origin azimuthal angle in radians
    :type gwphi: float

    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the
             pulsar and the GW source.
    :rtype: tuple
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = jnp.array([jnp.sin(gwphi), -jnp.cos(gwphi), 0.0])
    n = jnp.array(
        [
            -jnp.cos(gwtheta) * jnp.cos(gwphi),
            -jnp.cos(gwtheta) * jnp.sin(gwphi),
            jnp.sin(gwtheta),
        ]
    )
    omhat = jnp.array(
        [
            -jnp.sin(gwtheta) * jnp.cos(gwphi),
            -jnp.sin(gwtheta) * jnp.sin(gwphi),
            -jnp.cos(gwtheta),
        ]
    )

    fplus = (
        0.5 * (jnp.dot(m, pos) ** 2 - jnp.dot(n, pos) ** 2) / (1 + jnp.dot(omhat, pos))
    )
    fcross = (jnp.dot(m, pos) * jnp.dot(n, pos)) / (1 + jnp.dot(omhat, pos))
    cosMu = -jnp.dot(omhat, pos)

    return fplus, fcross, cosMu


@register_pytree_node_class
class CW_Signal(object):
    """Class for single-pulsar deterministic continuous-wave signals. Main
    output function is `get_delay`, with different forms of the
    function defined at initialization of the class.

    :param psr: A Pulsar object containing pulsar TOAs and residuals
    :type psr: :class:`enterprise.pulsar.Pulsar`
    :param tref: The reference time for phase and frequency [s]
    :type tref: float, optional
    :param evolve: Whether to use full frequency evolution or phase approximation
    :type evolve: bool, optional
    """

    def __init__(self, psr, tref=0, evolve=True):
        """Constructor method"""
        self.psr = psr
        self.tref = tref
        self.evolve = evolve

        self._init_delay(evolve=self.evolve)

    def _init_delay(self, evolve=True):
        """Set the form of the delay function"""
        if evolve:
            self._freqevol_fn = self._full_evolve
        else:
            self._freqevol_fn = self._phase_approx

    def _full_evolve(self, w0, mc, toas, phase0):
        """Full frequency and phase evolution"""
        omega = w0 * (1 - 256 / 5 * mc ** (5 / 3) * w0 ** (8 / 3) * toas) ** (-3 / 8)
        phase = phase0 + 1 / 32 / mc ** (5 / 3) * (w0 ** (-5 / 3) - omega ** (-5 / 3))
        return omega, phase

    def _phase_approx(self, w0, mc, toas, phase0):
        """Phase approximation across observational timespan"""
        omega = w0
        phase = phase0 + omega * toas
        return omega, phase

    def get_delay(self, pars, **kwargs):
        """Call underlying `_get_delay` function for input parameters"""
        return self._get_delay(**pars, **kwargs)
    
    def get_fulldelay(self, pars, **kwargs):
        """Call underlying `_get_delay` function for input parameters"""
        return self._get_fulldelay(**pars, **kwargs)

    @jax.jit
    def _get_delay(
        self,
        cw_costheta=0,
        cw_phi=0,
        cw_cosinc=0,
        cw_log10_Mc=9,
        cw_log10_fgw=-8,
        cw_log10_dist=2,
        cw_phase0=0,
        cw_psi=0,
        **kwargs,
    ):
        """Generalized function to compute GW induced residuals from a SMBHB,
        defined in Ellis et. al 2012, 2013.

        :param cw_costheta: Cosine of the GW source polar angle in
            celestial coordinates [radians]
        :type cw_costheta: float, optional
        :param cw_phi: GW source azimuthal angle in celestial
            coordinates [radians]
        :type cw_phi: float, optional
        :param cw_cosinc: Cosine of the inclination of the GW source [radians]
        :type cw_cosinc: float, optional
        :param cw_log10_Mc: log10 of the SMBHB chirp mass [solar masses]
        :type cw_log10_Mc: float, optional
        :param cw_log10_fgw: log10 of the GW frequency [Hz]
        :type cw_log10_fgw: float, optional
        :param cw_log10_dist: log10 of the GW source distance [Mpc]
        :type cw_log10_dist: float, optional
        :param cw_phase0: Initial phase of the GW source [radians]
        :type cw_phase0: float, optional
        :param cw_psi: Polarization angle of the GW source [radians]
        :type cw_psi: float, optional

        :return: GW induced residuals from continuous wave source
        :rtype: array-like
        """
        # convert all units to time
        mc = 10**cw_log10_Mc * const.Tsun
        dist = 10**cw_log10_dist * const.Mpc / const.c
        fgw = 10**cw_log10_fgw
        gwtheta = jnp.arccos(cw_costheta)
        inc = jnp.arccos(cw_cosinc)

        # calculate antenna pattern and cosMu
        fplus, fcross, _ = create_gw_antenna_pattern(self.psr._pos, gwtheta, cw_phi)

        # subtract reference time from TOAs
        toas = self.psr.toas - self.tref

        # orbital frequency and phase
        w0 = jnp.pi * fgw
        phase0 = cw_phase0 / 2

        # calculate frequency and phaes evolution
        omega, phase = self._freqevol_fn(w0, mc, toas, phase0)

        # define time dependent coefficients and amplitudes
        At = -0.5 * jnp.sin(2 * phase) * (3 + jnp.cos(2 * inc))
        Bt = 2 * jnp.cos(2 * phase) * jnp.cos(inc)
        alpha = mc ** (5.0 / 3.0) / (dist * omega ** (1.0 / 3.0))

        # calculate rplus and rcross
        rplus = alpha * (-At * jnp.cos(2 * cw_psi) + Bt * jnp.sin(2 * cw_psi))
        rcross = alpha * (At * jnp.sin(2 * cw_psi) + Bt * jnp.cos(2 * cw_psi))

        # calculate residuals
        res = -fplus * rplus - fcross * rcross

        return res
    
    @jax.jit
    def _get_fulldelay(
        self,
        cw_costheta=0,
        cw_phi=0,
        cw_cosinc=0,
        cw_log10_Mc=9,
        cw_log10_fgw=-8,
        cw_log10_dist=2,
        cw_phase0=0,
        cw_psi=0,
        **kwargs,
    ):
        """Generalized function to compute GW induced residuals from a SMBHB,
        defined in Ellis et. al 2012, 2013.

        :param cw_costheta: Cosine of the GW source polar angle in
            celestial coordinates [radians]
        :type cw_costheta: float, optional
        :param cw_phi: GW source azimuthal angle in celestial
            coordinates [radians]
        :type cw_phi: float, optional
        :param cw_cosinc: Cosine of the inclination of the GW source [radians]
        :type cw_cosinc: float, optional
        :param cw_log10_Mc: log10 of the SMBHB chirp mass [solar masses]
        :type cw_log10_Mc: float, optional
        :param cw_log10_fgw: log10 of the GW frequency [Hz]
        :type cw_log10_fgw: float, optional
        :param cw_log10_dist: log10 of the GW source distance [Mpc]
        :type cw_log10_dist: float, optional
        :param cw_phase0: Initial phase of the GW source [radians]
        :type cw_phase0: float, optional
        :param cw_psi: Polarization angle of the GW source [radians]
        :type cw_psi: float, optional

        :return: GW induced residuals from continuous wave source
        :rtype: array-like
        """
        # convert all units to time
        mc = 10**cw_log10_Mc * const.Tsun
        dist = 10**cw_log10_dist * const.Mpc / const.c
        fgw = 10**cw_log10_fgw
        gwtheta = jnp.arccos(cw_costheta)
        inc = jnp.arccos(cw_cosinc)
        dist_key = f"{self.psr.name}_pdist"
        psr_dist = kwargs.get(dist_key, None)
        p_dist = (self.psr.pdist[0] + self.psr.pdist[1] * psr_dist) * const.kpc / const.c
        phase_key = f"{self.psr.name}_pphase"
        p_phase = kwargs.get(phase_key, None)

        # calculate antenna pattern and cosMu
        fplus, fcross, cosMu = create_gw_antenna_pattern(self.psr._pos, gwtheta, cw_phi)

        # subtract reference time from TOAs
        toas = self.psr.toas - self.tref
        # get pulsar time
        tp = toas - p_dist * (1 - cosMu)

        # orbital frequency and phase
        w0 = jnp.pi * fgw
        phase0 = cw_phase0 / 2

        omega, phase = self._freqevol_fn(w0, mc, toas, phase0)
        omega_p, phase_p = self._freqevol_fn(w0, mc, tp, phase0)
        omega_p0 = omega_p[0]

        if p_phase is None:
                phase_p = phase0 + 1 / 32 / mc ** (5 / 3) * (
                    w0 ** (-5 / 3) - omega_p ** (-5 / 3)
                )
        else:
            phase_p = (
                phase0
                + p_phase
                + 1 / 32 * mc ** (-5 / 3) * (omega_p0 ** (-5 / 3) - omega_p ** (-5 / 3))
            )

        # define time dependent coefficients
        At = -0.5 * jnp.sin(2 * phase) * (3 + jnp.cos(2 * inc))
        Bt = 2 * jnp.cos(2 * phase) * jnp.cos(inc)
        At_p = -0.5 * jnp.sin(2 * phase_p) * (3 + jnp.cos(2 * inc))
        Bt_p = 2 * jnp.cos(2 * phase_p) * jnp.cos(inc)

        # now define time dependent amplitudes
        alpha = mc ** (5.0 / 3.0) / (dist * omega ** (1.0 / 3.0))
        alpha_p = mc ** (5.0 / 3.0) / (dist * omega_p ** (1.0 / 3.0))

        # define rplus and rcross
        rplus = alpha * (-At * jnp.cos(2 * cw_psi) + Bt * jnp.sin(2 * cw_psi))
        rcross = alpha * (At * jnp.sin(2 * cw_psi) + Bt * jnp.cos(2 * cw_psi))
        rplus_p = alpha_p * (-At_p * jnp.cos(2 * cw_psi) + Bt_p * jnp.sin(2 * cw_psi))
        rcross_p = alpha_p * (At_p * jnp.sin(2 * cw_psi) + Bt_p * jnp.cos(2 * cw_psi))

        # residuals
        res = fplus * (rplus_p - rplus) + fcross * (rcross_p - rcross)

        return res

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (), (self.psr, self.tref, self.evolve)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method for reconstructing custom PyTree"""
        return cls(*children, *aux_data)
    

def compute_fisher(cw_signal, pars, pterm=False):
    """
    Compute the Fisher information matrix for a continuous-wave signal.

    Parameters
    ----------
    cw_signal : object
        An object representing the continuous-wave (CW) signal model. It must provide:
          - psr.toaerrs : array_like, shape (n_toas,)
              The uncertainties on each time-of-arrival (TOA).
          - get_delay(pars : dict) -> array_like, shape (n_toas,)
              A method that returns the signal delays for a given set of parameters.
    pars : dict
        A mapping from parameter names (str) to their current values (float or array-like).
        The number of keys defines n_params, the dimensionality of the parameter space.

    Returns
    -------
    info_mat : jax.numpy.ndarray, shape (n_params, n_params)
        The Fisher information matrix evaluated at the given parameter values:
            info_mat = J^T · (J / σ²)
        where J is the Jacobian of the delay model w.r.t. the parameters
        and σ are the TOA errors from cw_signal.psr.toaerrs.

    Notes
    -----
    - Uses JAX’s automatic differentiation to compute the Jacobian.
    - Assumes TOA errors are uncorrelated and Gaussian.

    Example
    -------
    >>> pars = {'f0': 100.0, 'f1': 1e-10}
    >>> info = compute_fisher(cw_signal, pars)
    """
    param_keys = list(pars.keys())
    param_vec = jnp.array(list(pars.values()))

    if pterm:
        sig_func = cw_signal.get_fulldelay
    else:
        sig_func = cw_signal.get_delay
    
    # rebuild delays function
    def delay_vec(vec):
        p = dict(zip(param_keys, vec))
        return sig_func(pars=p)
        #return cw_signal.get_delay(pars=p)
    
    # Jacobian: shape (n_params, n_toas)
    jac = jax.jacrev(delay_vec)(param_vec)
    
    # “test” matrix and Fisher
    info_mat = jac.T @ (jac/cw_signal.psr.toaerrs[:,None]**2)
    
    return info_mat


def computer_snr2(cw_signal, pars, pterm=False):
    """
    Compute the squared signal-to-noise ratio (SNR²) for a continuous-wave signal.

    Parameters
    ----------
    cw_signal : object
        Continuous-wave signal instance that must provide:
          - get_delay(pars) → array-like: predicted timing residuals (signal template)
          - psr.toaerrs : array-like of shape (N,)
            time‐of‐arrival uncertainties for each residual.
    pars : dict or array-like
        Model parameters passed to cw_signal.get_delay to generate the signal template.

    Returns
    -------
    float
        The optimal SNR squared, computed as
        ⟨s | s / σ²⟩ = sᵀ · (s / σ²),
        where s is the signal template and σ are the TOA errors.
    """
    if pterm:
        sig_func = cw_signal.get_fulldelay
    else:
        sig_func = cw_signal.get_delay

    sig = sig_func(pars=pars)
    #sig = cw_signal.get_delay(pars=pars)
    snr2 = sig @ (sig/cw_signal.psr.toaerrs**2)
    return snr2


def plot_fisher_ellipse_on_sky(F_uphi, center_uphi, ax=None, confidence=0.68, **ellipse_kwargs):
    """
    F_uphi : 2×2 Fisher matrix for parameters (u=cosθ, φ) at the point center_uphi=(u0, φ0).
    center_uphi : (u0, φ0) in radians.
    """
    u0, phi0 = center_uphi
    # compute latitude beta0 = arcsin(u0)
    beta0 = jnp.arcsin(u0)

    # Jacobian from q=(λ=φ, β=latitude) to p=(u, φ)
    J = jnp.array([[0.0,            jnp.cos(beta0)],
                   [1.0,            0.0           ]])
    # transform Fisher into (λ,β)
    F_q = J.T @ F_uphi @ J
    # covariance in (λ,β)
    cov_q = jnp.linalg.inv(F_q)

    # eigen-decompose covariance
    eigvals, eigvecs = jnp.linalg.eigh(cov_q)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:,order]

    # χ² for 2D ellipse
    chi2_dict = {0.68:2.30, 0.95:5.99, 0.99:9.21}
    s = chi2_dict.get(confidence, 2.30)

    # widths in radians
    width  = 2 * jnp.sqrt(eigvals[0]*s)
    height = 2 * jnp.sqrt(eigvals[1]*s)
    # compute angle of ellipse major‐axis in degrees using jnp only
    angle = jnp.degrees(jnp.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # set up Mollweide if needed
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={'projection':'mollweide'})
    # center in Mollweide coords: (lon, lat)
    center = (phi0, beta0)

    # draw
    ell = Ellipse(xy=center, width=width, height=height,
                  angle=angle, **ellipse_kwargs)
    ax.add_patch(ell)
    return ax


def pulsar_ring_generator(pars, ang_radius=10.0, npsrs=20, toas=None, toaerrs=None, pdist=None):
    # Generate npsrs pulsars on a ring around the GW sky location
    n_ring = npsrs
    alpha = jnp.deg2rad(ang_radius)  # angular radius of the ring in radians
    phi_ring = jnp.linspace(0.1, 2*jnp.pi+0.1, n_ring, endpoint=False)

    # center coordinates of GW source from pars dict
    dec0 = jnp.arcsin(pars['cw_costheta']) #jnp.pi/2 - pars['cw_costheta']
    ra0  = (pars['cw_phi'] + jnp.pi) % (2*jnp.pi) - jnp.pi

    # compute ring coordinates
    dec_ring = jnp.arcsin(
        jnp.sin(dec0) * jnp.cos(alpha) +
        jnp.cos(dec0) * jnp.sin(alpha) * jnp.cos(phi_ring)
    )
    ra_ring = ra0 + jnp.arctan2(
        jnp.sin(alpha) * jnp.sin(phi_ring),
        jnp.cos(dec0) * jnp.cos(alpha) - jnp.sin(dec0) * jnp.sin(alpha) * jnp.cos(phi_ring)
    )
    ra_ring = (ra_ring + jnp.pi) % (2*jnp.pi) - jnp.pi

    # instantiate pulsars on the ring (using existing toas_new and toaerrs)
    psrs_ring = [
            Pulsar(
            name=f"RING{i:03d}",
            ra=ra_ring[i],
            dec=dec_ring[i],
            toas=toas,
            toaerrs=toaerrs,
            pdist=pdist
        )
        for i in range(n_ring)
    ]
    return psrs_ring


def pulsar_annulus_generator(
    pars,
    inner_ang_radius=10.0,
    width_deg=1.0,
    npsrs=20,
    toas=None,
    toaerrs=None,
    pdist=None,
    key=None,
):
    """
    Generate npsrs pulsars uniformly at random inside a thin spherical annulus
    centered on the GW sky location.

    Sampling:
      - Let α be the angular separation from the GW source direction.
        We draw μ = cos(α) uniformly between cos(α_outer) and cos(α_inner)
        (this gives uniform surface density in the annulus).
      - φ is uniform in [0, 2π).
    Geometry:
      Given source (ra0, dec0), for each (α, φ):
        sin(dec) = sin(dec0) cos α + cos(dec0) sin α cos φ
        ΔRA = atan2( sin α sin φ,
                     cos(dec0) cos α - sin(dec0) sin α cos φ )
        ra = wrap(ra0 + ΔRA) into [-π, π].

    Parameters
    ----------
    pars : dict
        Must contain 'cw_costheta' (cos dec) and 'cw_phi' (RA) of GW source.
    inner_ang_radius : float
        Inner edge of annulus (degrees) measured from source direction.
    width_deg : float
        Angular width of annulus in degrees (outer = inner + width).
    npsrs : int
        Number of pulsars to generate.
    toas, toaerrs : array-like
        Timing arrays passed to Pulsar constructor.
    pdist : array-like or None
        Pulsar distance info passed through.
    key : jax.random.PRNGKey or None
        If provided, used for reproducible sampling. If None, a default is created.

    Returns
    -------
    list[Pulsar]
        List of Pulsar objects in the annulus.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    key_phi, key_mu = jax.random.split(key)

    # GW source coordinates
    dec0 = jnp.arcsin(pars["cw_costheta"])
    ra0 = (pars["cw_phi"] + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # Convert bounds to radians
    alpha_in = jnp.deg2rad(inner_ang_radius)
    alpha_out = jnp.deg2rad(inner_ang_radius + width_deg)

    # Ensure inner < outer (swap if user inverted them)
    alpha_inner = jnp.minimum(alpha_in, alpha_out)
    alpha_outer = jnp.maximum(alpha_in, alpha_out)

    # Uniform in μ = cos α between cos(outer) and cos(inner)
    mu_min = jnp.cos(alpha_outer)
    mu_max = jnp.cos(alpha_inner)
    mu = jax.random.uniform(key_mu, shape=(npsrs,), minval=mu_min, maxval=mu_max)
    alpha = jnp.arccos(mu)

    # Uniform φ
    phi = 2 * jnp.pi * jax.random.uniform(key_phi, shape=(npsrs,))

    # Precompute sines/cosines
    sin_dec0 = jnp.sin(dec0)
    cos_dec0 = jnp.cos(dec0)
    sin_alpha = jnp.sin(alpha)
    cos_alpha = jnp.cos(alpha)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    # declinations
    sin_dec = sin_dec0 * cos_alpha + cos_dec0 * sin_alpha * cos_phi
    # clamp numerical drift
    sin_dec = jnp.clip(sin_dec, -1.0, 1.0)
    dec = jnp.arcsin(sin_dec)

    # RA offset
    delta_ra = jnp.arctan2(
        sin_alpha * sin_phi,
        cos_dec0 * cos_alpha - sin_dec0 * sin_alpha * cos_phi,
    )
    ra = ra0 + delta_ra
    ra = (ra + jnp.pi) % (2 * jnp.pi) - jnp.pi

    psrs = [
        Pulsar(
            name=f"ANNULUS{i:03d}",
            ra=ra[i],
            dec=dec[i],
            toas=toas,
            toaerrs=toaerrs,
            pdist=pdist,
        )
        for i in range(npsrs)
    ]
    return psrs


def pulsar_cap_generator(
    pars,
    ang_radius=10.0,
    npsrs=20,
    toas=None,
    toaerrs=None,
    pdist=None,
    key=None,
):
    """
    Generate npsrs pulsars uniformly at random inside a spherical cap
    of angular radius ang_radius (degrees) centered on the GW sky location.

    Sampling:
      - Let α be the angular separation from the GW source direction.
        Draw μ = cos(α) uniformly in [cos(ang_radius), 1]  -> uniform surface density.
      - φ uniform in [0, 2π).
      - Convert (α, φ) relative to source direction (ra0, dec0).

    Parameters
    ----------
    pars : dict
        Must contain 'cw_costheta' (cos dec) and 'cw_phi' (RA) of GW source.
    ang_radius : float
        Cap angular radius in degrees (0 < ang_radius ≤ 180).
    npsrs : int
        Number of pulsars to generate.
    toas, toaerrs : array-like
        Timing arrays passed to Pulsar constructor.
    pdist : array-like or None
        Pulsar distance info passed through.
    key : jax.random.PRNGKey or None
        PRNG key for reproducibility; if None a default key is used.

    Returns
    -------
    list[Pulsar]
        Pulsars uniformly distributed over the cap.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    key_phi, key_mu = jax.random.split(key)

    # Center (GW source) coordinates
    dec0 = jnp.arcsin(pars["cw_costheta"])
    ra0 = (pars["cw_phi"] + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # Max separation in radians
    alpha_max = jnp.deg2rad(ang_radius)

    # Sample cos(alpha) uniformly
    mu = jax.random.uniform(
        key_mu,
        shape=(npsrs,),
        minval=jnp.cos(alpha_max),
        maxval=1.0,
    )
    alpha = jnp.arccos(mu)

    # Sample φ uniformly
    phi = 2 * jnp.pi * jax.random.uniform(key_phi, shape=(npsrs,))

    # Precompute
    sin_dec0 = jnp.sin(dec0)
    cos_dec0 = jnp.cos(dec0)
    sin_alpha = jnp.sin(alpha)
    cos_alpha = jnp.cos(alpha)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    # Declination
    sin_dec = sin_dec0 * cos_alpha + cos_dec0 * sin_alpha * cos_phi
    sin_dec = jnp.clip(sin_dec, -1.0, 1.0)
    dec = jnp.arcsin(sin_dec)

    # RA offset
    delta_ra = jnp.arctan2(
        sin_alpha * sin_phi,
        cos_dec0 * cos_alpha - sin_dec0 * sin_alpha * cos_phi,
    )
    ra = ra0 + delta_ra
    ra = (ra + jnp.pi) % (2 * jnp.pi) - jnp.pi

    psrs = [
        Pulsar(
            name=f"CAP{i:03d}",
            ra=ra[i],
            dec=dec[i],
            toas=toas,
            toaerrs=toaerrs,
            pdist=pdist,
        )
        for i in range(npsrs)
    ]
    return psrs


def fisher_marg(pars, F):
    # Find indices of sky location parameters
    keys = list(pars.keys())
    try:
        iu = keys.index("cw_costheta")
        iphi = keys.index("cw_phi")
    except ValueError as e:
        raise KeyError("Required keys 'cw_costheta' and 'cw_phi' not both present in pars.") from e

    # Build permutation: put (cw_costheta, cw_phi) first, keep relative order of the rest
    remaining = [k for k in range(len(keys)) if k not in (iu, iphi)]
    perm = [iu, iphi] + remaining

    # Permute Fisher matrix
    Fp = F[jnp.array(perm)[:, None], jnp.array(perm)]

    # Block components after reordering
    B = Fp[2:, :2]
    C = Fp[2:, 2:]

    # Schur complement (marginal Fisher over cw_costheta, cw_phi)
    Ftilde = Fp[:2, :2] - B.T @ jnp.linalg.pinv(C) @ B
    return Ftilde