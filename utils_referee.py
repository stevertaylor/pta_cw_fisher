# -*- coding: utf-8 -*-
"""
Shared machinery for the referee-response studies (arXiv:2603.10120).

Imported by the results notebook ring_pterm_referee.ipynb, keeping the
plotted results and the underlying code paths in a single audited module.

Contents
--------
- compute_sky_area / compute_sky_area_schur: identical to the audited
  definitions in ring_pterm_vectorized_fixed.ipynb (CELL 0), with an
  rtol_scale knob for robustness tests.
- load_ng15 / build_ng15_batch / scale_pdist_sigma: the NG15-like array
  (data/ng15_psrs.csv) represented on a common TOA grid with differing
  baselines handled by weight masking.
- calibrated_fisher: SNR=10 calibration + Fisher + priors for an arbitrary
  PulsarBatch (generalizes the notebook helpers to heterogeneous arrays).
- ring_sweep_area: Fig.-7-style ring sweeps generalized to arbitrary mean
  pulsar distance L0 (the production notebook hardcoded 1 kpc).
- area_for_pos: sky-map helper (generalizes compute_fisher_for_pos).
- galactic helpers (astropy) for Galactic-coordinate sky maps.
"""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)  # must precede any array creation

import jax.numpy as jnp
import numpy as np

import constants as const
import utils_vectorized_fixed as uvf

# Paper-fiducial GW source and PTA timing configuration
# (ring_pterm_vectorized_fixed.ipynb CELL 1)
PAPER_PARS = dict(
    cw_costheta=0.01, cw_phi=0.0, cw_cosinc=0.99, cw_log10_Mc=9.5,
    cw_log10_fgw=-8.0, cw_log10_dist=2.385, cw_phase0=0.0, cw_psi=0.0,
)
PAPER_T0 = 58000.0 * 86400.0
PAPER_TOAS = jnp.linspace(PAPER_T0, PAPER_T0 + 10.0 * 365.25 * 86400.0, 100)
PAPER_TOAERRS = jnp.full_like(PAPER_TOAS, 1e-7)
WAVELENGTH_KPC = const.c / (10 ** PAPER_PARS["cw_log10_fgw"]) / const.kpc

SIGMA_PAD = 1e10  # [s] TOA error assigned to masked (inactive) epochs


def pars_vec(pars=None):
    return jnp.array(list((pars or PAPER_PARS).values()))


# =============================================================================
# Sky-area functions (audited notebook CELL 0 definitions + rtol_scale knob)
# =============================================================================
def compute_sky_area(F, marginalized=True, rtol_scale=1.0):
    F = (F + F.T) / 2.0
    if marginalized:
        diag_F = jnp.diag(F)
        D_inv = 1.0 / jnp.sqrt(jnp.maximum(diag_F, 1e-30))
        F_scaled = D_inv[:, None] * F * D_inv[None, :]
        rtol = jnp.finfo(F.dtype).eps * F.shape[0] * rtol_scale
        F_inv_scaled = jnp.linalg.pinv(F_scaled, rtol=rtol)
        F_inv = D_inv[:, None] * F_inv_scaled * D_inv[None, :]
        det_val = jnp.linalg.det(F_inv[:2, :2])
        return (180 / jnp.pi) ** 2 * 2 * jnp.pi * jnp.sqrt(jnp.maximum(det_val, 1e-30))
    det_val = jnp.linalg.det(F[:2, :2])
    return (180 / jnp.pi) ** 2 * 2 * jnp.pi / jnp.sqrt(jnp.maximum(det_val, 1e-30))


def compute_sky_area_schur(F):
    F = (F + F.T) / 2.0
    A = F[:2, :2]
    B = F[2:, :2]
    C = F[2:, 2:]
    diag_C = jnp.diag(C)
    D_inv = 1.0 / jnp.sqrt(jnp.maximum(diag_C, 1e-30))
    C_scaled = D_inv[:, None] * C * D_inv[None, :]
    rtol = jnp.finfo(C.dtype).eps * C.shape[0]
    C_inv_scaled = jnp.linalg.pinv(C_scaled, rtol=rtol)
    C_inv = D_inv[:, None] * C_inv_scaled * D_inv[None, :]
    F_sky = A - B.T @ C_inv @ B
    F_sky = (F_sky + F_sky.T) / 2.0
    det_F_sky = jnp.linalg.det(F_sky)
    cov_sky = jnp.array(
        [[F_sky[1, 1], -F_sky[0, 1]], [-F_sky[1, 0], F_sky[0, 0]]]
    ) / jnp.maximum(det_F_sky, 1e-30)
    det_cov = jnp.linalg.det(cov_sky)
    return (180 / jnp.pi) ** 2 * 2 * jnp.pi * jnp.sqrt(jnp.maximum(det_cov, 1e-30))


# =============================================================================
# NG15-like array
# =============================================================================
def load_ng15(path="data/ng15_psrs.csv"):
    """Load the ingested NG15 pulsar summary into a dict of numpy arrays."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            rows.append(line.strip().split(","))
    header, rows = rows[0], rows[1:]
    cat = {}
    for j, col in enumerate(header):
        vals = [r[j] for r in rows]
        cat[col] = np.array(vals) if col == "name" else np.array(vals, dtype=float)
    return cat


def build_ng15_batch(cat, n_toas=200, sigma_pad=SIGMA_PAD):
    """NG15 array on a common uniform TOA grid with weight-masked baselines.

    Pulsar i is 'active' only within its real observing window
    [t_first_i, t_last_i]; inactive epochs get sigma_pad (zero statistical
    weight). Active epochs get an effective white-noise level
        sigma_eff_i = wrms_i * sqrt(n_active_i / n_epochs_i),
    which preserves each pulsar's total statistical weight
        sum(1/sigma^2) = n_epochs_i / wrms_i^2,
    i.e. the information content implied by its real epoch count and total
    rms, distributed over its real observing window.

    Returns (PulsarBatch, tref) with tref = grid start.
    """
    t0 = float(cat["t_first"].min())
    t1 = float(cat["t_last"].max())
    grid = np.linspace(t0, t1, n_toas)
    n_psr = len(cat["name"])

    active = (grid[None, :] >= cat["t_first"][:, None]) & (
        grid[None, :] <= cat["t_last"][:, None]
    )
    n_active = active.sum(axis=1)
    if np.any(n_active < 2):
        raise ValueError("a pulsar has <2 active grid epochs; increase n_toas")
    sigma_eff = cat["wrms_s"] * np.sqrt(n_active / cat["n_epochs"])
    toaerrs = np.where(active, sigma_eff[:, None], sigma_pad)
    toas = np.tile(grid, (n_psr, 1))
    pdist = np.stack([cat["pdist_mean"], cat["pdist_sigma"]], axis=1)

    batch = uvf.PulsarBatch(
        tuple(cat["name"]),
        jnp.asarray(toas), jnp.asarray(toaerrs),
        jnp.asarray(cat["ra"]), jnp.asarray(cat["dec"]),
        jnp.asarray(pdist),
    )
    return batch, t0


def scale_pdist_sigma(batch, f):
    """New PulsarBatch with all distance-prior widths multiplied by f."""
    pdist = batch.pdist * jnp.array([1.0, f])
    return uvf.PulsarBatch(batch.names, batch.toas, batch.toaerrs,
                           batch.ra, batch.dec, pdist, pos=batch.pos)


# =============================================================================
# Fisher construction (SNR=10 calibration + priors), arbitrary batch
# =============================================================================
@partial(jax.jit, static_argnums=(2, 3))
def compute_total_fisher_fwd(pars_vec, pulsar_batch, pterm=True,
                             phase_free=False, tref=0.0):
    """Forward-mode twin of uvf.compute_total_fisher (same J^T W J, same
    signal model). jacrev materializes cotangent batches of shape
    (N_psrs*N_toas, N_psrs, N_toas) -- ~1.4 GB for the 67x200 NG15 array --
    whereas for these tall Jacobians (outputs >> parameters) jacfwd needs
    only (N_params, N_psrs, N_toas). Equivalence to the reverse-mode
    production implementation is verified numerically (rev == fwd to ~1e-15)."""

    def model_fn(p):
        return uvf.get_delay_batch(p, pulsar_batch, pterm, phase_free, tref=tref)

    J = jax.jacfwd(model_fn)(pars_vec)  # (N_psrs, N_toas, N_params)
    J_weighted = J / pulsar_batch.toaerrs[:, :, None]
    J_flat = J_weighted.reshape(-1, J.shape[-1])
    return J_flat.T @ J_flat


def calibrated_fisher(batch, base8, tref, phase_free=False, snr_target=10.0,
                      use_fwd=False):
    """SNR-calibrated Fisher with priors, mirroring the audited notebook
    helpers (compute_fisher_for_angle / compute_fisher_comparison).

    Priors: none on the 8 CW parameters; unit Gaussian on each normalized
    distance parameter (physical width = pdist[:,1]); none on the free
    pulsar phases in the decoupled case.

    use_fwd selects the forward-mode Jacobian (memory-efficient for large
    arrays; see compute_total_fisher_fwd). Default False = the audited
    production reverse-mode path.

    Returns (F_with_priors, pv_full, snr_before_calibration).
    """
    fisher_fn = compute_total_fisher_fwd if use_fwd else uvf.compute_total_fisher
    n = batch.pos.shape[0]
    pv = jnp.concatenate([base8, jnp.zeros(n)])
    snr = jnp.sqrt(jnp.maximum(
        uvf.computer_snr2_batch(pv, batch, pterm=True, phase_free=False, tref=tref),
        1e-12))
    pv = pv.at[5].set(jnp.log10(snr * 10 ** base8[5] / snr_target))
    if phase_free:
        pph = uvf.compute_pulsar_phases_batch(pv, batch, tref=tref)
        pv_full = jnp.concatenate([pv, pph])
        prior = jnp.concatenate([jnp.zeros(8), jnp.ones(n), jnp.zeros(n)])
    else:
        pv_full = pv
        prior = jnp.concatenate([jnp.zeros(8), jnp.ones(n)])
    F = fisher_fn(pv_full, batch, pterm=True, phase_free=phase_free, tref=tref)
    return F + jnp.diag(prior), pv_full, snr


def per_pulsar_snr2(batch, pv_full, tref, phase_free=False):
    res = uvf.get_delay_batch(pv_full, batch, pterm=True,
                              phase_free=phase_free, tref=tref)
    return jnp.sum((res / batch.toaerrs) ** 2, axis=1)


# =============================================================================
# Study R1: ring sweeps at arbitrary mean pulsar distance L0
# =============================================================================
@partial(jax.jit, static_argnums=(3, 4))
def _ring_area(sigma_d, mean_d, angle_deg, n_psrs, phase_free,
               base8, toas, toaerrs, tref):
    pd = {"cw_costheta": base8[0], "cw_phi": base8[1]}
    batch = uvf.pulsar_ring_generator_vmap(
        pd, ang_radius=angle_deg, npsrs=n_psrs, toas=toas, toaerrs=toaerrs,
        pdist=[mean_d, sigma_d])
    F, _, _ = calibrated_fisher(batch, base8, tref, phase_free=phase_free)
    return compute_sky_area(F, marginalized=True)


def ring_sweep_area(sigma_ds, mean_d, angle_deg, n_psrs=20, phase_free=False,
                    base8=None, toas=None, toaerrs=None, tref=None):
    """Marginalized dOmega for an array of sigma_L values at mean distance
    mean_d [kpc] and ring radius angle_deg. Paper-fiducial timing defaults."""
    base8 = pars_vec() if base8 is None else base8
    toas = PAPER_TOAS if toas is None else toas
    toaerrs = PAPER_TOAERRS if toaerrs is None else toaerrs
    tref = PAPER_T0 if tref is None else tref
    fn = jax.vmap(lambda s: _ring_area(s, mean_d, angle_deg, n_psrs,
                                       phase_free, base8, toas, toaerrs, tref))
    return fn(jnp.asarray(sigma_ds))


# =============================================================================
# Study R2: sky positions and Galactic-coordinate maps
# =============================================================================
@partial(jax.jit, static_argnums=(5,))
def area_for_pos(costheta, phi, base8, batch, tref, phase_free=False):
    """Marginalized dOmega for a GW source at (costheta, phi), SNR calibrated
    to 10; generalizes the notebook's compute_fisher_for_pos. phase_free
    selects the decoupled model (free per-pulsar phases, no prior). Uses the
    forward-mode Fisher (rev/fwd equivalence verified numerically)."""
    b8 = base8.at[0].set(costheta).at[1].set(phi)
    F, _, _ = calibrated_fisher(batch, b8, tref, phase_free=phase_free,
                                use_fwd=True)
    return compute_sky_area(F, marginalized=True)


def sky_map(batch, tref, base8=None, nside=16, chunk=64, frame="galactic",
            phase_free=False):
    """All-sky map of marginalized dOmega over a HEALPix grid defined in
    `frame` ('galactic' or 'equatorial'). Returns the map (RING ordering in
    that frame). Chunked vmap in the Study-2 pattern. phase_free selects the
    decoupled model (8 + 2N parameters; use a smaller chunk for memory)."""
    import healpy as hp

    base8 = pars_vec() if base8 is None else base8
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    if frame == "galactic":
        ra, dec = galactic_to_radec(phi, np.pi / 2 - theta)
        costheta, cw_phi = np.sin(dec), ra
    else:
        costheta, cw_phi = np.cos(theta), phi

    n_chunks = (npix + chunk - 1) // chunk
    pad = n_chunks * chunk - npix
    ct = jnp.asarray(np.pad(costheta, (0, pad)))
    ph = jnp.asarray(np.pad(cw_phi, (0, pad)))

    def do_chunk(io):
        c, p = io
        return jax.vmap(lambda a, b: area_for_pos(a, b, base8, batch, tref,
                                                  phase_free))(c, p)

    omega = jax.lax.map(do_chunk, (ct.reshape(n_chunks, chunk),
                                   ph.reshape(n_chunks, chunk)))
    return np.asarray(omega.reshape(-1)[:npix])


def radec_to_galactic(ra, dec):
    """Equatorial (ICRS) [rad] -> Galactic (l, b) [rad]."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(ra=np.asarray(ra) * u.rad, dec=np.asarray(dec) * u.rad,
                 frame="icrs").galactic
    return c.l.rad, c.b.rad


def galactic_to_radec(l, b):
    """Galactic (l, b) [rad] -> equatorial ICRS (ra, dec) [rad]."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(l=np.asarray(l) * u.rad, b=np.asarray(b) * u.rad,
                 frame="galactic").icrs
    return c.ra.rad, c.dec.rad
