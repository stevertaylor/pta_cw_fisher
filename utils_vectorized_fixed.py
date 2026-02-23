# -*- coding: utf-8 -*-
"""
Vectorized utility functions for JAX optimization.
Adapted from utils.py to support Struct-of-Arrays (SoA) and batch operations.

FIXED VERSION: Addresses two issues from physics review:
  1. omega_p0 convention mismatch between get_delay_batch and
     compute_pulsar_phases_batch — now both use the analytic formula.
  2. Added tref parameter to get_delay_batch (and dependents) for
     full parity with utils.py CW_Signal.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from functools import partial
import constants as const
from matplotlib.patches import Ellipse

# =============================================================================
# Vectorized Core Classes (Struct-of-Arrays)
# =============================================================================

@register_pytree_node_class
class PulsarBatch:
    """
    A batch of N pulsars treated as a single PyTree node.
    All attributes are arrays with leading dimension (N_psrs, ...).
    """
    def __init__(self, names, toas, toaerrs, ra, dec, pdist, pos=None):
        self.names = names          # List/Array of names (metadata, not usually tracing)
        self.toas = toas            # Shape (N_psrs, N_toas)
        self.toaerrs = toaerrs      # Shape (N_psrs, N_toas)
        self.ra = ra                # Shape (N_psrs,)
        self.dec = dec              # Shape (N_psrs,)
        self.pdist = pdist          # Shape (N_psrs, 2)
        
        # Precompute positions if not provided
        if pos is None:
            self.pos = jnp.stack([
                jnp.cos(dec) * jnp.cos(ra),
                jnp.cos(dec) * jnp.sin(ra),
                jnp.sin(dec),
            ], axis=-1) # Shape (N_psrs, 3)
        else:
            self.pos = pos

    def __repr__(self):
        return f"PulsarBatch(N={self.pos.shape[0]})"

    def tree_flatten(self):
        # We treat names as auxiliary data (static)
        children = (self.toas, self.toaerrs, self.ra, self.dec, self.pdist, self.pos)
        aux_data = (self.names,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        toas, toaerrs, ra, dec, pdist, pos = children
        names = aux_data[0]
        return cls(names, toas, toaerrs, ra, dec, pdist, pos=pos)


# =============================================================================
# Vectorized Generators
# =============================================================================

def pulsar_ring_generator_vmap(pars, ang_radius=10.0, npsrs=20, toas=None, toaerrs=None, pdist=None):
    """
    Vectorized version of pulsar_ring_generator that creates a PulsarBatch.
    Can be JIT-compiled if npsrs is static.
    """
    n_ring = npsrs
    alpha = jnp.deg2rad(ang_radius)
    phi_ring = jnp.linspace(0.1, 2*jnp.pi+0.1, n_ring, endpoint=False)

    # GW source coordinates
    dec0 = jnp.arcsin(pars['cw_costheta'])
    ra0  = (pars['cw_phi'] + jnp.pi) % (2*jnp.pi) - jnp.pi

    # ring coordinates
    dec_ring = jnp.arcsin(
        jnp.sin(dec0) * jnp.cos(alpha) +
        jnp.cos(dec0) * jnp.sin(alpha) * jnp.cos(phi_ring)
    )
    ra_ring = ra0 + jnp.arctan2(
        jnp.sin(alpha) * jnp.sin(phi_ring),
        jnp.cos(dec0) * jnp.cos(alpha) - jnp.sin(dec0) * jnp.sin(alpha) * jnp.cos(phi_ring)
    )
    ra_ring = (ra_ring + jnp.pi) % (2*jnp.pi) - jnp.pi

    # Broadcast toas and toaerrs to (N_psrs, N_toas)
    # Assumes toas/toaerrs are 1D arrays for a single pulsar template
    N_toas = toas.shape[0]
    toas_batch = jnp.tile(toas, (npsrs, 1))
    toaerrs_batch = jnp.tile(toaerrs, (npsrs, 1))
    
    # Broadcast pdist to (N_psrs, 2)
    pdist_batch = jnp.tile(jnp.array(pdist), (npsrs, 1))
    
    # Dummy names (strings not traced by JAX)
    names = [f"PSR{i:02d}" for i in range(npsrs)]

    return PulsarBatch(names, toas_batch, toaerrs_batch, ra_ring, dec_ring, pdist_batch)


# =============================================================================
# Vectorized Physics & Fisher Calculation
# =============================================================================

@jax.jit
def create_gw_antenna_pattern_vmap(pos_batch, gwtheta, gwphi):
    """
    Computes antenna patterns for a batch of pulsars.
    pos_batch: (N_psrs, 3)
    Returns: (fplus, fcross, cosMu) each shape (N_psrs,)
    """
    m = jnp.array([jnp.sin(gwphi), -jnp.cos(gwphi), 0.0])
    n = jnp.array([
        -jnp.cos(gwtheta) * jnp.cos(gwphi),
        -jnp.cos(gwtheta) * jnp.sin(gwphi),
        jnp.sin(gwtheta)
    ])
    omhat = jnp.array([
        -jnp.sin(gwtheta) * jnp.cos(gwphi),
        -jnp.sin(gwtheta) * jnp.sin(gwphi),
        -jnp.cos(gwtheta)
    ])

    # Dot products: (N_psrs, 3) @ (3,) -> (N_psrs,)
    m_pos = jnp.dot(pos_batch, m)
    n_pos = jnp.dot(pos_batch, n)
    omhat_pos = jnp.dot(pos_batch, omhat)

    # Protect against singularity when pulsar is opposite to GW source
    # (omhat·pos ≈ -1 causes denominator → 0)
    denom = jnp.maximum(1 + omhat_pos, 1e-10)
    
    fplus = 0.5 * (m_pos**2 - n_pos**2) / denom
    fcross = (m_pos * n_pos) / denom
    cosMu = -omhat_pos

    return fplus, fcross, cosMu

@partial(jax.jit, static_argnums=(2, 3))
def get_delay_batch(pars_vec, pulsar_batch, pterm=True, phase_free=False, tref=0.0):
    """
    Computes residuals for a batch of pulsars.
    
    pars_vec: Array of CW parameters [costheta, phi, cosinc, log10_Mc, log10_fgw, log10_dist, phase0, psi]
              PLUS [pdist_1, ..., pdist_N] if pterm=True
              PLUS [pphase_1, ..., pphase_N] if phase_free=True
    tref: Reference time for phase and frequency [s]. Default 0.
              
    Returns:
        residuals: Shape (N_psrs, N_toas)
    """
    # 1. Parse common CW parameters (first 8)
    cw_costheta, cw_phi, cw_cosinc, cw_log10_Mc, cw_log10_fgw, cw_log10_dist, cw_phase0, cw_psi = pars_vec[:8]
    
    n_psrs = pulsar_batch.pos.shape[0]
    
    # 2. Convert units
    mc = 10**cw_log10_Mc * const.Tsun
    dist = 10**cw_log10_dist * const.Mpc / const.c
    fgw = 10**cw_log10_fgw
    # Clamp to valid domain to prevent NaN from arccos
    gwtheta = jnp.arccos(jnp.clip(cw_costheta, -1.0, 1.0))
    inc = jnp.arccos(jnp.clip(cw_cosinc, -1.0, 1.0))
    
    # 3. Antenna Patterns (N_psrs,)
    fplus, fcross, cosMu = create_gw_antenna_pattern_vmap(pulsar_batch.pos, gwtheta, cw_phi)
    
    # Enable broadcasting for fplus/fcross against (N_psrs, N_toas)
    fplus = fplus[:, None]
    fcross = fcross[:, None]
    cosMu = cosMu[:, None] # Needed for pterm
    
    # 4. Evolution setup
    # Subtract reference time from TOAs (matching utils.py CW_Signal)
    toas = pulsar_batch.toas - tref
    w0 = jnp.pi * fgw
    phase0 = cw_phase0 / 2
    
    # Frequency evolution function (inline full_evolve)
    def evolve(t):
        # Protect against negative/zero evolution factor
        # (happens near coalescence or for very large t)
        evolution_factor = jnp.maximum(1 - 256/5 * mc**(5/3) * w0**(8/3) * t, 1e-10)
        omega = w0 * evolution_factor**(-3/8)
        phase = phase0 + 1/32 * mc**(-5/3) * (w0**(-5/3) - omega**(-5/3))
        return omega, phase

    omega, phase = evolve(toas)
    
    # Earth term coefficients (N_psrs, N_toas)
    At = -0.5 * jnp.sin(2 * phase) * (3 + jnp.cos(2 * inc))
    Bt = 2 * jnp.cos(2 * phase) * jnp.cos(inc)
    alpha = mc**(5.0/3.0) / (dist * omega**(1.0/3.0))
    
    rplus = alpha * (-At * jnp.cos(2 * cw_psi) + Bt * jnp.sin(2 * cw_psi))
    rcross = alpha * (At * jnp.sin(2 * cw_psi) + Bt * jnp.cos(2 * cw_psi))
    
    res = -fplus * rplus - fcross * rcross
    
    if pterm:
        # Extract pulsar distance parameters from pars_vec
        # Indices 8 to 8+N_psrs are pulsar distances (normalized/parameterized)
        pdist_params = pars_vec[8 : 8+n_psrs]
        
        # Calculate physical distances
        # pulsar_batch.pdist has shape (N_psrs, 2) -> [mean, std]
        # p_dist = mean + std * param
        p_dist_kpc = pulsar_batch.pdist[:, 0] + pulsar_batch.pdist[:, 1] * pdist_params
        p_dist = p_dist_kpc[:, None] * const.kpc / const.c # (N_psrs, 1)
        
        # Pulsar term time
        tp = toas - p_dist * (1 - cosMu)
        omega_p, phase_p_evolved = evolve(tp)
        
        # Pulsar term phase handling
        if phase_free:
            # Indices 8+N to 8+2N are phases
            pphase_params = pars_vec[8+n_psrs : 8+2*n_psrs] # (N_psrs,)
            
            # Compute omega_p0 analytically at t_earth=0 (t_pulsar = -p_dist*(1-cosMu))
            # This is consistent with compute_pulsar_phases_batch.
            tp0 = -p_dist * (1 - cosMu)  # (N_psrs, 1)
            evolution_factor_p0 = jnp.maximum(1 - 256/5 * mc**(5/3) * w0**(8/3) * tp0, 1e-10)
            omega_p0 = w0 * evolution_factor_p0**(-3/8)  # (N_psrs, 1)
            
            # Override phase_p
            # phase_p = phase0 + p_phase + ...
            term2 = 1/32 * mc**(-5/3) * (omega_p0**(-5/3) - omega_p**(-5/3))
            phase_p = phase0 + pphase_params[:, None] + term2
            
        else:
            phase_p = phase_p_evolved

        At_p = -0.5 * jnp.sin(2 * phase_p) * (3 + jnp.cos(2 * inc))
        Bt_p = 2 * jnp.cos(2 * phase_p) * jnp.cos(inc)
        alpha_p = mc**(5.0/3.0) / (dist * omega_p**(1.0/3.0))
        
        rplus_p = alpha_p * (-At_p * jnp.cos(2 * cw_psi) + Bt_p * jnp.sin(2 * cw_psi))
        rcross_p = alpha_p * (At_p * jnp.sin(2 * cw_psi) + Bt_p * jnp.cos(2 * cw_psi))
        
        res = res + fplus * rplus_p + fcross * rcross_p

    return res # Shape (N_psrs, N_toas)


@partial(jax.jit, static_argnums=(2, 3))
def computer_snr2_batch(pars_vec, pulsar_batch, pterm=True, phase_free=False, tref=0.0):
    """Compute combined SNR^2 for the entire batch"""
    # Get all residuals: (N_psrs, N_toas)
    res = get_delay_batch(pars_vec, pulsar_batch, pterm, phase_free, tref=tref)
    
    # Whiten with errors: res / sigma
    weighted = res / pulsar_batch.toaerrs
    
    # Square and sum over all pulsars and TOAs
    return jnp.sum(weighted**2)


@partial(jax.jit, static_argnums=(2, 3))
def compute_total_fisher(pars_vec, pulsar_batch, pterm=True, phase_free=False, tref=0.0):
    """
    Compute the total Fisher matrix by summing individual pulsar Fisher matrices.
    
    1. Jacobian J: d(residuals)/d(params) -> Shape (N_psrs, N_toas, N_params)
    2. Block-diagonal approximation: We sum J_i.T @ J_i / sigma_i^2
    """
    
    # Define wrapper to take params and return (N_psrs, N_toas) residuals
    def model_fn(p):
        return get_delay_batch(p, pulsar_batch, pterm, phase_free, tref=tref)
    
    # Compute Jacobian of the entire batch output w.r.t parameters
    # Jac shape: (N_psrs, N_toas, N_params)
    # Note: simple jacrev works because output is array
    J = jax.jacrev(model_fn)(pars_vec)
    
    # Weight by inverse sigma
    # weights shape: (N_psrs, N_toas, 1) to broadcast over params
    weights = (1.0 / pulsar_batch.toaerrs[:, :, None])
    J_weighted = J * weights
    
    # Flatten pulsar and TOA dimensions for the dot product
    # (N_psrs * N_toas, N_params)
    N_psrs, N_toas, N_params = J.shape
    J_flat = J_weighted.reshape(N_psrs * N_toas, N_params)
    
    # Fisher = J.T @ J
    F = J_flat.T @ J_flat
    
    return F


@jax.jit
def compute_pulsar_phases_batch(pars_vec, pulsar_batch, tref=0.0):
    """
    Computes the linked pulsar-term phases for a batch of pulsars.
    
    pars_vec: Array of CW parameters [costheta, phi, cosinc, log10_Mc, log10_fgw, log10_dist, phase0, psi]
              PLUS [pdist_1, ..., pdist_N]
    tref: Reference time for phase and frequency [s]. Default 0.
    
    Returns:
        phases: Shape (N_psrs,) linked phases at t=tref (relative to phase0).
    """
    cw_costheta, cw_phi, cw_cosinc, cw_log10_Mc, cw_log10_fgw, cw_log10_dist, cw_phase0, cw_psi = pars_vec[:8]
    n_psrs = pulsar_batch.pos.shape[0]
    
    mc = 10**cw_log10_Mc * const.Tsun
    fgw = 10**cw_log10_fgw
    w0 = jnp.pi * fgw
    
    gwtheta = jnp.arccos(cw_costheta)
    _, _, cosMu = create_gw_antenna_pattern_vmap(pulsar_batch.pos, gwtheta, cw_phi)
    
    pdist_params = pars_vec[8 : 8+n_psrs]
    p_dist_kpc = pulsar_batch.pdist[:, 0] + pulsar_batch.pdist[:, 1] * pdist_params
    p_dist = p_dist_kpc * const.kpc / const.c
    
    tp0 = -p_dist * (1 - cosMu)
    omega_p0 = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * tp0)**(-3/8)
    
    pphase = 1/32 * mc**(-5/3) * (w0**(-5/3) - omega_p0**(-5/3))
    
    return pphase