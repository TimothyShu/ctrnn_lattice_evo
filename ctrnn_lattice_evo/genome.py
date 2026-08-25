from __future__ import annotations
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .config import Config

# Neuron type constants
E   = 0  # excitatory
FSI = 1  # fast-spiking inhibitory
SII = 2  # slow-integrating inhibitory


@dataclass
class Genome:
    """
    Fixed-capacity padded genome for one CTRNN organism.

    All arrays have a leading N_max dimension so a population of P organisms
    is represented as a single batched Genome where every field has shape
    [P, N_max, ...], enabling jax.vmap across the population.

    Weight magnitudes are stored non-negative; Dale's law sign is derived from
    neuron_type at forward-pass time via effective_weights().
    """
    active_mask:   jnp.ndarray  # [N_max]         bool    — which slots are live
    neuron_type:   jnp.ndarray  # [N_max]          uint8   — 0=E, 1=FSI, 2=SII
    tau:           jnp.ndarray  # [N_max]          float32 — time constants (ms)
    bias:          jnp.ndarray  # [N_max]          float32
    position:      jnp.ndarray  # [N_max, 2]       float32 — spatial coords in [0,1]^2
    weight_matrix: jnp.ndarray  # [N_max, N_max]   float32 — non-negative magnitudes
    edge_mask:     jnp.ndarray  # [N_max, N_max]   bool    — structural connectivity


# Register as a JAX pytree so jit/vmap can look inside the dataclass.
jax.tree_util.register_pytree_node(
    Genome,
    lambda g: (
        [g.active_mask, g.neuron_type, g.tau, g.bias,
         g.position, g.weight_matrix, g.edge_mask],
        None,
    ),
    lambda _, children: Genome(*children),
)


def prune_isolated(genome: Genome, cfg: Config) -> Genome:
    """
    Deactivate hidden neurons that have no edges (incoming or outgoing) among
    active neurons.  I/O neurons are always kept regardless of connectivity.

    Called by random_genome to guarantee no isolated neurons at initialisation,
    and by mutate() after every structural mutation step.
    """
    active_pairs = genome.active_mask[:, None] & genome.active_mask[None, :]
    active_edges = genome.edge_mask & active_pairs
    has_in  = jnp.any(active_edges, axis=0)   # receives signal from another active neuron
    has_out = jnp.any(active_edges, axis=1)   # sends signal to another active neuron

    hidden      = jnp.zeros(cfg.N_max, dtype=bool).at[cfg.n_in: cfg.N_max - cfg.n_out].set(True)
    new_active  = genome.active_mask & ((has_in & has_out) | ~hidden)
    new_edges   = genome.edge_mask & new_active[:, None] & new_active[None, :]

    return replace(genome, active_mask=new_active, edge_mask=new_edges)


def random_genome(key: jax.Array, cfg: Config, n_active: int | None = None) -> Genome:
    """
    Initialise a single random genome.

    Input slots (first n_in) and output slots (last n_out) are always active
    and typed as excitatory. Hidden slots are randomly activated up to n_active.
    """
    if n_active is None:
        n_active = cfg.N_max // 2

    keys = jax.random.split(key, 7)

    # --- Activity mask ---
    fixed = cfg.n_in + cfg.n_out
    n_hidden_on = max(0, n_active - fixed)
    n_hidden = cfg.N_max - fixed
    perm = jax.random.permutation(keys[0], n_hidden)
    hidden_mask = jnp.zeros(n_hidden, dtype=bool).at[perm[:n_hidden_on]].set(True)
    active_mask = jnp.concatenate([
        jnp.ones(cfg.n_in,  dtype=bool),
        hidden_mask,
        jnp.ones(cfg.n_out, dtype=bool),
    ])

    # --- Neuron types ---
    neuron_type = jax.random.randint(keys[1], (cfg.N_max,), 0, 3, dtype=jnp.uint8)
    neuron_type = neuron_type.at[:cfg.n_in].set(E)
    neuron_type = neuron_type.at[-cfg.n_out:].set(E)

    # --- Time constants sampled within type-specific range ---
    tau_lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    tau_hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    u   = jax.random.uniform(keys[2], (cfg.N_max,))
    tau = tau_lo[neuron_type] + u * (tau_hi[neuron_type] - tau_lo[neuron_type])

    # --- Biases ---
    bias = jax.random.normal(keys[3], (cfg.N_max,)) * 0.1

    # --- Spatial positions ---
    position = jax.random.uniform(keys[4], (cfg.N_max, 2))

    # --- Weight magnitudes ---
    weight_matrix = jnp.abs(jax.random.normal(keys[5], (cfg.N_max, cfg.N_max)) * 0.5)

    # --- Edge mask at target density, only between active neurons ---
    candidate = jax.random.uniform(keys[6], (cfg.N_max, cfg.N_max)) < cfg.init_edge_density
    edge_mask = candidate & active_mask[:, None] & active_mask[None, :]

    return prune_isolated(Genome(
        active_mask=active_mask,
        neuron_type=neuron_type,
        tau=tau,
        bias=bias,
        position=position,
        weight_matrix=weight_matrix,
        edge_mask=edge_mask,
    ), cfg)


def effective_weights(genome: Genome) -> jnp.ndarray:
    """
    Compute W_eff [N_max, N_max] applying Dale's law, edge mask, and activity mask.

    W_eff[i, j] = weight from neuron j to neuron i, with sign determined by
    the type of source neuron j.
    """
    sign_vec = jnp.where(genome.neuron_type == E, 1.0, -1.0)  # [N_max]
    return (
        genome.weight_matrix
        * sign_vec[None, :]              # source neuron sign (Dale's law)
        * genome.edge_mask               # structural connectivity
        * genome.active_mask[:, None]    # silence inactive post-synaptic neurons
        * genome.active_mask[None, :]    # silence inactive pre-synaptic neurons
    )


def validate_genome(genome: Genome, cfg: Config) -> bool:
    """Assert all structural invariants. Returns True on success, raises on failure."""
    assert jnp.all(genome.weight_matrix >= 0),  "Negative weight magnitudes"
    assert jnp.all(genome.tau > 0),              "Non-positive time constants"
    assert jnp.all(genome.neuron_type <= 2),     "Invalid neuron type"
    assert jnp.all(genome.active_mask[:cfg.n_in]),  "Input neurons must be active"
    assert jnp.all(genome.active_mask[-cfg.n_out:]), "Output neurons must be active"

    active_pairs = genome.active_mask[:, None] & genome.active_mask[None, :]
    assert jnp.all(genome.edge_mask <= active_pairs), "Edge to or from an inactive neuron"

    tau_lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    tau_hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    in_range = (
        (genome.tau >= tau_lo[genome.neuron_type] - 1e-4) &
        (genome.tau <= tau_hi[genome.neuron_type] + 1e-4)
    )
    assert jnp.all(jnp.where(genome.active_mask, in_range, True)), \
        "Time constant out of type range for an active neuron"

    return True
