from __future__ import annotations
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .config import Config
from .genome import Genome, E, prune_isolated


# ── MutationRates ─────────────────────────────────────────────────────────────

@dataclass
class MutationRates:
    """Per-operator mutation intensities.  Set any rate to 0 to disable."""
    # Parameter mutations
    weight_sigma:    float = 0.1
    tau_sigma:       float = 0.1
    bias_sigma:      float = 0.1
    position_sigma:  float = 0.05
    # Type flip
    type_flip_prob:  float = 0.05
    # Structural mutations (applied with these independent probabilities per genome)
    add_node_prob:    float = 0.05
    remove_node_prob: float = 0.05
    add_edge_prob:    float = 0.1
    remove_edge_prob: float = 0.1


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tau_bounds(cfg: Config):
    """Return (lo, hi) arrays [3] for type-indexed tau clamping."""
    lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    return lo, hi


def _io_protect(cfg: Config) -> jnp.ndarray:
    """Bool mask [N_max]: True for I/O slots that must never be structurally mutated."""
    N = cfg.N_max
    return jnp.zeros(N, dtype=bool).at[:cfg.n_in].set(True).at[N - cfg.n_out:].set(True)


def _hidden_slots(cfg: Config) -> jnp.ndarray:
    """Bool mask [N_max]: True for hidden neuron slots."""
    N = cfg.N_max
    return jnp.zeros(N, dtype=bool).at[cfg.n_in: N - cfg.n_out].set(True)


def _random_eligible_1d(key: jax.Array, eligible: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Randomly select one True index from a 1-D bool mask.

    Returns (slot, any_eligible).  When no slot is eligible, slot=0 but
    any_eligible=False — callers guard the result with any_eligible.
    """
    noise  = jax.random.uniform(key, eligible.shape)
    scores = jnp.where(eligible, noise, -1.0)
    slot   = jnp.argmax(scores)
    return slot, jnp.any(eligible)


# ── Parameter mutations ───────────────────────────────────────────────────────

def perturb_weights(key: jax.Array, genome: Genome, cfg: Config, *, sigma: float = 0.1) -> Genome:
    """Gaussian perturbation of weight magnitudes; clamped to non-negative."""
    noise = jax.random.normal(key, genome.weight_matrix.shape) * sigma
    w     = jnp.maximum(0.0, genome.weight_matrix + noise)
    return replace(genome, weight_matrix=w)


def perturb_tau(key: jax.Array, genome: Genome, cfg: Config, *, sigma: float = 0.1) -> Genome:
    """Gaussian perturbation of time constants; clamped to per-type range."""
    lo, hi = _tau_bounds(cfg)
    noise  = jax.random.normal(key, genome.tau.shape) * sigma
    tau    = jnp.clip(genome.tau + noise, lo[genome.neuron_type], hi[genome.neuron_type])
    return replace(genome, tau=tau)


def perturb_bias(key: jax.Array, genome: Genome, cfg: Config, *, sigma: float = 0.1) -> Genome:
    """Gaussian perturbation of per-neuron biases."""
    noise = jax.random.normal(key, genome.bias.shape) * sigma
    return replace(genome, bias=genome.bias + noise)


def perturb_position(key: jax.Array, genome: Genome, cfg: Config, *, sigma: float = 0.05) -> Genome:
    """Gaussian perturbation of spatial positions; clamped to the unit square."""
    noise    = jax.random.normal(key, genome.position.shape) * sigma
    position = jnp.clip(genome.position + noise, 0.0, 1.0)
    return replace(genome, position=position)


# ── Type flip ─────────────────────────────────────────────────────────────────

def type_flip(
    key: jax.Array,
    genome: Genome,
    cfg: Config,
    *,
    flip_prob: float = 0.05,
) -> Genome:
    """
    For each hidden neuron, flip its type with probability flip_prob.

    I/O slots are protected.  After flipping, tau is re-clamped to the new
    type's range (if it was already in range it stays unchanged).
    """
    k1, k2 = jax.random.split(key)
    N = cfg.N_max

    new_types  = jax.random.randint(k1, (N,), 0, 3, dtype=jnp.uint8)
    flip_where = (jax.random.uniform(k2, (N,)) < flip_prob) & ~_io_protect(cfg)

    neuron_type = jnp.where(flip_where, new_types, genome.neuron_type)

    lo, hi = _tau_bounds(cfg)
    tau = jnp.clip(genome.tau, lo[neuron_type], hi[neuron_type])

    return replace(genome, neuron_type=neuron_type, tau=tau)


# ── Structural mutations ──────────────────────────────────────────────────────

def add_node(key: jax.Array, genome: Genome, cfg: Config) -> Genome:
    """
    Activate the first free hidden slot, initialise its fields, and wire it
    with one random incoming edge and one random outgoing edge to/from
    existing active neurons.

    Two connections ensure the new node is part of a computation path
    (receives signal AND can propagate it) so prune_isolated never
    immediately removes it.  No-op when all slots are already active.
    """
    k_type, k_tau, k_pos, k_bias, k_src, k_dst, k_wi, k_wo = jax.random.split(key, 8)
    N = cfg.N_max

    eligible = _hidden_slots(cfg) & ~genome.active_mask
    any_free = jnp.any(eligible)
    slot     = jnp.argmax(eligible)  # first True; guarded by apply below

    # Initialise new node parameters
    new_type = jax.random.randint(k_type, (), 0, 3, dtype=jnp.uint8)
    lo, hi   = _tau_bounds(cfg)
    new_tau  = lo[new_type] + jax.random.uniform(k_tau) * (hi[new_type] - lo[new_type])
    new_pos  = jax.random.uniform(k_pos, (2,))
    new_bias = jax.random.normal(k_bias) * 0.1

    # Pick one source (src → slot) and one destination (slot → dst)
    src, any_src = _random_eligible_1d(k_src, genome.active_mask)
    dst, any_dst = _random_eligible_1d(k_dst, genome.active_mask)
    wi = jnp.abs(jax.random.normal(k_wi)) * 0.5 + 0.01
    wo = jnp.abs(jax.random.normal(k_wo)) * 0.5 + 0.01

    new_edges   = (genome.edge_mask
                   .at[src, slot].set(True)
                   .at[slot, dst].set(True))
    new_weights = (genome.weight_matrix
                   .at[src, slot].set(wi)
                   .at[slot, dst].set(wo))

    apply = any_free & any_src & any_dst

    active_mask   = jnp.where(apply, genome.active_mask.at[slot].set(True),     genome.active_mask)
    neuron_type   = jnp.where(apply, genome.neuron_type.at[slot].set(new_type), genome.neuron_type)
    tau           = jnp.where(apply, genome.tau.at[slot].set(new_tau),           genome.tau)
    bias          = jnp.where(apply, genome.bias.at[slot].set(new_bias),         genome.bias)
    position      = jnp.where(apply, genome.position.at[slot].set(new_pos),      genome.position)
    edge_mask     = jnp.where(apply, new_edges,   genome.edge_mask)
    weight_matrix = jnp.where(apply, new_weights, genome.weight_matrix)

    return replace(genome, active_mask=active_mask, neuron_type=neuron_type,
                   tau=tau, bias=bias, position=position,
                   edge_mask=edge_mask, weight_matrix=weight_matrix)


def remove_node(key: jax.Array, genome: Genome, cfg: Config) -> Genome:
    """
    Mask off a randomly selected active hidden neuron and clear its edges.
    No-op when no removable hidden neuron exists.
    """
    N = cfg.N_max

    eligible          = _hidden_slots(cfg) & genome.active_mask
    slot, any_remove  = _random_eligible_1d(key, eligible)

    # Clear the node and all its edges
    new_active = genome.active_mask.at[slot].set(False)
    new_edges  = genome.edge_mask.at[slot, :].set(False).at[:, slot].set(False)

    active_mask = jnp.where(any_remove, new_active, genome.active_mask)
    edge_mask   = jnp.where(any_remove, new_edges,  genome.edge_mask)

    return replace(genome, active_mask=active_mask, edge_mask=edge_mask)


def add_edge(key: jax.Array, genome: Genome, cfg: Config) -> Genome:
    """
    Add a randomly selected missing edge between two active neurons.
    Initialises the weight at that position with a fresh positive sample.
    No-op when all active-neuron pairs already have edges.
    """
    k_slot, k_w = jax.random.split(key)
    N = cfg.N_max

    # Eligible: active-to-active pair with no existing edge
    eligible      = genome.active_mask[:, None] & genome.active_mask[None, :] & ~genome.edge_mask
    eligible_flat = eligible.reshape(-1)
    slot_flat, any_add = _random_eligible_1d(k_slot, eligible_flat)

    new_weight = jnp.abs(jax.random.normal(k_w)) * 0.5 + 0.01  # guaranteed > 0

    new_edge_flat   = genome.edge_mask.reshape(-1).at[slot_flat].set(True)
    new_weight_flat = genome.weight_matrix.reshape(-1).at[slot_flat].set(new_weight)

    edge_mask     = jnp.where(any_add, new_edge_flat.reshape(N, N),   genome.edge_mask)
    weight_matrix = jnp.where(any_add, new_weight_flat.reshape(N, N), genome.weight_matrix)

    return replace(genome, edge_mask=edge_mask, weight_matrix=weight_matrix)


def remove_edge(key: jax.Array, genome: Genome, cfg: Config) -> Genome:
    """
    Remove a randomly selected active edge between two active neurons.

    An edge i→j is only eligible for removal if doing so would not strand
    either endpoint as a source-only or sink-only hidden neuron:
      - source i (if hidden) must have ≥2 outgoing active edges
      - dest   j (if hidden) must have ≥2 incoming active edges

    I/O neurons are exempt from this constraint (they are never deactivated
    by prune_isolated regardless of edge count).  No-op when no eligible edge
    exists.
    """
    N = cfg.N_max

    active_pairs = genome.active_mask[:, None] & genome.active_mask[None, :]
    active_edges = genome.edge_mask & active_pairs
    out_degree   = jnp.sum(active_edges, axis=1)   # [N] outgoing edges per neuron
    in_degree    = jnp.sum(active_edges, axis=0)   # [N] incoming edges per neuron

    hidden = _hidden_slots(cfg)  # [N] True for hidden neurons only

    # Removing i→j would strand i if i is hidden and it's its last outgoing edge
    # Removing i→j would strand j if j is hidden and it's its last incoming edge
    would_strand_source = hidden[:, None] & (out_degree[:, None] <= 1)
    would_strand_sink   = hidden[None, :] & (in_degree[None, :] <= 1)

    eligible      = active_edges & ~would_strand_source & ~would_strand_sink
    eligible_flat = eligible.reshape(-1)
    slot_flat, any_remove = _random_eligible_1d(key, eligible_flat)

    new_edge_flat = genome.edge_mask.reshape(-1).at[slot_flat].set(False)
    edge_mask     = jnp.where(any_remove, new_edge_flat.reshape(N, N), genome.edge_mask)

    return replace(genome, edge_mask=edge_mask)


# ── Combined operator ─────────────────────────────────────────────────────────

def mutate(key: jax.Array, genome: Genome, cfg: Config, rates: MutationRates) -> Genome:
    """
    Apply all mutation operators in sequence.

    Parameter mutations are always applied (sigma=0 is a no-op).
    Structural mutations are applied with independent Bernoulli draws.
    """
    k = jax.random.split(key, 13)

    # Parameter mutations
    g = perturb_weights( k[0], genome, cfg, sigma=rates.weight_sigma)
    g = perturb_tau(     k[1], g,      cfg, sigma=rates.tau_sigma)
    g = perturb_bias(    k[2], g,      cfg, sigma=rates.bias_sigma)
    g = perturb_position(k[3], g,      cfg, sigma=rates.position_sigma)
    g = type_flip(       k[4], g,      cfg, flip_prob=rates.type_flip_prob)

    # Structural mutations — each applied with independent probability
    do_add_node    = jax.random.uniform(k[5])  < rates.add_node_prob
    do_remove_node = jax.random.uniform(k[6])  < rates.remove_node_prob
    do_add_edge    = jax.random.uniform(k[7])  < rates.add_edge_prob
    do_remove_edge = jax.random.uniform(k[8])  < rates.remove_edge_prob

    g = jax.lax.cond(do_add_node,    lambda g_: add_node(   k[9],  g_, cfg), lambda g_: g_, g)
    g = jax.lax.cond(do_remove_node, lambda g_: remove_node(k[10], g_, cfg), lambda g_: g_, g)
    g = jax.lax.cond(do_add_edge,    lambda g_: add_edge(   k[11], g_, cfg), lambda g_: g_, g)
    g = jax.lax.cond(do_remove_edge, lambda g_: remove_edge(k[12], g_, cfg), lambda g_: g_, g)

    return prune_isolated(g, cfg)  # no-op when genome is already valid
