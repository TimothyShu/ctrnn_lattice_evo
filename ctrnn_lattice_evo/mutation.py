from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .config import Config
from .genome import Genome, E, prune_isolated

__all__ = [
    "MutationRates",
    "perturb_weights", "perturb_tau", "perturb_bias",
    "type_flip",
    "add_edge", "remove_edges",
    "add_node", "remove_node",
    "mutate",
]


# ── MutationRates ─────────────────────────────────────────────────────────────

@dataclass
class MutationRates:
    """Per-operator mutation intensities.  Set any rate to 0 to disable.

    `position_sigma` is gone — positions are a property of the lattice, not of
    the genome.
    """
    # Continuous parameter mutations (scaled by mutation_warmup_scale during
    # the penalty warm-up; evolution.py applies that scaling before calling
    # mutate, so these are the post-scaling values).
    weight_sigma: float = 0.1
    tau_sigma:    float = 0.1
    bias_sigma:   float = 0.1

    # Type flip, per hidden neuron
    type_flip_prob: float = 0.05

    # Edge addition: one edge per genome per generation, fired with this
    # probability.  Deliberately unmasked with respect to the lattice — a
    # long-range shortcut can be bought if it earns its cost.
    add_edge_prob: float = 0.1

    # Edge removal: an INDEPENDENT Bernoulli draw PER ACTIVE EDGE, applied
    # every generation.  This is the change that makes the experiment
    # runnable.
    #
    # ctrnn_evo's remove_edge deleted exactly one edge per genome per
    # generation (an argmax over noise), capping lineage removal at 1
    # edge/generation regardless of population size.  Against 1092 lattice
    # edges, 50-70% pruning needs 550-760 removal events — unreachable inside
    # 500 generations, and the failure is silent: the run completes, the logs
    # look fine, nothing prunes.
    #
    # At 0.003 a 1092-edge genome loses ~3.3 edges/generation, so ~300
    # generations of directional pressure can clear ~50%.
    remove_edge_p_per_edge: float = 0.003

    # Node operators — only meaningful for the sparse arm, and gated on
    # cfg.node_ops_enabled regardless of what is set here.  On a full lattice
    # add_node self-disables (no free slots) and remove_node would punch holes
    # in the substrate.
    add_node_prob:    float = 0.05
    remove_node_prob: float = 0.05


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tau_bounds(cfg: Config):
    lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    return lo, hi


def _io_protect(cfg: Config) -> jnp.ndarray:
    """[N_max] bool — I/O slots that must never be structurally mutated."""
    N = cfg.N_max
    return jnp.zeros(N, dtype=bool).at[:cfg.n_in].set(True).at[N - cfg.n_out:].set(True)


def _hidden_slots(cfg: Config) -> jnp.ndarray:
    """[N_max] bool — hidden neuron slots."""
    N = cfg.N_max
    return jnp.zeros(N, dtype=bool).at[cfg.n_in: N - cfg.n_out].set(True)


def _random_eligible_1d(key: jax.Array, eligible: jnp.ndarray):
    """Pick one True index from a 1-D bool mask.

    Returns (slot, any_eligible).  With nothing eligible, slot=0 but
    any_eligible=False — callers must guard on the flag.
    """
    noise  = jax.random.uniform(key, eligible.shape)
    scores = jnp.where(eligible, noise, -1.0)
    return jnp.argmax(scores), jnp.any(eligible)


def _no_self(cfg: Config) -> jnp.ndarray:
    return ~jnp.eye(cfg.N_max, dtype=bool)


# ── Continuous parameter mutations ────────────────────────────────────────────

def perturb_weights(key: jax.Array, genome: Genome, cfg: Config, *,
                    sigma: float = 0.1) -> Genome:
    """Gaussian perturbation of weight magnitudes, clamped non-negative.

    Magnitudes only — Dale's law sign comes from neuron_type at forward time,
    so a weight may never go negative here.
    """
    noise = jax.random.normal(key, genome.weight_matrix.shape) * sigma
    return replace(genome, weight_matrix=jnp.maximum(0.0, genome.weight_matrix + noise))


def perturb_tau(key: jax.Array, genome: Genome, cfg: Config, *,
                sigma: float = 0.1) -> Genome:
    """Gaussian perturbation of time constants, clamped to the per-type range.

    The clamp also keeps tau >= tau_min >= dt, which is the Euler stability
    condition — at tau_fsi_range[0]=1.0 against dt=0.5 the margin is only 2x.
    """
    lo, hi = _tau_bounds(cfg)
    noise  = jax.random.normal(key, genome.tau.shape) * sigma
    tau    = jnp.clip(genome.tau + noise, lo[genome.neuron_type], hi[genome.neuron_type])
    return replace(genome, tau=tau)


def perturb_bias(key: jax.Array, genome: Genome, cfg: Config, *,
                 sigma: float = 0.1) -> Genome:
    """Gaussian perturbation of per-neuron biases.  Signed — no clamp."""
    noise = jax.random.normal(key, genome.bias.shape) * sigma
    return replace(genome, bias=genome.bias + noise)


def type_flip(key: jax.Array, genome: Genome, cfg: Config, *,
              flip_prob: float = 0.05) -> Genome:
    """Flip each hidden neuron's type with probability flip_prob.

    I/O slots are protected — an inhibitory sensor would invert the whole
    input signal.  tau is re-clamped afterwards, since a neuron flipped E->FSI
    would otherwise keep a tau from the E range and violate the type
    invariant.
    """
    k1, k2 = jax.random.split(key)
    N = cfg.N_max

    new_types  = jax.random.randint(k1, (N,), 0, 3, dtype=jnp.uint8)
    flip_where = (jax.random.uniform(k2, (N,)) < flip_prob) & ~_io_protect(cfg)
    neuron_type = jnp.where(flip_where, new_types, genome.neuron_type)

    lo, hi = _tau_bounds(cfg)
    tau = jnp.clip(genome.tau, lo[neuron_type], hi[neuron_type])

    return replace(genome, neuron_type=neuron_type, tau=tau)


# ── Edge operators ────────────────────────────────────────────────────────────

def add_edge(key: jax.Array, genome: Genome, cfg: Config) -> Genome:
    """Add one randomly chosen missing edge between two active neurons.

    DELIBERATELY UNMASKED with respect to the lattice.  Locality is the
    initialisation, not a hard ceiling: evolution may buy a long-range
    shortcut, and the distance penalty decides whether it keeps it.  This is
    also why local_fraction is worth logging per generation — it measures how
    fast that prior erodes.

    Initialises the new weight strictly positive: a zero-weight edge would be
    invisible to selection and would simply be removed again before it could
    ever be evaluated.

    No-op when every active pair already has an edge.
    """
    k_slot, k_w = jax.random.split(key)
    N = cfg.N_max

    eligible = (
        genome.active_mask[:, None]
        & genome.active_mask[None, :]
        & ~genome.edge_mask
        & _no_self(cfg)
    )
    slot_flat, any_add = _random_eligible_1d(k_slot, eligible.reshape(-1))
    new_weight = jnp.abs(jax.random.normal(k_w)) * 0.5 + 0.01

    new_edges   = genome.edge_mask.reshape(-1).at[slot_flat].set(True).reshape(N, N)
    new_weights = genome.weight_matrix.reshape(-1).at[slot_flat].set(new_weight).reshape(N, N)

    return replace(
        genome,
        edge_mask=jnp.where(any_add, new_edges, genome.edge_mask),
        weight_matrix=jnp.where(any_add, new_weights, genome.weight_matrix),
    )


def remove_edges(key: jax.Array, genome: Genome, cfg: Config, *,
                 p_per_edge: float) -> Genome:
    """Remove each active edge independently with probability p_per_edge.

    Throughput scales with edge count, which is the property ctrnn_evo's
    single-edge operator lacked (see MutationRates.remove_edge_p_per_edge).

    The old stranding guard is deliberately dropped: it precomputed
    `out_degree <= 1` and refused edges whose removal would orphan an
    endpoint, which is invalid once several edges vanish simultaneously —
    the degrees it checked are stale the moment the first removal lands.
    prune_isolated runs after mutate and handles orphaned nodes instead.  The
    consequence is that node death is now driven by edge pruning rather than
    prevented by it, and it is one-way, so n_active is worth watching per
    generation.
    """
    active_pairs = genome.active_mask[:, None] & genome.active_mask[None, :]
    active_edges = genome.edge_mask & active_pairs

    draw = jax.random.uniform(key, active_edges.shape) < p_per_edge
    return replace(genome, edge_mask=genome.edge_mask & ~(active_edges & draw))


# ── Node operators (sparse arm only) ──────────────────────────────────────────

def add_node(key: jax.Array, genome: Genome, cfg: Config) -> Genome:
    """Activate the first free hidden slot, wired with one incoming and one
    outgoing edge to existing active neurons.

    Two connections ensure the new node sits on a computation path (it both
    receives and propagates), so prune_isolated does not immediately remove
    it.

    SPARSE ARM ONLY.  On a full lattice this self-disables: `eligible` is
    empty when every slot is active, so `apply` is False and nothing changes.
    Note also that the partners are chosen uniformly over active neurons, so
    on a lattice this would introduce long-range edges through a channel
    local_fraction could not distinguish from add_edge — another reason
    cfg.node_ops_enabled is restricted to the sparse arm.
    """
    k_type, k_tau, k_bias, k_src, k_dst, k_wi, k_wo = jax.random.split(key, 7)

    eligible = _hidden_slots(cfg) & ~genome.active_mask
    any_free = jnp.any(eligible)
    slot     = jnp.argmax(eligible)

    new_type = jax.random.randint(k_type, (), 0, 3, dtype=jnp.uint8)
    lo, hi   = _tau_bounds(cfg)
    new_tau  = lo[new_type] + jax.random.uniform(k_tau) * (hi[new_type] - lo[new_type])
    new_bias = jax.random.normal(k_bias) * 0.1

    src, any_src = _random_eligible_1d(k_src, genome.active_mask)
    dst, any_dst = _random_eligible_1d(k_dst, genome.active_mask)
    wi = jnp.abs(jax.random.normal(k_wi)) * 0.5 + 0.01
    wo = jnp.abs(jax.random.normal(k_wo)) * 0.5 + 0.01

    new_edges = genome.edge_mask.at[src, slot].set(True).at[slot, dst].set(True)
    new_edges = new_edges & _no_self(cfg)          # guard src == slot or slot == dst
    new_weights = genome.weight_matrix.at[src, slot].set(wi).at[slot, dst].set(wo)

    apply = any_free & any_src & any_dst

    return replace(
        genome,
        active_mask=jnp.where(apply, genome.active_mask.at[slot].set(True), genome.active_mask),
        neuron_type=jnp.where(apply, genome.neuron_type.at[slot].set(new_type), genome.neuron_type),
        tau=jnp.where(apply, genome.tau.at[slot].set(new_tau), genome.tau),
        bias=jnp.where(apply, genome.bias.at[slot].set(new_bias), genome.bias),
        edge_mask=jnp.where(apply, new_edges, genome.edge_mask),
        weight_matrix=jnp.where(apply, new_weights, genome.weight_matrix),
    )


def remove_node(key: jax.Array, genome: Genome, cfg: Config) -> Genome:
    """Deactivate a random active hidden neuron and clear its edges.

    SPARSE ARM ONLY, and enforced rather than assumed: on a lattice this would
    punch holes in the substrate under study.  cfg.node_ops_enabled is
    validated against init_mode in Config.__post_init__.
    """
    eligible = _hidden_slots(cfg) & genome.active_mask
    slot, any_remove = _random_eligible_1d(key, eligible)

    new_active = genome.active_mask.at[slot].set(False)
    new_edges  = genome.edge_mask.at[slot, :].set(False).at[:, slot].set(False)

    return replace(
        genome,
        active_mask=jnp.where(any_remove, new_active, genome.active_mask),
        edge_mask=jnp.where(any_remove, new_edges, genome.edge_mask),
    )


# ── Combined operator ─────────────────────────────────────────────────────────

def mutate(key: jax.Array, genome: Genome, cfg: Config,
           rates: MutationRates) -> Genome:
    """Apply every mutation operator in sequence.

    Continuous mutations always run (sigma=0 is a no-op).  add_edge fires on a
    Bernoulli draw; remove_edges runs every generation with a per-edge
    probability.  Node operators run only when cfg.node_ops_enabled is set,
    which Config restricts to the sparse arm.

    The node gate is a plain Python `if` on a static config field, not a
    lax.cond: the arm is known at trace time, so the branch is compiled out
    entirely for the grid and uniform arms rather than traced and skipped.
    """
    k = jax.random.split(key, 8)

    g = perturb_weights(k[0], genome, cfg, sigma=rates.weight_sigma)
    g = perturb_tau(    k[1], g,      cfg, sigma=rates.tau_sigma)
    g = perturb_bias(   k[2], g,      cfg, sigma=rates.bias_sigma)
    g = type_flip(      k[3], g,      cfg, flip_prob=rates.type_flip_prob)

    do_add_edge = jax.random.uniform(k[4]) < rates.add_edge_prob
    g = jax.lax.cond(do_add_edge,
                     lambda g_: add_edge(k[5], g_, cfg),
                     lambda g_: g_, g)

    g = remove_edges(k[6], g, cfg, p_per_edge=rates.remove_edge_p_per_edge)

    if cfg.node_ops_enabled:
        k_node = jax.random.split(k[7], 4)
        do_add    = jax.random.uniform(k_node[0]) < rates.add_node_prob
        do_remove = jax.random.uniform(k_node[1]) < rates.remove_node_prob
        g = jax.lax.cond(do_add,
                         lambda g_: add_node(k_node[2], g_, cfg),
                         lambda g_: g_, g)
        g = jax.lax.cond(do_remove,
                         lambda g_: remove_node(k_node[3], g_, cfg),
                         lambda g_: g_, g)

    return prune_isolated(g, cfg)