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
    "count_active_edges",
    "add_edges", "remove_edges",
    "add_node", "remove_node",
    "mutate",
]


# ── MutationRates ─────────────────────────────────────────────────────────────

@dataclass
class MutationRates:
    """Per-operator mutation intensities.  Set any rate to 0 to disable.

    `position_sigma` is gone — positions belong to the lattice, not the genome.
    """
    # Continuous parameter mutations.  evolution.py scales these by
    # mutation_warmup_scale before calling mutate, so these are post-scaling.
    weight_sigma: float = 0.1
    tau_sigma:    float = 0.1
    bias_sigma:   float = 0.1

    type_flip_prob: float = 0.05

    # ── Structural rate ─────────────────────────────────────────────────────
    # ONE rate drives both edge operators, and both scale with the current
    # edge count E, so at equal rates their expected contributions cancel
    # identically at every density:
    #
    #     E[dE] = p*E  (added)  -  p*E  (removed)  =  0
    #
    # That makes edge count a driftless random walk: an arm stays near
    # whatever density it started at, and any systematic movement observed
    # under a live penalty is SELECTION rather than operator bias.
    #
    # Why not a per-slot addition rate over the M-E empty slots?  Because
    # p_add*(M-E) - p_rem*E has an attractor at M*p_add/(p_add+p_rem), which
    # depends only on the rates — so a single global rate would drag every
    # arm to the same density regardless of how it was initialised (M/2 at
    # equal rates), destroying the arm distinction just as thoroughly as an
    # unbalanced operator.
    #
    # The original operators were badly unbalanced: removal was per-edge
    # (p*E deletions) while addition was a single coin flip (<=1 per
    # generation), giving an attractor at p_add_fires/p_rem ~= 33 edges.
    # Every arm collapsed to ~27-41 edges and ~10-24 nodes at edge_frac=0,
    # with no cost pressure at all — pure drift, and it dominated the result.
    #
    # At 0.003 a 1092-edge genome turns over ~3.3 edges each way per
    # generation, and the random walk wanders roughly +/-80 edges over 1000
    # generations (~7%).
    edge_churn: float = 0.003

    # Node operators — sparse arm only, gated on cfg.node_ops_enabled.  On a
    # full lattice add_node self-disables (no free slots) and remove_node
    # would punch holes in the substrate under study.
    add_node_prob:    float = 0.05
    remove_node_prob: float = 0.05


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tau_bounds(cfg: Config):
    lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    return lo, hi


def _io_protect(cfg: Config) -> jnp.ndarray:
    N = cfg.N_max
    return jnp.zeros(N, dtype=bool).at[:cfg.n_in].set(True).at[N - cfg.n_out:].set(True)


def _hidden_slots(cfg: Config) -> jnp.ndarray:
    N = cfg.N_max
    return jnp.zeros(N, dtype=bool).at[cfg.n_in: N - cfg.n_out].set(True)


def _random_eligible_1d(key: jax.Array, eligible: jnp.ndarray):
    """Pick one True index from a 1-D bool mask; returns (idx, any_eligible)."""
    noise = jax.random.uniform(key, eligible.shape)
    return jnp.argmax(jnp.where(eligible, noise, -1.0)), jnp.any(eligible)


def _no_self(cfg: Config) -> jnp.ndarray:
    return ~jnp.eye(cfg.N_max, dtype=bool)


def _active_pairs(genome: Genome) -> jnp.ndarray:
    return genome.active_mask[:, None] & genome.active_mask[None, :]


def count_active_edges(genome: Genome) -> jnp.ndarray:
    """Edges joining two active neurons — the E both operators are driven by."""
    return jnp.sum(genome.edge_mask & _active_pairs(genome))


def _pick_n(key: jax.Array, eligible: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """Select n positions uniformly from `eligible`.

    n is a traced value, so it cannot slice — the selection is done by ranking
    all eligible positions on random noise and masking the top n.  Ineligible
    positions score -1 and so rank last, and the final AND guarantees nothing
    outside `eligible` is ever returned even when n exceeds the eligible count.
    """
    noise = jnp.where(eligible, jax.random.uniform(key, eligible.shape), -1.0)
    order = jnp.argsort(-noise.ravel())
    take = jnp.arange(order.shape[0]) < n
    chosen = jnp.zeros(order.shape, dtype=bool).at[order].set(take)
    return chosen.reshape(eligible.shape) & eligible


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

    The clamp also keeps tau >= tau_min >= dt, the Euler stability condition —
    at tau_fsi_range[0]=1.0 against dt=0.5 the margin is only 2x.
    """
    lo, hi = _tau_bounds(cfg)
    noise = jax.random.normal(key, genome.tau.shape) * sigma
    return replace(genome, tau=jnp.clip(genome.tau + noise,
                                        lo[genome.neuron_type],
                                        hi[genome.neuron_type]))


def perturb_bias(key: jax.Array, genome: Genome, cfg: Config, *,
                 sigma: float = 0.1) -> Genome:
    """Gaussian perturbation of per-neuron biases.  Signed — no clamp."""
    noise = jax.random.normal(key, genome.bias.shape) * sigma
    return replace(genome, bias=genome.bias + noise)


def type_flip(key: jax.Array, genome: Genome, cfg: Config, *,
              flip_prob: float = 0.05) -> Genome:
    """Flip each hidden neuron's type with probability flip_prob.

    I/O slots are protected — an inhibitory sensor would invert the whole
    input signal.  tau is re-clamped afterwards, since a neuron flipped
    E -> FSI would otherwise keep a tau from the E range.
    """
    k1, k2 = jax.random.split(key)
    N = cfg.N_max

    new_types = jax.random.randint(k1, (N,), 0, 3, dtype=jnp.uint8)
    flip_where = (jax.random.uniform(k2, (N,)) < flip_prob) & ~_io_protect(cfg)
    neuron_type = jnp.where(flip_where, new_types, genome.neuron_type)

    lo, hi = _tau_bounds(cfg)
    return replace(genome, neuron_type=neuron_type,
                   tau=jnp.clip(genome.tau, lo[neuron_type], hi[neuron_type]))


# ── Edge operators ────────────────────────────────────────────────────────────

def add_edges(key: jax.Array, genome: Genome, cfg: Config, *,
              p_per_edge: float, n_edges: jnp.ndarray | None = None) -> Genome:
    """Add ~p_per_edge * n_edges new edges at uniformly chosen empty slots.

    `n_edges` is the PRE-MUTATION edge count.  Pass the same value to
    remove_edges so both operators are driven by the identical number: their
    expectations then cancel exactly, with no dependence on which ran first.
    Letting each recompute E would make the second see the first's effect,
    a small ordering bias of order p^2 * E per generation.

    The count is Poisson(p*E) rather than a rounded p*E so that addition is
    stochastic like removal; a deterministic count would pair a fixed
    addition against a Binomial removal and bias the walk.

    DELIBERATELY UNMASKED with respect to the lattice: locality is the
    initialisation, not a ceiling, and the distance penalty decides whether a
    long-range edge earns its keep.  Note the consequence — removals come out
    of the current topology while additions land anywhere, so a lattice loses
    local edges and gains mostly non-local ones.  local_fraction declines even
    with edge count flat, which is why it is logged per generation.

    New weights are strictly positive: a zero-weight edge is invisible to
    selection and would be removed again before it could ever be evaluated.
    """
    k_n, k_pick, k_w = jax.random.split(key, 3)

    if n_edges is None:
        n_edges = count_active_edges(genome)

    eligible = _active_pairs(genome) & ~genome.edge_mask & _no_self(cfg)
    n_add = jax.random.poisson(k_n, p_per_edge * n_edges.astype(jnp.float32))
    new = _pick_n(k_pick, eligible, n_add)

    w = jnp.abs(jax.random.normal(k_w, eligible.shape)) * 0.5 + 0.01
    return replace(
        genome,
        edge_mask=genome.edge_mask | new,
        weight_matrix=jnp.where(new, w, genome.weight_matrix),
    )


def remove_edges(key: jax.Array, genome: Genome, cfg: Config, *,
                 p_per_edge: float, n_edges: jnp.ndarray | None = None) -> Genome:
    """Remove ~p_per_edge * n_edges edges, chosen uniformly from those present.

    Drawn as Poisson(p*E) on the pre-mutation E and then sampled, rather than
    an independent coin flip per edge, so that it mirrors add_edges exactly
    and both are driven by the same number.  (A per-edge Bernoulli would give
    Binomial(E, p), near-identical in this regime, but would see the
    post-addition edge set and reintroduce the ordering bias.)

    The old stranding guard is deliberately absent: it precomputed
    `out_degree <= 1` and refused edges whose removal would orphan an
    endpoint, which is invalid once several edges vanish at once — those
    degrees are stale the moment the first removal lands.  prune_isolated runs
    after mutate and handles orphans instead.  The consequence is that node
    death is driven by edge pruning rather than prevented by it, and it is
    one-way, so n_active is worth watching per generation.
    """
    k_n, k_pick = jax.random.split(key)

    if n_edges is None:
        n_edges = count_active_edges(genome)

    present = genome.edge_mask & _active_pairs(genome)
    n_rem = jax.random.poisson(k_n, p_per_edge * n_edges.astype(jnp.float32))
    gone = _pick_n(k_pick, present, n_rem)

    return replace(genome, edge_mask=genome.edge_mask & ~gone)


# ── Node operators (sparse arm only) ──────────────────────────────────────────

def add_node(key: jax.Array, genome: Genome, cfg: Config) -> Genome:
    """Activate the first free hidden slot, wired with one incoming and one
    outgoing edge to existing active neurons.

    Two connections ensure the new node sits on a computation path, so
    prune_isolated does not immediately remove it.

    SPARSE ARM ONLY.  On a full lattice this self-disables — `eligible` is
    empty when every slot is active.  Its partners are chosen uniformly over
    active neurons, so on a lattice it would introduce long-range edges
    through a channel local_fraction could not distinguish from add_edges:
    another reason cfg.node_ops_enabled is restricted to the sparse arm.
    """
    k_type, k_tau, k_bias, k_src, k_dst, k_wi, k_wo = jax.random.split(key, 7)

    eligible = _hidden_slots(cfg) & ~genome.active_mask
    any_free = jnp.any(eligible)
    slot = jnp.argmax(eligible)

    new_type = jax.random.randint(k_type, (), 0, 3, dtype=jnp.uint8)
    lo, hi = _tau_bounds(cfg)
    new_tau = lo[new_type] + jax.random.uniform(k_tau) * (hi[new_type] - lo[new_type])
    new_bias = jax.random.normal(k_bias) * 0.1

    src, any_src = _random_eligible_1d(k_src, genome.active_mask)
    dst, any_dst = _random_eligible_1d(k_dst, genome.active_mask)
    wi = jnp.abs(jax.random.normal(k_wi)) * 0.5 + 0.01
    wo = jnp.abs(jax.random.normal(k_wo)) * 0.5 + 0.01

    new_edges = genome.edge_mask.at[src, slot].set(True).at[slot, dst].set(True)
    new_edges = new_edges & _no_self(cfg)      # guard src == slot or slot == dst
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

    SPARSE ARM ONLY, enforced in Config: on a lattice this would punch holes
    in the substrate under study.
    """
    eligible = _hidden_slots(cfg) & genome.active_mask
    slot, any_remove = _random_eligible_1d(key, eligible)

    new_active = genome.active_mask.at[slot].set(False)
    new_edges = genome.edge_mask.at[slot, :].set(False).at[:, slot].set(False)

    return replace(
        genome,
        active_mask=jnp.where(any_remove, new_active, genome.active_mask),
        edge_mask=jnp.where(any_remove, new_edges, genome.edge_mask),
    )


# ── Combined operator ─────────────────────────────────────────────────────────

def mutate(key: jax.Array, genome: Genome, cfg: Config,
           rates: MutationRates) -> Genome:
    """Apply every mutation operator in sequence.

    The edge count E is measured ONCE, before either edge operator runs, and
    handed to both.  That is what makes the pair exactly neutral: each adds
    and removes Poisson(p*E) edges from the same E, so E[dE] = 0 at any
    density and independent of ordering.  Recomputing E inside each operator
    would let the second see the first's effect and introduce a small
    systematic drift.

    Node operators run only when cfg.node_ops_enabled, which Config restricts
    to the sparse arm.  The gate is a plain Python `if` on a static config
    field rather than a lax.cond, so the branch compiles out entirely for the
    fixed-lattice arms.
    """
    k = jax.random.split(key, 8)

    g = perturb_weights(k[0], genome, cfg, sigma=rates.weight_sigma)
    g = perturb_tau(    k[1], g,      cfg, sigma=rates.tau_sigma)
    g = perturb_bias(   k[2], g,      cfg, sigma=rates.bias_sigma)
    g = type_flip(      k[3], g,      cfg, flip_prob=rates.type_flip_prob)

    n_edges = count_active_edges(g)          # measured once, shared by both
    g = add_edges(   k[4], g, cfg, p_per_edge=rates.edge_churn, n_edges=n_edges)
    g = remove_edges(k[5], g, cfg, p_per_edge=rates.edge_churn, n_edges=n_edges)

    if cfg.node_ops_enabled:
        k_node = jax.random.split(k[6], 4)
        do_add    = jax.random.uniform(k_node[0]) < rates.add_node_prob
        do_remove = jax.random.uniform(k_node[1]) < rates.remove_node_prob
        g = jax.lax.cond(do_add,
                         lambda g_: add_node(k_node[2], g_, cfg),
                         lambda g_: g_, g)
        g = jax.lax.cond(do_remove,
                         lambda g_: remove_node(k_node[3], g_, cfg),
                         lambda g_: g_, g)

    return prune_isolated(g, cfg)