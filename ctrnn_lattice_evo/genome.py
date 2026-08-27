from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .config import Config
from .topology import local_mask

# Neuron type constants
E   = 0  # excitatory
FSI = 1  # fast-spiking inhibitory
SII = 2  # slow-integrating inhibitory


@dataclass
class Genome:
    """
    Fixed-capacity padded genome for one CTRNN organism.

    All arrays have a leading N_max dimension so a population of P organisms
    is a single batched Genome with every field shaped [P, N_max, ...],
    enabling jax.vmap across the population.

    Weight magnitudes are stored non-negative; Dale's law sign is derived from
    neuron_type at forward-pass time via effective_weights().

    SIX fields — `position` is gone.  Geometry is a property of the lattice,
    not of the individual: every genome sits on the same fixed grid, so
    carrying a [P, N_max, 2] position array per population and mutating it
    would be pure overhead.  Coordinates and distances live in topology.py.
    """
    active_mask:   jnp.ndarray  # [N_max]         bool    — which slots are live
    neuron_type:   jnp.ndarray  # [N_max]         uint8   — 0=E, 1=FSI, 2=SII
    tau:           jnp.ndarray  # [N_max]         float32 — time constants (ms)
    bias:          jnp.ndarray  # [N_max]         float32
    weight_matrix: jnp.ndarray  # [N_max, N_max]  float32 — non-negative magnitudes
    edge_mask:     jnp.ndarray  # [N_max, N_max]  bool    — structural connectivity


# Register as a JAX pytree so jit/vmap can look inside the dataclass.
# This field order is also what logger._GENOME_FIELDS must use, since
# load_genome reconstructs positionally via Genome(*children).
jax.tree_util.register_pytree_node(
    Genome,
    lambda g: (
        [g.active_mask, g.neuron_type, g.tau, g.bias,
         g.weight_matrix, g.edge_mask],
        None,
    ),
    lambda _, children: Genome(*children),
)


# ── Structural helpers ────────────────────────────────────────────────────────

def prune_isolated(genome: Genome, cfg: Config) -> Genome:
    """
    Deactivate hidden neurons with no incoming or no outgoing edge among active
    neurons.  I/O neurons are always kept regardless of connectivity.

    On a lattice this is now the ONLY route to node death — add_node and
    remove_node are gone — and it is one-way: a slot that goes inactive cannot
    come back.  Worth watching n_active per generation once remove_edges starts
    firing.
    """
    active_pairs = genome.active_mask[:, None] & genome.active_mask[None, :]
    active_edges = genome.edge_mask & active_pairs
    has_in  = jnp.any(active_edges, axis=0)   # receives from another active neuron
    has_out = jnp.any(active_edges, axis=1)   # sends to another active neuron

    hidden = jnp.zeros(cfg.N_max, dtype=bool).at[cfg.n_in: cfg.N_max - cfg.n_out].set(True)
    new_active = genome.active_mask & ((has_in & has_out) | ~hidden)
    new_edges  = genome.edge_mask & new_active[:, None] & new_active[None, :]

    return replace(genome, active_mask=new_active, edge_mask=new_edges)


def effective_weights(genome: Genome) -> jnp.ndarray:
    """
    W_eff [N_max, N_max] with Dale's law, edge mask and activity mask applied.

    W_eff[i, j] is the weight FROM neuron j TO neuron i, signed by the type of
    the source neuron j.
    """
    sign_vec = jnp.where(genome.neuron_type == E, 1.0, -1.0)   # [N_max]
    return (
        genome.weight_matrix
        * sign_vec[None, :]              # source neuron sign (Dale's law)
        * genome.edge_mask               # structural connectivity
        * genome.active_mask[:, None]    # silence inactive post-synaptic neurons
        * genome.active_mask[None, :]    # silence inactive pre-synaptic neurons
    )


# ── Key discipline ────────────────────────────────────────────────────────────
#
# Every arm splits its key the SAME way and uses the SAME sub-key for
# parameters, even where it has no use for the others:
#
#     k_params, k_edges, k_active = jax.random.split(key, 3)
#
# So grid_genome(k) and uniform_genome(k) produce identical weights, tau,
# biases and neuron types, differing ONLY in edge placement.  That makes the
# grid-vs-uniform contrast paired rather than independent: at equal replicate
# count a paired design has materially lower variance, because the parametric
# draw is held constant instead of being another source of noise.
#
# Do not "optimise" this by giving an arm fewer splits — it silently
# unpairs the comparison.

_N_KEYS = 3


def _split(key: jax.Array):
    """(k_params, k_edges, k_active) — the same split for every arm."""
    return jax.random.split(key, _N_KEYS)


def _sample_params(key: jax.Array, cfg: Config):
    """Sample the non-structural fields: neuron type, tau, bias, weights.

    Given the same key this returns the same values regardless of arm, which
    is what pairs the comparison (see the note above).
    """
    k_type, k_tau, k_bias, k_w = jax.random.split(key, 4)

    neuron_type = jax.random.randint(k_type, (cfg.N_max,), 0, 3, dtype=jnp.uint8)
    neuron_type = neuron_type.at[:cfg.n_in].set(E)
    neuron_type = neuron_type.at[-cfg.n_out:].set(E)

    tau_lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    tau_hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    u = jax.random.uniform(k_tau, (cfg.N_max,))
    tau = tau_lo[neuron_type] + u * (tau_hi[neuron_type] - tau_lo[neuron_type])

    bias = jax.random.normal(k_bias, (cfg.N_max,)) * 0.1
    weight_matrix = jnp.abs(jax.random.normal(k_w, (cfg.N_max, cfg.N_max)) * 0.5)

    return neuron_type, tau, bias, weight_matrix


def _all_active(cfg: Config) -> jnp.ndarray:
    """Every lattice slot live — the grid and uniform arms."""
    return jnp.ones(cfg.N_max, dtype=bool)


def _lattice_edge_count(cfg: Config) -> int:
    """Edge count of a fresh lattice, as a Python int.

    Read from local_mask rather than cfg.C0_edge so that overriding C0 to
    reproduce an old run cannot change what the uniform arm is matched
    against — the control has to track the lattice, not the denominator.
    """
    return int(local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H).sum())


def _exact_random_edges(key: jax.Array, n_edges: int,
                        eligible: jnp.ndarray) -> jnp.ndarray:
    """Choose exactly n_edges positions uniformly from `eligible`.

    Exact rather than Bernoulli because the uniform arm must MATCH the grid
    arm's edge count, not merely match it in expectation.  On the 4x4 test
    lattice a Bernoulli draw has sd ~7.4 against a target of 84, so a 10%
    tolerance would fail roughly a quarter of the time on correct code.

    n_edges is a Python int derived from static cfg fields, so the argsort
    slice has a static shape and this stays vmap-safe over `key`.
    """
    noise  = jax.random.uniform(key, eligible.shape)
    scores = jnp.where(eligible, noise, -1.0).ravel()
    idx    = jnp.argsort(-scores)[:n_edges]
    flat   = jnp.zeros(scores.shape, dtype=bool).at[idx].set(True)
    return flat.reshape(eligible.shape)


# ── Arm constructors ──────────────────────────────────────────────────────────

def grid_genome(key: jax.Array, cfg: Config) -> Genome:
    """Locally dense Chebyshev lattice — the proposal.

    Every slot is live and the edge mask IS the lattice, so all individuals
    share one topology at generation 0.  Diversity comes only from weights,
    tau, bias and neuron type — a real departure from ctrnn_evo, where each
    individual carried its own random edge set.  Structural mutation is
    therefore the sole source of topological diversity from here on.
    """
    k_params, _k_edges, _k_active = _split(key)
    neuron_type, tau, bias, weight_matrix = _sample_params(k_params, cfg)

    return prune_isolated(Genome(
        active_mask=_all_active(cfg),
        neuron_type=neuron_type,
        tau=tau,
        bias=bias,
        weight_matrix=weight_matrix,
        edge_mask=local_mask(cfg.grid_W, cfg.grid_r, cfg.grid_H),
    ), cfg)


def uniform_genome(key: jax.Array, cfg: Config) -> Genome:
    """Same node count and same edge count as the lattice, scattered placement.

    THE locality control.  An Erdos-Renyi random digraph, not a layered
    network: edges are drawn uniformly over all off-diagonal pairs, so it is
    fully recurrent with both directions equally likely.

    Matching the grid arm on node count AND edge count means a difference
    between the two cannot be attributed to either — only to locality.  Without
    this arm, "grid beats sparse" reads most parsimoniously as "dense init
    beats sparse init", which is already known and needs no lattice.

    Note its local_fraction starts near n_edges/(N^2-N) — about 0.27 at
    production scale — not at 0: that share of random edges lands inside the
    lattice ball by chance.  That is the floor the locality metric is read
    against.
    """
    k_params, k_edges, _k_active = _split(key)
    neuron_type, tau, bias, weight_matrix = _sample_params(k_params, cfg)

    eligible = ~jnp.eye(cfg.N_max, dtype=bool)   # any pair except self-edges
    edge_mask = _exact_random_edges(k_edges, _lattice_edge_count(cfg), eligible)

    # prune_isolated is effectively a no-op at production scale (8x8 r=2 gives
    # p~0.27, so P(any node isolated) ~ 1e-7).  On the 4x4 test lattice p~0.35
    # and the chance is a few percent — deterministic per seed, but worth
    # knowing if test_uniform_genome_is_fully_active ever trips.
    return prune_isolated(Genome(
        active_mask=_all_active(cfg),
        neuron_type=neuron_type,
        tau=tau,
        bias=bias,
        weight_matrix=weight_matrix,
        edge_mask=edge_mask,
    ), cfg)


def sparse_genome(key: jax.Array, cfg: Config) -> Genome:
    """Start small and grow — the ctrnn_evo regime, and the incumbent baseline.

    cfg.sparse_n_active slots live, edges Bernoulli at cfg.init_edge_density.
    Unlike the other two arms this one has inactive slots at generation 0, so
    node count is a free variable here and fixed elsewhere.

    At the 4x4 test scale this arm is very thin (~4 hidden nodes, ~8 edges) and
    prune_isolated strips much of it.  That is fine for the tests but says
    nothing about production behaviour, where N=64 at density 0.15 gives ~149
    edges and a healthy ~4.6 in/out per node.
    """
    k_params, k_edges, k_active = _split(key)
    neuron_type, tau, bias, weight_matrix = _sample_params(k_params, cfg)

    fixed       = cfg.n_in + cfg.n_out
    n_hidden    = cfg.N_max - fixed
    n_hidden_on = max(0, cfg.sparse_n_active - fixed)

    perm = jax.random.permutation(k_active, n_hidden)
    hidden_mask = jnp.zeros(n_hidden, dtype=bool).at[perm[:n_hidden_on]].set(True)
    active_mask = jnp.concatenate([
        jnp.ones(cfg.n_in, dtype=bool),
        hidden_mask,
        jnp.ones(cfg.n_out, dtype=bool),
    ])

    candidate = jax.random.uniform(k_edges, (cfg.N_max, cfg.N_max)) < cfg.init_edge_density
    edge_mask = (
        candidate
        & active_mask[:, None]
        & active_mask[None, :]
        & ~jnp.eye(cfg.N_max, dtype=bool)
    )

    return prune_isolated(Genome(
        active_mask=active_mask,
        neuron_type=neuron_type,
        tau=tau,
        bias=bias,
        weight_matrix=weight_matrix,
        edge_mask=edge_mask,
    ), cfg)


_CONSTRUCTORS = {
    "grid":    grid_genome,
    "uniform": uniform_genome,
    "sparse":  sparse_genome,
}


def constructor_for(cfg: Config):
    """The concrete arm constructor, for callers that want to vmap it.

    The arm is chosen by picking a function, not by passing a mode string
    through: a Python string cannot be a traced argument, so dispatch has to
    happen before any vmap.
    """
    return _CONSTRUCTORS[cfg.init_mode]


def init_genome(key: jax.Array, cfg: Config) -> Genome:
    """Build one genome for the arm named by cfg.init_mode."""
    return constructor_for(cfg)(key, cfg)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_genome(genome: Genome, cfg: Config) -> bool:
    """Assert all structural invariants.  Returns True, raises on failure."""
    assert jnp.all(genome.weight_matrix >= 0), "Negative weight magnitudes"
    assert jnp.all(genome.tau > 0),            "Non-positive time constants"
    assert jnp.all(genome.neuron_type <= 2),   "Invalid neuron type"
    assert jnp.all(genome.active_mask[:cfg.n_in]),   "Input neurons must be active"
    assert jnp.all(genome.active_mask[-cfg.n_out:]), "Output neurons must be active"

    # Edges may only join two active neurons.  A violation is SILENT:
    # effective_weights masks it out, so the forward pass is unaffected, but
    # edge_count_cost overcounts and every penalty is miscalibrated.
    active_pairs = genome.active_mask[:, None] & genome.active_mask[None, :]
    assert jnp.all(genome.edge_mask <= active_pairs), \
        "Edge to or from an inactive neuron"

    # Self-connections are excluded by design in all three arms (see the
    # topology decision: tau already provides a fixed leak, and a self-weight
    # would add single-neuron bistability that is not a locality question).
    assert not jnp.any(jnp.diag(genome.edge_mask)), "Self-edge present"

    tau_lo = jnp.array([cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0]])
    tau_hi = jnp.array([cfg.tau_e_range[1], cfg.tau_fsi_range[1], cfg.tau_sii_range[1]])
    in_range = (
        (genome.tau >= tau_lo[genome.neuron_type] - 1e-4) &
        (genome.tau <= tau_hi[genome.neuron_type] + 1e-4)
    )
    assert jnp.all(jnp.where(genome.active_mask, in_range, True)), \
        "Time constant out of type range for an active neuron"

    return True