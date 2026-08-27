"""
ctrnn_lattice_evo — neuroevolution of CTRNNs on a locally dense lattice.

Every neuron occupies a fixed site on a W x H grid and connects only to
Chebyshev-nearby sites at initialisation.  Structural mutation may add edges
anywhere, so locality is a prior rather than a hard constraint, and the
pruning costs decide which non-local edges earn their keep.

Three arms, selected by Config.init_mode:

    grid     locally dense lattice                      (the proposal)
    uniform  same node and edge count, scattered        (isolates locality)
    sparse   start small and grow                       (the ctrnn_evo regime)

Module layering — imports run strictly one way, top to bottom:

    topology, world                 no internal imports
    config -> topology              controllers -> world
    genome -> config, topology
    cost, forward, mutation -> config, genome
    brain -> config, genome, forward, world
    analysis -> config, genome, cost, topology
    logger -> config, genome, world, mutation
    evolution -> everything
"""

from .config import Config, INIT_MODES, FITNESS_MODES

from .topology import (
    grid_coords,
    dist_matrix,
    local_mask,
    expected_edges,
    reference_costs,
)

from .genome import (
    Genome,
    E, FSI, SII,
    grid_genome,
    uniform_genome,
    sparse_genome,
    init_genome,
    constructor_for,
    prune_isolated,
    effective_weights,
    validate_genome,
)

from .world import (
    WorldConfig,
    WorldState,
    food_at,
    sensor_readout,
    step_world,
    reset_world,
    run_episode,
)

from .controllers import random_walk, nearest_hotspot

from .forward import forward_pass, batch_forward

# Stage 5:
# from .cost import edge_count_cost, dist_cost, adjusted_fitness
#
# Stage 6:
# from .mutation import (
#     MutationRates,
#     perturb_weights, perturb_tau, perturb_bias,
#     type_flip, add_edge, remove_edges,
#     mutate,
# )
#
# Stage 8:
# from .brain import (
#     run_brain_episode, run_brain_episode_full, batch_run_brain_episode,
# )
#
# Stage 9:
# from .analysis import (
#     modularity_q, local_fraction, network_stats,
#     analyse_genome, analyse_population, summarise_run,
# )
#
# Stage 10:
# from .logger import (
#     make_run_dir, save_config, load_config,
#     save_genome, load_genome,
#     append_history, load_history, make_logger,
#     save_training_state, load_training_state, latest_state_checkpoint,
# )
#
# Stage 11:
# from .evolution import (
#     init_population, eval_population, compute_fitness,
#     tournament_select_idx, select_parents, reproduce,
#     evolve_step, collect_stats, run_evolution,
# )

__all__ = [
    # config
    "Config", "INIT_MODES", "FITNESS_MODES",
    # topology
    "grid_coords", "dist_matrix", "local_mask",
    "expected_edges", "reference_costs",
    # genome
    "Genome", "E", "FSI", "SII",
    "grid_genome", "uniform_genome", "sparse_genome",
    "init_genome", "constructor_for",
    "prune_isolated", "effective_weights", "validate_genome",
    # world
    "WorldConfig", "WorldState", "food_at", "sensor_readout",
    "step_world", "reset_world", "run_episode",
    # controllers
    "random_walk", "nearest_hotspot",
]