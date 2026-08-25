from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # Network capacity
    N_max: int = 64
    n_out: int = 2

    # Food types — n_in is derived automatically as 2 * n_food_types
    n_food_types: int = 1

    # Integration
    dt:       float = 0.5   # neural timestep (ms); must satisfy dt <= tau_min
    dt_world: float = 10.0  # world timestep (ms)
    K:        int   = 20    # inner CTRNN ticks per world step

    # Type-specific tau ranges (ms) — E, FS-I, SI-I
    tau_e_range:   Tuple[float, float] = (10.0,  100.0)
    tau_fsi_range: Tuple[float, float] = (1.0,    15.0)
    tau_sii_range: Tuple[float, float] = (80.0,  500.0)

    # Initial edge density (fraction of possible edges active at init)
    init_edge_density: float = 0.15

    # Cost coefficients — absolute mode (0 = disabled)
    lambda_edge: float = 0.0   # penalises edge count regardless of length
    lambda_dist: float = 0.0   # penalises total wire length (distance-weighted)
    lambda_act:  float = 0.0   # penalises mean neural activation per tick

    # Penalty warm-up — ramp all λ from 0 to their full values over this many
    # generations, so early-generation networks are not pruned before any
    # foraging strategy has evolved.  0 = disabled (lambdas are constant).
    penalty_warmup_gens: int = 0

    # Mutation warm-up scale — during penalty_warmup_gens, scale all continuous
    # mutation sigmas (weight, tau, bias, position) by this factor, linearly
    # decaying to 1.0 by the end of the warmup window.  Mirrors the penalty
    # ramp: high exploration when penalty=0, normal rates when penalty=full.
    # 1.0 = disabled (constant mutation rates throughout).
    mutation_warmup_scale: float = 1.0

    # Cyclic loosening — after the initial warm-up, drop all penalties to zero
    # for penalty_cycle_free_gens at the end of every penalty_cycle_gens window.
    # Allows cold reps that converged on a local optimum to re-explore before
    # pressure resumes.  Both must be > 0 to enable; 0 = disabled.
    #
    # Example: warmup=200, cycle_gens=300, free_gens=100 →
    #   gen 0-199:   warmup ramp
    #   gen 200-499: full penalty
    #   gen 500-599: free (penalty=0)   ← rescue window 1
    #   gen 600-899: full penalty
    #   gen 900-999: free (penalty=0)   ← rescue window 2
    penalty_cycle_gens:      int = 0
    penalty_cycle_free_gens: int = 0

    # Cost fractions — proportional mode (0 = disabled; takes priority over λ when > 0)
    #
    # Penalty = frac × f_raw × (C / C0_ref)
    #
    # This keeps the penalty as a fixed percentage of raw fitness when the
    # network is at reference cost C0_ref, making the regularisation scale
    # automatically with the fitness signal.  Works identically whether
    # f_raw ~ 0.009 (food-score gen 0) or f_raw ~ 0.9 (survival), so no
    # per-experiment recalibration of λ is needed.
    #
    # Reference costs (C0_*) are the empirically observed initial values for
    # a random 64-neuron network in the default arena.  Override if N_max or
    # arena_size differ significantly.
    dist_frac:  float = 0.0    # wiring-length penalty as fraction of f_raw
    act_frac:   float = 0.0    # activation penalty as fraction of f_raw
    edge_frac:  float = 0.0    # edge-count penalty as fraction of f_raw
    C0_wiring:  float = 77.0   # reference wiring cost  (random init, default params)
    C0_act:     float = 1.0    # theoretical max of normalised c_act (mean |tanh(v)| ∈ [0,1]);
                               # no calibration needed — saturation ceiling is always 1.0
    C0_edge:    float = 154.0  # reference edge count   (random init, default params)

    # Evolution
    population_size:  int = 1000
    tournament_size:  int = 4

    # Fitness metric
    # "survival": steps_survived / episode_steps  → [0, 1]
    # "food":     cumulative raw food score / (episode_steps * n_food_types)
    #             → can exceed 1.0 for agents that actively forage near hotspots
    fitness_mode: str = "survival"

    # Position sensors — if True, normalised (x, y) in [0, 1] are appended to the
    # sensor vector, giving the agent proprioceptive awareness of its arena location.
    # Without these, agents starting far from any food hotspot receive a zero food
    # signal and have no gradient to follow — they run open-loop into walls.
    # Adds 2 to n_in.  Default False for backward compatibility.
    position_sensors: bool = False

    # Derived — set by __post_init__, not a constructor argument
    n_in: int = field(init=False)

    def __post_init__(self):
        self.n_in = 2 * self.n_food_types + (2 if self.position_sensors else 0)

    def tau_range(self, neuron_type: int) -> Tuple[float, float]:
        return (self.tau_e_range, self.tau_fsi_range, self.tau_sii_range)[neuron_type]
