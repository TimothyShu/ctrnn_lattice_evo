from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from .topology import reference_costs

# Valid values for Config.init_mode — the three experimental arms.
INIT_MODES = ("grid", "uniform", "sparse")
FITNESS_MODES = ("survival", "food")


@dataclass
class Config:
    # ── Network capacity ─────────────────────────────────────────────────────
    # N_max must equal grid_W * grid_H: every slot is a lattice site.
    N_max: int = 64
    n_out: int = 2

    # ── Lattice geometry ─────────────────────────────────────────────────────
    # grid_W is the number of ROWS, grid_H the number of COLUMNS, matching
    # topology.grid_coords, which is row-major: slot k -> (k // H, k % H).
    # This is the opposite of the usual width/height reading, and it only
    # matters for rectangular lattices — 4x16 and 16x4 share a corner distance
    # of 15 but have different degree structure.
    #
    # grid_r is the Chebyshev neighbourhood radius.  At r=1 every lattice edge
    # has length 1, so C0_dist == C0_edge and the wiring-length penalty is
    # exactly collinear with the edge-count penalty; the two are independent
    # axes only at r >= 2.
    grid_W: int = 8
    grid_H: int | None = None   # defaults to grid_W (square lattice)
    grid_r: int = 2

    # Which arm this run belongs to:
    #   "grid"    — locally dense Chebyshev lattice (the proposal)
    #   "uniform" — same edge count, no spatial structure (isolates locality)
    #   "sparse"  — start small and grow (the ctrnn_evo regime)
    init_mode: str = "grid"

    # ── Food types — n_in is derived as 2 * n_food_types (+2 with position) ──
    n_food_types: int = 1

    # ── Integration ──────────────────────────────────────────────────────────
    dt:       float = 0.5   # neural timestep (ms); must satisfy dt <= tau_min
    dt_world: float = 10.0  # world timestep (ms)
    K:        int   = 20    # inner CTRNN ticks per world step

    # Type-specific tau ranges (ms) — E, FS-I, SI-I.
    # tau_fsi_range[0] is the binding constraint on dt: at 1.0 against dt=0.5
    # the Euler integrator has only a 2x stability margin.
    tau_e_range:   Tuple[float, float] = (10.0,  100.0)
    tau_fsi_range: Tuple[float, float] = (1.0,    15.0)
    tau_sii_range: Tuple[float, float] = (80.0,  500.0)

    # ── Sparse-arm initialisation (ignored by grid / uniform) ────────────────
    # The regime the lattice is being compared against: start small, grow via
    # structural mutation.  Lifted out of the constructor so the "how sparse"
    # half of the headline comparison is a run parameter rather than a
    # hardcoded default.
    init_edge_density: float = 0.15
    sparse_n_active:   int | None = None   # defaults to N_max // 2

    # ── Cost fractions — proportional penalties only ─────────────────────────
    #
    #   f = f_raw * max(0, 1 - edge_frac*C_edge/C0_edge
    #                       - dist_frac*C_dist/C0_dist
    #                       - act_frac *C_act /C0_act)
    #
    # frac reads directly as "fraction of fitness surrendered at reference
    # cost", so it is comparable across lattice sizes and radii with no
    # recalibration, and works whether f_raw ~ 0.009 or ~ 0.9.
    #
    # The clamp at 0 is load-bearing.  Without it the bracket goes negative
    # once frac*C/C0 > 1, and since f_raw >= 0 always, a BETTER network then
    # maps to a MORE negative adjusted fitness — tournament selection silently
    # runs evolution backwards.
    #
    # ctrnn_evo's absolute lambda_* mode is deliberately gone: one code path,
    # so there is nowhere for that sign flip to hide.
    edge_frac: float = 0.0
    dist_frac: float = 0.0
    act_frac:  float = 0.0

    # Reference costs (the denominators above).  Leave as None to derive from
    # the lattice in __post_init__ via topology.reference_costs — that is the
    # single definition, and Config's job is to hold and serialise the result,
    # not to reimplement it.  Pass explicitly only to reproduce an older run.
    C0_edge: float | None = None
    C0_dist: float | None = None
    C0_act:  float = 1.0   # c_act is mean |tanh(v)| over active neurons, so its
                           # ceiling is exactly 1.0 regardless of network size

    # ── Penalty warm-up ──────────────────────────────────────────────────────
    # Ramp every frac from 0 to full over this many generations, so networks
    # are not pruned before any foraging strategy has evolved.  0 = disabled.
    penalty_warmup_gens: int = 0

    # During warm-up, scale continuous mutation sigmas (weight, tau, bias) by
    # this factor, decaying linearly to 1.0 — wide exploration while the
    # penalty is still near zero.  1.0 = disabled.
    mutation_warmup_scale: float = 1.0

    # Cyclic loosening — after warm-up, drop all penalties to zero for
    # penalty_cycle_free_gens at the end of every penalty_cycle_gens window,
    # letting replicates stuck in a local optimum re-explore.  Both > 0 to
    # enable.
    #
    # Example: warmup=200, cycle_gens=300, free_gens=100 ->
    #   gen 0-199:   warm-up ramp
    #   gen 200-499: full penalty
    #   gen 500-599: free            <- rescue window 1
    #   gen 600-899: full penalty
    #   gen 900-999: free            <- rescue window 2
    penalty_cycle_gens:      int = 0
    penalty_cycle_free_gens: int = 0

    # ── Evolution ────────────────────────────────────────────────────────────
    population_size: int = 1000
    tournament_size: int = 4

    # ── Fitness metric ───────────────────────────────────────────────────────
    # "survival": steps_survived / episode_steps  -> [0, 1]
    # "food":     cumulative raw food / (episode_steps * n_food_types)
    #             -> may exceed 1.0, but is never negative (the clamp above
    #                relies on f_raw >= 0 in both modes)
    fitness_mode: str = "survival"

    # Append normalised (x, y) to the sensor vector.  Adds 2 to n_in.
    position_sensors: bool = False

    # ── Derived ──────────────────────────────────────────────────────────────
    n_in: int = field(init=False)

    # ─────────────────────────────────────────────────────────────────────────

    def __post_init__(self):
        # Resolve defaults before anything reads them.
        if self.grid_H is None:
            self.grid_H = self.grid_W
        self.n_in = 2 * self.n_food_types + (2 if self.position_sensors else 0)
        if self.sparse_n_active is None:
            self.sparse_n_active = self.N_max // 2

        # Validate the lattice FIRST — reference_costs below builds a mask from
        # these fields and would fail obscurely on bad geometry.
        self._validate_lattice()
        self._validate_io()
        self._validate_integration()
        self._validate_schedule()

        # Derive C0 from the lattice unless supplied.  Conditional so that
        # load_config on an older run restores that run's denominators rather
        # than silently recomputing them.
        if self.C0_edge is None or self.C0_dist is None:
            c0_edge, c0_dist = reference_costs(self.grid_W, self.grid_r, self.grid_H)
            if self.C0_edge is None:
                self.C0_edge = c0_edge
            if self.C0_dist is None:
                self.C0_dist = c0_dist

        # Penalty validation runs last: C0 must be concrete to be checked.
        self._validate_penalties()

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_lattice(self):
        if self.grid_W < 1 or self.grid_H < 1:
            raise ValueError(
                f"grid dimensions must be positive, got {self.grid_W}x{self.grid_H}"
            )
        if self.grid_r < 1:
            raise ValueError(f"grid_r must be >= 1, got {self.grid_r}")
        if self.N_max != self.grid_W * self.grid_H:
            raise ValueError(
                f"N_max ({self.N_max}) must equal grid_W * grid_H "
                f"({self.grid_W} * {self.grid_H} = {self.grid_W * self.grid_H}). "
                "A mismatch is otherwise silent: local_mask returns the wrong "
                "shape and the failure surfaces much later as a broadcast error."
            )
        if self.init_mode not in INIT_MODES:
            raise ValueError(
                f"init_mode must be one of {INIT_MODES}, got {self.init_mode!r}. "
                "Caught here rather than defaulted at init time — a typo'd arm "
                "name would otherwise run the wrong experiment and log it as correct."
            )
        if not (0 < self.sparse_n_active <= self.N_max):
            raise ValueError(
                f"sparse_n_active ({self.sparse_n_active}) must be in (0, N_max]"
            )

    def _validate_io(self):
        if self.n_out < 1:
            raise ValueError(f"n_out must be >= 1, got {self.n_out}")
        if self.n_food_types < 1:
            raise ValueError(f"n_food_types must be >= 1, got {self.n_food_types}")
        if self.n_in + self.n_out >= self.N_max:
            raise ValueError(
                f"n_in ({self.n_in}) + n_out ({self.n_out}) must be < N_max "
                f"({self.N_max}) — the lattice needs at least one hidden slot"
            )

    def _validate_penalties(self):
        for name in ("edge_frac", "dist_frac", "act_frac"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(
                    f"{name} must be >= 0, got {value}. A negative fraction "
                    "would REWARD wiring rather than penalise it."
                )
        for name in ("C0_edge", "C0_dist", "C0_act"):
            value = getattr(self, name)
            if value is None or value <= 0.0:
                raise ValueError(
                    f"{name} must be > 0, got {value} — it is a denominator"
                )

    def _validate_integration(self):
        for name in ("tau_e_range", "tau_fsi_range", "tau_sii_range"):
            lo, hi = getattr(self, name)
            if lo <= 0 or hi <= lo:
                raise ValueError(f"{name} must satisfy 0 < lo < hi, got ({lo}, {hi})")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")

        tau_min = min(self.tau_e_range[0], self.tau_fsi_range[0], self.tau_sii_range[0])
        if self.dt > tau_min:
            raise ValueError(
                f"dt ({self.dt}) must be <= the smallest tau ({tau_min}) for "
                "Euler stability"
            )

    def _validate_schedule(self):
        if self.penalty_warmup_gens < 0:
            raise ValueError(
                f"penalty_warmup_gens must be >= 0, got {self.penalty_warmup_gens}"
            )
        if self.mutation_warmup_scale < 1.0:
            raise ValueError(
                f"mutation_warmup_scale must be >= 1.0, got "
                f"{self.mutation_warmup_scale} — below 1 would NARROW exploration "
                "during warm-up, the opposite of the intent"
            )
        if self.penalty_cycle_gens < 0 or self.penalty_cycle_free_gens < 0:
            raise ValueError("penalty cycle lengths must be >= 0")
        if self.penalty_cycle_gens > 0 or self.penalty_cycle_free_gens > 0:
            if self.penalty_cycle_free_gens >= self.penalty_cycle_gens:
                raise ValueError(
                    f"penalty_cycle_free_gens ({self.penalty_cycle_free_gens}) "
                    f"must be < penalty_cycle_gens ({self.penalty_cycle_gens}) — "
                    "otherwise the penalty is never on and the run silently "
                    "becomes an unpenalised baseline"
                )
        if self.population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {self.population_size}")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError(
                f"tournament_size ({self.tournament_size}) must be in "
                f"[1, population_size ({self.population_size})]"
            )
        if self.fitness_mode not in FITNESS_MODES:
            raise ValueError(
                f"fitness_mode must be one of {FITNESS_MODES}, "
                f"got {self.fitness_mode!r}"
            )

    # ── Convenience ──────────────────────────────────────────────────────────

    def tau_range(self, neuron_type: int) -> Tuple[float, float]:
        return (self.tau_e_range, self.tau_fsi_range, self.tau_sii_range)[neuron_type]

    @property
    def n_possible_edges(self) -> int:
        """Directed pairs excluding self-edges — the density denominator.

        Note this is N*(N-1), NOT the lattice mask size: a fresh 8x8 r=2
        lattice reads as 27% dense, not 100%, and add_edge is unmasked so the
        active edge count can exceed the mask.
        """
        return self.N_max * (self.N_max - 1)