"""
Tests for config.py.

Config now carries invariants every other fixture depends on:

  * N_max must equal grid_W * grid_H, or every mask is the wrong shape
  * C0 references must be derived from the lattice, not inherited constants
  * dt <= min(tau) is the Euler stability condition, and tau_fsi_range[0]=1.0
    against dt=0.5 leaves only a 2x margin

reference_costs lives in topology.py and takes plain ints — Config calls it,
never the reverse, so config -> topology is a single downward edge.
"""

from __future__ import annotations

import dataclasses

import pytest

from ctrnn_lattice_evo import Config
from ctrnn_lattice_evo.topology import expected_edges, reference_costs


# ── Lattice / capacity invariant ─────────────────────────────────────────────

def test_square_lattice_accepted():
    cfg = Config(N_max=64, grid_W=8, grid_H=8, grid_r=2)
    assert cfg.grid_W * cfg.grid_H == cfg.N_max


def test_rectangular_lattice_accepted():
    cfg = Config(N_max=64, grid_W=4, grid_H=16, grid_r=2)
    assert cfg.grid_W * cfg.grid_H == cfg.N_max


@pytest.mark.parametrize("N,W,H", [(16, 8, 8), (64, 4, 4), (63, 8, 8), (100, 8, 8)])
def test_mismatched_lattice_rejected(N, W, H):
    """A mismatch is silent otherwise: local_mask returns the wrong shape and
    the failure surfaces much later as a broadcast error."""
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=N, grid_W=W, grid_H=H, grid_r=1)


def test_grid_h_defaults_to_square():
    cfg = Config(N_max=64, grid_W=8, grid_r=2)
    assert cfg.grid_H == 8


def test_radius_must_be_positive():
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4, grid_r=0)


# ── Derived n_in ─────────────────────────────────────────────────────────────

def test_n_in_single_food_type():
    cfg = Config(N_max=16, grid_W=4, grid_H=4, n_food_types=1, position_sensors=False)
    assert cfg.n_in == 2


def test_n_in_with_position_sensors():
    cfg = Config(N_max=16, grid_W=4, grid_H=4, n_food_types=1, position_sensors=True)
    assert cfg.n_in == 4


def test_io_fits_in_lattice():
    cfg = Config(N_max=16, grid_W=4, grid_H=4, n_out=2)
    assert cfg.n_in + cfg.n_out < cfg.N_max


def test_oversized_io_rejected():
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=4, grid_W=2, grid_H=2, n_out=3, n_food_types=1)


# ── C0 calibration ───────────────────────────────────────────────────────────

def test_c0_matches_topology_reference_costs():
    """Config must store exactly what topology computes — one definition, and
    Config's job is to hold and serialise it, not to reimplement it."""
    cfg = Config(N_max=64, grid_W=8, grid_H=8, grid_r=2)
    c0_edge, c0_dist = reference_costs(cfg.grid_W, cfg.grid_r, cfg.grid_H)
    assert cfg.C0_edge == pytest.approx(c0_edge)
    assert cfg.C0_dist == pytest.approx(c0_dist)


def test_c0_edge_matches_lattice_edge_count():
    """C0_edge is the lattice's own edge count, so edge_frac reads as
    'fraction of fitness surrendered at full local connectivity'."""
    cfg = Config(N_max=64, grid_W=8, grid_H=8, grid_r=2)
    assert cfg.C0_edge == pytest.approx(expected_edges(8, 2, 8))


def test_c0_production_values():
    """Pinned: 8x8 r=2 is the lattice the whole frac sweep is calibrated on."""
    cfg = Config(N_max=64, grid_W=8, grid_H=8, grid_r=2)
    assert cfg.C0_edge == pytest.approx(1092.0)
    assert cfg.C0_dist == pytest.approx(1764.0)


def test_c0_rectangular_values():
    cfg = Config(N_max=64, grid_W=4, grid_H=16, grid_r=2)
    assert cfg.C0_edge == pytest.approx(972.0)
    assert cfg.C0_dist == pytest.approx(1548.0)


def test_c0_edge_scales_with_radius():
    small = Config(N_max=64, grid_W=8, grid_H=8, grid_r=1)
    large = Config(N_max=64, grid_W=8, grid_H=8, grid_r=2)
    assert large.C0_edge > small.C0_edge
    assert large.C0_dist > small.C0_dist


def test_c0_edge_scales_with_lattice_size():
    small = Config(N_max=16, grid_W=4, grid_H=4, grid_r=1)
    large = Config(N_max=64, grid_W=8, grid_H=8, grid_r=1)
    assert large.C0_edge > small.C0_edge


def test_c0_collinear_at_radius_one():
    """At r=1 every edge has length 1, so the two denominators coincide and
    dist_frac is not an independent axis."""
    cfg = Config(N_max=64, grid_W=8, grid_H=8, grid_r=1)
    assert cfg.C0_dist == pytest.approx(cfg.C0_edge)


def test_c0_not_legacy_constants():
    """ctrnn_evo's C0_edge=154 / C0_wiring=77 were measured on a sparse random
    init.  A lattice is several times those, so inheriting them applies a
    crushing penalty at generation 0 that reads as 'locality fails'."""
    cfg = Config(N_max=64, grid_W=8, grid_H=8, grid_r=2)
    assert cfg.C0_edge != pytest.approx(154.0, rel=0.01)
    assert cfg.C0_dist != pytest.approx(77.0, rel=0.01)


def test_c0_wiring_field_is_gone():
    """Renamed to C0_dist to pair with dist_frac."""
    assert not hasattr(Config(N_max=16, grid_W=4, grid_H=4), "C0_wiring")


def test_c0_act_is_one():
    """c_act is mean |tanh(v)| over active neurons, so its ceiling is exactly
    1.0 regardless of network size — no calibration needed."""
    assert Config(N_max=16, grid_W=4, grid_H=4).C0_act == pytest.approx(1.0)


def test_c0_values_are_positive():
    cfg = Config(N_max=16, grid_W=4, grid_H=4, grid_r=1)
    assert cfg.C0_edge > 0 and cfg.C0_dist > 0 and cfg.C0_act > 0


def test_c0_edge_can_be_overridden():
    """An explicit value must survive __post_init__, so load_config on an older
    run restores that run's denominators rather than recomputing them."""
    cfg = Config(N_max=16, grid_W=4, grid_H=4, grid_r=1, C0_edge=999.0)
    assert cfg.C0_edge == pytest.approx(999.0)


def test_c0_partial_override_derives_the_other():
    cfg = Config(N_max=16, grid_W=4, grid_H=4, grid_r=1, C0_edge=999.0)
    assert cfg.C0_edge == pytest.approx(999.0)
    assert cfg.C0_dist == pytest.approx(84.0)


def test_nonpositive_c0_rejected():
    """A zero denominator would divide by zero inside adjusted_fitness."""
    with pytest.raises((AssertionError, ValueError, ZeroDivisionError)):
        Config(N_max=16, grid_W=4, grid_H=4, grid_r=1, C0_edge=0.0)


# ── Penalty fractions ────────────────────────────────────────────────────────

def test_fracs_default_to_zero():
    cfg = Config(N_max=16, grid_W=4, grid_H=4)
    assert cfg.edge_frac == 0.0 and cfg.dist_frac == 0.0 and cfg.act_frac == 0.0


def test_absolute_lambda_mode_is_gone():
    """One code path only.  The dual absolute/proportional branch in
    ctrnn_evo's adjusted_fitness is where the sign-flip hides."""
    cfg = Config(N_max=16, grid_W=4, grid_H=4)
    for name in ("lambda_edge", "lambda_dist", "lambda_act", "lambda_conn"):
        assert not hasattr(cfg, name), f"{name} should have been removed"


def test_negative_frac_rejected():
    """A negative frac would REWARD wiring — an easy typo to miss."""
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4, edge_frac=-0.1)


# ── Addition distance kernel ─────────────────────────────────────────────────

def test_add_kernel_lambda_defaults_to_uniform():
    assert Config(N_max=16, grid_W=4, grid_H=4).add_kernel_lambda == 0.0


def test_add_kernel_lambda_accepts_positive_finite():
    cfg = Config(N_max=16, grid_W=4, grid_H=4, add_kernel_lambda=1.0)
    assert cfg.add_kernel_lambda == pytest.approx(1.0)


def test_add_kernel_lambda_accepts_inf():
    """inf is a legal 'uniform, but explicit' value alongside 0."""
    cfg = Config(N_max=16, grid_W=4, grid_H=4, add_kernel_lambda=float("inf"))
    assert cfg.add_kernel_lambda == float("inf")


def test_negative_add_kernel_lambda_rejected():
    """distance_kernel's own <= 0 guard would silently fold a negative value
    into 'uniform' — reject it here instead of hiding a typo."""
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4, add_kernel_lambda=-0.5)


def test_nan_add_kernel_lambda_rejected():
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4, add_kernel_lambda=float("nan"))


# ── Init mode ────────────────────────────────────────────────────────────────

def test_init_mode_defaults_to_grid():
    assert Config(N_max=16, grid_W=4, grid_H=4).init_mode == "grid"


@pytest.mark.parametrize("mode", ["grid", "uniform", "sparse"])
def test_valid_init_modes_accepted(mode):
    assert Config(N_max=16, grid_W=4, grid_H=4, init_mode=mode).init_mode == mode


def test_unknown_init_mode_rejected():
    """Caught at config time, not silently defaulted at init time — a typo'd
    arm name would otherwise run the wrong experiment and log it as correct."""
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4, init_mode="lattice")


def test_sparse_n_active_defaults_below_capacity():
    """The sparse arm starts small and grows — that is the regime the lattice
    is being compared against."""
    cfg = Config(N_max=64, grid_W=8, grid_H=8)
    assert 0 < cfg.sparse_n_active < cfg.N_max


def test_sparse_n_active_can_be_set():
    cfg = Config(N_max=64, grid_W=8, grid_H=8, sparse_n_active=20)
    assert cfg.sparse_n_active == 20


# ── Integration stability ────────────────────────────────────────────────────

def test_dt_within_euler_stability():
    """dt <= min tau across all three type ranges.  tau_fsi_range[0]=1.0 vs
    dt=0.5 is the binding constraint and leaves only a 2x margin."""
    cfg = Config(N_max=16, grid_W=4, grid_H=4)
    tau_min = min(cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0])
    assert cfg.dt <= tau_min


def test_oversized_dt_rejected():
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4, dt=2.0, tau_fsi_range=(1.0, 15.0))


def test_tau_ranges_are_ordered():
    cfg = Config(N_max=16, grid_W=4, grid_H=4)
    for lo, hi in (cfg.tau_e_range, cfg.tau_fsi_range, cfg.tau_sii_range):
        assert lo > 0 and hi > lo


def test_inverted_tau_range_rejected():
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4, tau_e_range=(100.0, 10.0))


def test_tau_range_lookup_by_type():
    cfg = Config(N_max=16, grid_W=4, grid_H=4)
    assert cfg.tau_range(0) == cfg.tau_e_range
    assert cfg.tau_range(1) == cfg.tau_fsi_range
    assert cfg.tau_range(2) == cfg.tau_sii_range


# ── Warm-up and cyclic loosening schedule ────────────────────────────────────

def test_warmup_defaults_off():
    cfg = Config(N_max=16, grid_W=4, grid_H=4)
    assert cfg.penalty_warmup_gens == 0
    assert cfg.mutation_warmup_scale == 1.0


def test_cycle_free_window_must_fit_inside_cycle():
    """free_gens >= cycle_gens means the penalty is never on — the run would
    silently become an unpenalised baseline."""
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4,
               penalty_cycle_gens=100, penalty_cycle_free_gens=100)


def test_valid_cycle_accepted():
    cfg = Config(N_max=16, grid_W=4, grid_H=4, penalty_warmup_gens=200,
                 penalty_cycle_gens=300, penalty_cycle_free_gens=100)
    assert cfg.penalty_cycle_free_gens < cfg.penalty_cycle_gens


def test_mutation_warmup_scale_at_least_one():
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4, mutation_warmup_scale=0.5)


def test_negative_warmup_rejected():
    with pytest.raises((AssertionError, ValueError)):
        Config(N_max=16, grid_W=4, grid_H=4, penalty_warmup_gens=-10)


# ── Serialisation ────────────────────────────────────────────────────────────

def test_config_is_dataclass_serialisable():
    d = dataclasses.asdict(Config(N_max=64, grid_W=8, grid_H=8, grid_r=2))
    assert d["grid_W"] == 8 and d["grid_r"] == 2


def test_c0_is_serialised():
    """config.json is the record of what edge_frac=0.2 actually meant for a
    given run — the denominators have to be in it."""
    d = dataclasses.asdict(Config(N_max=64, grid_W=8, grid_H=8, grid_r=2))
    assert d["C0_edge"] == pytest.approx(1092.0)
    assert d["C0_dist"] == pytest.approx(1764.0)


def test_config_roundtrip_through_asdict():
    """load_config reconstructs from this dict — n_in is derived and must be
    stripped, or __post_init__ collides with it.  Concrete C0 values must
    round-trip unchanged rather than being recomputed."""
    cfg = Config(N_max=64, grid_W=8, grid_H=8, grid_r=2, edge_frac=0.2)
    d = dataclasses.asdict(cfg)
    d.pop("n_in", None)
    assert Config(**d) == cfg


def test_roundtrip_preserves_overridden_c0():
    cfg = Config(N_max=16, grid_W=4, grid_H=4, grid_r=1, C0_edge=999.0)
    d = dataclasses.asdict(cfg)
    d.pop("n_in", None)
    assert Config(**d).C0_edge == pytest.approx(999.0)


def test_config_equality_includes_lattice():
    a = Config(N_max=64, grid_W=8, grid_H=8, grid_r=1)
    b = Config(N_max=64, grid_W=8, grid_H=8, grid_r=2)
    assert a != b


def test_config_equality_includes_init_mode():
    """The arm label.  If runs compared equal across arms, a uniform run could
    reload as a grid run."""
    a = Config(N_max=16, grid_W=4, grid_H=4, init_mode="grid")
    b = Config(N_max=16, grid_W=4, grid_H=4, init_mode="uniform")
    assert a != b


# ── Production configuration ─────────────────────────────────────────────────

def test_production_config_is_coherent():
    """The 8x8 r=2 lattice the experiment actually runs on."""
    cfg = Config(N_max=64, n_out=2, grid_W=8, grid_H=8, grid_r=2)
    assert cfg.N_max == 64
    assert cfg.C0_edge == pytest.approx(1092.0)
    assert cfg.n_in + cfg.n_out <= cfg.N_max
    assert cfg.dt <= min(cfg.tau_e_range[0], cfg.tau_fsi_range[0], cfg.tau_sii_range[0])