"""
Tests — Logging layer.

Ported from ctrnn_evo.  One substantive change: the genome has six fields,
not seven.  load_genome reconstructs positionally via Genome(*children), so a
stale seven-field archive must fail loudly rather than bind arrays to the
wrong fields.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ctrnn_lattice_evo import Config, WorldConfig, Genome
from ctrnn_lattice_evo.genome import grid_genome
from ctrnn_lattice_evo.mutation import MutationRates
from ctrnn_lattice_evo.evolution import (
    run_evolution, init_population, eval_population, compute_fitness,
)
from ctrnn_lattice_evo.logger import (
    make_run_dir,
    save_config, load_config,
    save_genome, load_genome,
    append_history, load_history,
    make_logger,
    save_training_state, load_training_state, latest_state_checkpoint,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_base(tmp_path):
    return tmp_path / "runs"


@pytest.fixture(scope="module")
def cfg():
    return Config(N_max=16, n_out=2, grid_W=4, grid_H=4, grid_r=1,
                  K=4, population_size=10, tournament_size=3)


@pytest.fixture(scope="module")
def wcfg():
    return WorldConfig(episode_steps=50)


@pytest.fixture(scope="module")
def rates():
    return MutationRates(add_node_prob=0.0, remove_node_prob=0.0)


@pytest.fixture(scope="module")
def genome(cfg):
    return grid_genome(jax.random.PRNGKey(0), cfg)


# ── 1. make_run_dir ───────────────────────────────────────────────────────────

def test_make_run_dir_creates_run_directory(tmp_base):
    run_dir = make_run_dir(tmp_base)
    assert run_dir.exists() and run_dir.is_dir()


def test_make_run_dir_creates_checkpoints_subdir(tmp_base):
    run_dir = make_run_dir(tmp_base)
    assert (run_dir / "checkpoints").is_dir()


def test_make_run_dir_unique_without_run_id(tmp_base):
    import time
    d1 = make_run_dir(tmp_base)
    time.sleep(0.01)
    d2 = make_run_dir(tmp_base)
    assert d1 != d2


def test_make_run_dir_custom_run_id(tmp_base):
    assert "myrun" in make_run_dir(tmp_base, run_id="myrun").name


def test_make_run_dir_creates_base_if_missing(tmp_path):
    assert make_run_dir(tmp_path / "does" / "not" / "exist").exists()


# ── 2. save_config / load_config ──────────────────────────────────────────────

def test_save_config_creates_file(tmp_base, cfg, wcfg, rates):
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    assert (run_dir / "config.json").exists()


def test_save_config_is_valid_json(tmp_base, cfg, wcfg, rates):
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    with open(run_dir / "config.json") as f:
        data = json.load(f)
    assert {"config", "world_config", "mutation_rates"} <= set(data)


def test_load_config_roundtrip_cfg(tmp_base, cfg, wcfg, rates):
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    cfg2, _, _ = load_config(run_dir)
    assert cfg2 == cfg


def test_load_config_roundtrip_wcfg(tmp_base, cfg, wcfg, rates):
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    _, wcfg2, _ = load_config(run_dir)
    assert wcfg2 == wcfg


def test_load_config_roundtrip_rates(tmp_base, cfg, wcfg, rates):
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    _, _, rates2 = load_config(run_dir)
    assert rates2 == rates


def test_load_config_preserves_lattice_shape(tmp_base, wcfg, rates):
    """grid_W/H/r must survive the roundtrip — every mask is derived from them."""
    cfg = Config(N_max=64, n_out=2, grid_W=4, grid_H=16, grid_r=2)
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    cfg2, _, _ = load_config(run_dir)
    assert (cfg2.grid_W, cfg2.grid_H, cfg2.grid_r) == (4, 16, 2)


def test_load_config_preserves_init_mode(tmp_base, wcfg, rates):
    """The arm label.  If this is lost, a uniform run reloads as a grid run."""
    cfg = Config(N_max=16, n_out=2, grid_W=4, grid_H=4, init_mode="uniform")
    run_dir = make_run_dir(tmp_base)
    save_config(run_dir, cfg, wcfg, rates)
    cfg2, _, _ = load_config(run_dir)
    assert cfg2.init_mode == "uniform"


# ── 3. save_genome / load_genome ──────────────────────────────────────────────

def test_save_genome_creates_npz(tmp_base, genome):
    path = make_run_dir(tmp_base) / "g.npz"
    save_genome(path, genome)
    assert path.exists()


def test_load_genome_all_fields(tmp_base, genome):
    """Six fields — `position` is gone."""
    path = make_run_dir(tmp_base) / "genome.npz"
    save_genome(path, genome)
    g2 = load_genome(path)

    assert jnp.array_equal(g2.active_mask, genome.active_mask)
    assert jnp.array_equal(g2.neuron_type, genome.neuron_type)
    assert jnp.allclose(g2.tau, genome.tau)
    assert jnp.allclose(g2.bias, genome.bias)
    assert jnp.allclose(g2.weight_matrix, genome.weight_matrix)
    assert jnp.array_equal(g2.edge_mask, genome.edge_mask)


def test_saved_archive_has_six_fields(tmp_base, genome):
    path = make_run_dir(tmp_base) / "genome.npz"
    save_genome(path, genome)
    assert len(np.load(str(path)).files) == 6
    assert "position" not in np.load(str(path)).files


def test_legacy_seven_field_archive_rejected(tmp_base, genome):
    """A ctrnn_evo archive still carries `position`.  load_genome builds the
    Genome positionally, so a silent load would bind weight_matrix to
    edge_mask.  It must raise instead."""
    path = make_run_dir(tmp_base) / "legacy.npz"
    np.savez(
        str(path),
        active_mask=np.array(genome.active_mask),
        neuron_type=np.array(genome.neuron_type),
        tau=np.array(genome.tau),
        bias=np.array(genome.bias),
        position=np.zeros((16, 2), dtype=np.float32),
        weight_matrix=np.array(genome.weight_matrix),
        edge_mask=np.array(genome.edge_mask),
    )
    with pytest.raises((KeyError, ValueError, TypeError, AssertionError)):
        load_genome(path)


def test_load_genome_returns_genome_instance(tmp_base, genome):
    path = make_run_dir(tmp_base) / "genome.npz"
    save_genome(path, genome)
    assert isinstance(load_genome(path), Genome)


def test_load_genome_edge_mask_stays_boolean(tmp_base, genome):
    """edge_mask is used as a boolean predicate throughout — a float roundtrip
    would still 'work' but silently change edge_count_cost."""
    path = make_run_dir(tmp_base) / "genome.npz"
    save_genome(path, genome)
    assert load_genome(path).edge_mask.dtype == jnp.bool_


# ── 4. append_history / load_history ─────────────────────────────────────────

def test_append_history_single(tmp_base):
    run_dir = make_run_dir(tmp_base)
    append_history(run_dir, {"generation": 0, "max_fitness": 0.5})
    assert load_history(run_dir)[0]["generation"] == 0


def test_append_history_accumulates(tmp_base):
    run_dir = make_run_dir(tmp_base)
    for i in range(5):
        append_history(run_dir, {"generation": i})
    assert [h["generation"] for h in load_history(run_dir)] == list(range(5))


def test_append_history_values_preserved(tmp_base):
    run_dir = make_run_dir(tmp_base)
    stats = {"generation": 7, "max_fitness": 0.9123, "mean_n_edges": 84.0,
             "mean_local_fraction": 0.97}
    append_history(run_dir, stats)
    h = load_history(run_dir)[0]
    assert h["mean_n_edges"] == pytest.approx(84.0)
    assert h["mean_local_fraction"] == pytest.approx(0.97)


def test_load_history_empty_file(tmp_base):
    run_dir = make_run_dir(tmp_base)
    (run_dir / "history.jsonl").touch()
    assert load_history(run_dir) == []


def test_load_history_missing_file(tmp_base):
    assert load_history(make_run_dir(tmp_base)) == []


def test_append_history_is_incremental(tmp_base):
    run_dir = make_run_dir(tmp_base)
    append_history(run_dir, {"generation": 0})
    append_history(run_dir, {"generation": 1})
    lines = (run_dir / "history.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["generation"] == 0
    assert json.loads(lines[1])["generation"] == 1


# ── 5. make_logger ────────────────────────────────────────────────────────────

def _fake_stats(gen: int) -> dict:
    return {
        "generation": gen,
        "max_fitness": 0.8,
        "mean_fitness": 0.6,
        "max_steps": 80,
        "mean_steps": 60.0,
        "mean_n_active": 16.0,
        "mean_edge_cost": 84.0,
        "mean_wiring_cost": 84.0,
    }


def test_make_logger_creates_history_file(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    make_logger(run_dir, checkpoint_every=100, verbose=False)(_fake_stats(0), genome)
    assert (run_dir / "history.jsonl").exists()


def test_make_logger_history_grows_each_call(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=100, verbose=False)
    for i in range(7):
        cb(_fake_stats(i), genome)
    assert len(load_history(run_dir)) == 7


def test_make_logger_writes_best_genome_file(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    make_logger(run_dir, checkpoint_every=100, verbose=False)(_fake_stats(0), genome)
    assert (run_dir / "best_genome.npz").exists()


def test_make_logger_checkpoint_cadence(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=5, verbose=False)
    for i in range(11):
        cb(_fake_stats(i), genome)
    ckpt = run_dir / "checkpoints"
    assert (ckpt / "gen_000000.npz").exists()
    assert (ckpt / "gen_000005.npz").exists()
    assert (ckpt / "gen_000010.npz").exists()
    assert not (ckpt / "gen_000003.npz").exists()


def test_make_logger_checkpoint_content_valid(tmp_base, genome):
    run_dir = make_run_dir(tmp_base)
    make_logger(run_dir, checkpoint_every=1, verbose=False)(_fake_stats(0), genome)
    g2 = load_genome(run_dir / "checkpoints" / "gen_000000.npz")
    assert jnp.allclose(g2.weight_matrix, genome.weight_matrix)


def test_make_logger_verbose_false_no_stdout(tmp_base, genome, capsys):
    run_dir = make_run_dir(tmp_base)
    make_logger(run_dir, checkpoint_every=100, verbose=False)(_fake_stats(0), genome)
    assert capsys.readouterr().out == ""


def test_make_logger_verbose_true_prints(tmp_base, genome, capsys):
    run_dir = make_run_dir(tmp_base)
    make_logger(run_dir, checkpoint_every=100, verbose=True)(_fake_stats(42), genome)
    assert "42" in capsys.readouterr().out


# ── 6. run_evolution integration ─────────────────────────────────────────────

def test_run_evolution_callback_receives_genome(cfg, wcfg, rates):
    received = []
    run_evolution(jax.random.PRNGKey(99), 3, cfg, wcfg, rates, n_evals=2,
                  callback=lambda s, g: received.append((s["generation"], g)))
    assert len(received) == 3
    for _, g in received:
        assert isinstance(g, Genome)
        assert g.weight_matrix.shape == (cfg.N_max, cfg.N_max)


def test_run_evolution_with_logger_populates_files(cfg, wcfg, rates, tmp_base):
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=2, verbose=False)
    run_evolution(jax.random.PRNGKey(100), 5, cfg, wcfg, rates, n_evals=2, callback=cb)
    assert (run_dir / "history.jsonl").exists()
    assert (run_dir / "best_genome.npz").exists()
    assert len(load_history(run_dir)) == 5


def test_run_evolution_with_logger_checkpoint_files(cfg, wcfg, rates, tmp_base):
    run_dir = make_run_dir(tmp_base)
    cb = make_logger(run_dir, checkpoint_every=2, verbose=False)
    run_evolution(jax.random.PRNGKey(101), 5, cfg, wcfg, rates, n_evals=2, callback=cb)
    ckpt = run_dir / "checkpoints"
    assert (ckpt / "gen_000000.npz").exists()
    assert (ckpt / "gen_000002.npz").exists()
    assert (ckpt / "gen_000004.npz").exists()


# ── 7. Training-state snapshots ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def pop(cfg):
    wcfg_small = WorldConfig(episode_steps=20)
    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    population = init_population(k1, cfg)
    steps, c_acts, raw_food = eval_population(k2, population, cfg, wcfg_small, n_evals=1)
    fitness = compute_fitness(steps, c_acts, raw_food, population, cfg, wcfg_small)
    return population, fitness, steps


def test_save_training_state_creates_file(tmp_base, pop):
    population, fitness, steps = pop
    path = make_run_dir(tmp_base) / "checkpoints" / "state_gen_000100.npz"
    save_training_state(path, population, fitness, steps, jax.random.PRNGKey(7), 100)
    assert path.exists()


def test_load_training_state_generation(tmp_base, pop):
    population, fitness, steps = pop
    path = make_run_dir(tmp_base) / "checkpoints" / "state_gen_000050.npz"
    save_training_state(path, population, fitness, steps, jax.random.PRNGKey(8), 50)
    assert load_training_state(path)[4] == 50


def test_load_training_state_pop_shape(tmp_base, pop, cfg):
    population, fitness, steps = pop
    path = make_run_dir(tmp_base) / "checkpoints" / "state_gen_000010.npz"
    save_training_state(path, population, fitness, steps, jax.random.PRNGKey(9), 10)
    pop2 = load_training_state(path)[0]
    assert pop2.weight_matrix.shape == (cfg.population_size, cfg.N_max, cfg.N_max)


def test_load_training_state_fitness_close(tmp_base, pop):
    population, fitness, steps = pop
    path = make_run_dir(tmp_base) / "checkpoints" / "state_gen_000020.npz"
    save_training_state(path, population, fitness, steps, jax.random.PRNGKey(10), 20)
    assert jnp.allclose(load_training_state(path)[1], fitness, atol=1e-6)


def test_load_training_state_key_preserved(tmp_base, pop):
    population, fitness, steps = pop
    key = jax.random.PRNGKey(2025)
    path = make_run_dir(tmp_base) / "checkpoints" / "state_gen_000030.npz"
    save_training_state(path, population, fitness, steps, key, 30)
    assert jnp.array_equal(load_training_state(path)[3], key)


def test_load_training_state_edge_mask_preserved(tmp_base, pop):
    """Topology is the whole experiment — it must survive resume exactly."""
    population, fitness, steps = pop
    path = make_run_dir(tmp_base) / "checkpoints" / "state_gen_000040.npz"
    save_training_state(path, population, fitness, steps, jax.random.PRNGKey(11), 40)
    pop2 = load_training_state(path)[0]
    assert jnp.array_equal(pop2.edge_mask, population.edge_mask)
    assert jnp.array_equal(pop2.active_mask, population.active_mask)


def test_latest_state_checkpoint_none_when_empty(tmp_base):
    assert latest_state_checkpoint(make_run_dir(tmp_base)) is None


def test_latest_state_checkpoint_returns_latest(tmp_base, pop):
    population, fitness, steps = pop
    run_dir = make_run_dir(tmp_base)
    for gen in (100, 200, 50):
        save_training_state(run_dir / "checkpoints" / f"state_gen_{gen:06d}.npz",
                            population, fitness, steps, jax.random.PRNGKey(gen), gen)
    assert latest_state_checkpoint(run_dir).name == "state_gen_000200.npz"


# ── 8. Resume ────────────────────────────────────────────────────────────────

def test_run_evolution_saves_state_files(cfg, wcfg, rates, tmp_base):
    ckpt_dir = make_run_dir(tmp_base) / "checkpoints"
    run_evolution(jax.random.PRNGKey(200), 6, cfg, wcfg, rates, n_evals=1,
                  state_checkpoint_dir=ckpt_dir, state_checkpoint_every=2)
    assert (ckpt_dir / "state_gen_000002.npz").exists()
    assert (ckpt_dir / "state_gen_000004.npz").exists()
    assert not (ckpt_dir / "state_gen_000001.npz").exists()


def test_run_evolution_resume_starts_at_correct_gen(cfg, wcfg, rates, tmp_base):
    ckpt_dir = make_run_dir(tmp_base) / "checkpoints"
    run_evolution(jax.random.PRNGKey(300), 4, cfg, wcfg, rates, n_evals=1,
                  state_checkpoint_dir=ckpt_dir, state_checkpoint_every=2)
    _, _, resumed = run_evolution(
        jax.random.PRNGKey(999), 4, cfg, wcfg, rates, n_evals=1,
        resume_from=ckpt_dir / "state_gen_000002.npz",
    )
    assert resumed[0]["generation"] == 2
    assert len(resumed) == 2


def test_resume_matches_uninterrupted_run(cfg, wcfg, rates, tmp_base):
    """The README claims resume restores the RNG key so the sequence is
    identical to an uninterrupted run.  Nothing tested that claim."""
    ckpt_dir = make_run_dir(tmp_base) / "checkpoints"
    _, _, full = run_evolution(
        jax.random.PRNGKey(500), 4, cfg, wcfg, rates, n_evals=1,
        state_checkpoint_dir=ckpt_dir, state_checkpoint_every=2)

    _, _, resumed = run_evolution(
        jax.random.PRNGKey(12345), 4, cfg, wcfg, rates, n_evals=1,
        resume_from=ckpt_dir / "state_gen_000002.npz")

    for a, b in zip(full[2:], resumed):
        assert a["generation"] == b["generation"]
        assert a["max_fitness"] == pytest.approx(b["max_fitness"], rel=1e-5), \
            f"resume diverged at gen {a['generation']}"