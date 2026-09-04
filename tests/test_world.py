"""
World simulator tests.

Written before the implementation (TDD).  These tests define the required
interface and all validation gates:

  - Gradient-following controller survives the full episode
  - Random-walk controller reliably starves
  - Energy economics balance (metabolism, eating, movement cost)
  - Food hotspots drift at the intended rate
  - Boundary reflection keeps the agent inside the arena
  - Difficulty band: clear fitness gap between good and bad controllers
  - Multi-food-type: separate energy resources, separate sensor channels,
    agent dies when any energy type hits zero

Expected interface in ctrnn_lattice_evo.world:

    WorldConfig   dataclass  (n_food_types, n_food, ...)
    WorldState    dataclass (JAX pytree)
        agent_energy : [n_food_types]
        hotspot_pos  : [n_food_types, n_food, 2]

    reset_world(key, wcfg)                     -> WorldState
    step_world(state, action, wcfg)            -> WorldState
    sensor_readout(state, wcfg)                -> jnp.ndarray [2 * n_food_types]
    food_at(pos, hotspot_pos, wcfg)            -> float  (single type's hotspots)
    run_episode(key, controller_fn, wcfg)      -> (WorldState, int)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ctrnn_lattice_evo.world import (
    WorldConfig,
    WorldState,
    reset_world,
    step_world,
    sensor_readout,
    food_at,
    run_episode,
)
from ctrnn_lattice_evo.controllers import random_walk, nearest_hotspot


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def wcfg():
    return WorldConfig()


@pytest.fixture
def wcfg2():
    """Two food types."""
    return WorldConfig(n_food_types=2)


@pytest.fixture
def state(wcfg):
    return reset_world(jax.random.PRNGKey(0), wcfg)


@pytest.fixture
def state2(wcfg2):
    return reset_world(jax.random.PRNGKey(0), wcfg2)


# ── WorldConfig ───────────────────────────────────────────────────────────────

class TestWorldConfig:
    def test_has_required_fields(self, wcfg):
        assert hasattr(wcfg, "arena_size")
        assert hasattr(wcfg, "n_food_types")
        assert hasattr(wcfg, "n_food")
        assert hasattr(wcfg, "hotspot_sigma")
        assert hasattr(wcfg, "hotspot_drift")
        assert hasattr(wcfg, "init_energy")
        assert hasattr(wcfg, "metabolism")
        assert hasattr(wcfg, "move_cost")
        assert hasattr(wcfg, "eat_rate")
        assert hasattr(wcfg, "max_energy")
        assert hasattr(wcfg, "max_speed")
        assert hasattr(wcfg, "episode_steps")

    def test_default_n_food_types_is_one(self, wcfg):
        assert wcfg.n_food_types == 1

    def test_positive_values(self, wcfg):
        assert wcfg.arena_size > 0
        assert wcfg.n_food > 0
        assert wcfg.n_food_types > 0
        assert wcfg.hotspot_sigma > 0
        assert wcfg.metabolism > 0
        assert wcfg.eat_rate > 0
        assert wcfg.max_speed > 0
        assert wcfg.episode_steps > 0


# ── reset_world ───────────────────────────────────────────────────────────────

class TestResetWorld:
    def test_returns_world_state(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        assert isinstance(state, WorldState)

    def test_agent_pos_in_arena(self, wcfg):
        state = reset_world(jax.random.PRNGKey(1), wcfg)
        assert jnp.all(state.agent_pos >= 0)
        assert jnp.all(state.agent_pos <= wcfg.arena_size)

    def test_energy_shape_matches_n_food_types(self, wcfg):
        state = reset_world(jax.random.PRNGKey(2), wcfg)
        assert state.agent_energy.shape == (wcfg.n_food_types,)

    def test_energy_initialised_correctly(self, wcfg):
        state = reset_world(jax.random.PRNGKey(2), wcfg)
        assert jnp.allclose(state.agent_energy, wcfg.init_energy)

    def test_hotspot_shape(self, wcfg):
        state = reset_world(jax.random.PRNGKey(3), wcfg)
        assert state.hotspot_pos.shape == (wcfg.n_food_types, wcfg.n_food, 2)

    def test_hotspot_positions_in_arena(self, wcfg):
        state = reset_world(jax.random.PRNGKey(3), wcfg)
        assert jnp.all(state.hotspot_pos >= 0)
        assert jnp.all(state.hotspot_pos <= wcfg.arena_size)

    def test_step_counter_zero(self, wcfg):
        state = reset_world(jax.random.PRNGKey(4), wcfg)
        assert int(state.step) == 0

    def test_different_keys_give_different_states(self, wcfg):
        s1 = reset_world(jax.random.PRNGKey(5), wcfg)
        s2 = reset_world(jax.random.PRNGKey(6), wcfg)
        assert not jnp.array_equal(s1.agent_pos, s2.agent_pos)

    def test_is_jax_pytree(self, wcfg):
        state = reset_world(jax.random.PRNGKey(7), wcfg)
        leaves, treedef = jax.tree_util.tree_flatten(state)
        state2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.array_equal(state.agent_pos, state2.agent_pos)

    def test_two_food_types_energy_shape(self, wcfg2):
        state = reset_world(jax.random.PRNGKey(0), wcfg2)
        assert state.agent_energy.shape == (2,)
        assert state.hotspot_pos.shape == (2, wcfg2.n_food, 2)


# ── food_at ───────────────────────────────────────────────────────────────────

class TestFoodAt:
    def test_peak_at_hotspot_centre(self, wcfg):
        hotspot_pos = jnp.array([[50.0, 50.0]])
        wcfg_single = WorldConfig(n_food=1)
        density = food_at(jnp.array([50.0, 50.0]), hotspot_pos, wcfg_single)
        assert float(density) == pytest.approx(1.0, abs=1e-4)

    def test_decays_with_distance(self, wcfg):
        hotspot_pos = jnp.array([[50.0, 50.0]])
        wcfg_single = WorldConfig(n_food=1)
        near  = food_at(jnp.array([51.0, 50.0]), hotspot_pos, wcfg_single)
        far   = food_at(jnp.array([70.0, 50.0]), hotspot_pos, wcfg_single)
        assert float(near) > float(far)

    def test_nonnegative(self, state, wcfg):
        density = food_at(state.agent_pos, state.hotspot_pos[0], wcfg)
        assert float(density) >= 0.0

    def test_multiple_hotspots_add(self):
        wcfg2 = WorldConfig(n_food=2)
        hotspot_pos = jnp.array([[50.0, 50.0], [50.0, 50.0]])
        wcfg1 = WorldConfig(n_food=1)
        hotspot_pos1 = jnp.array([[50.0, 50.0]])
        d2 = food_at(jnp.array([50.0, 50.0]), hotspot_pos,  wcfg2)
        d1 = food_at(jnp.array([50.0, 50.0]), hotspot_pos1, wcfg1)
        assert float(d2) == pytest.approx(float(d1) * 2, rel=1e-4)


# ── sensor_readout ────────────────────────────────────────────────────────────

class TestSensorReadout:
    def test_shape_single_type(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        assert sensors.shape == (2,)  # [food_0, energy_0]

    def test_shape_two_types(self, state2, wcfg2):
        sensors = sensor_readout(state2, wcfg2)
        assert sensors.shape == (4,)  # [food_0, food_1, energy_0, energy_1]

    def test_food_sensor_in_range(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        assert 0.0 <= float(sensors[0]) <= 1.0

    def test_energy_sensor_matches_state(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        assert float(sensors[1]) == pytest.approx(float(state.agent_energy[0]), rel=1e-4)

    def test_food_sensor_high_at_hotspot(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        centre = state.hotspot_pos[0, 0]  # first type, first hotspot
        at_hotspot = WorldState(
            agent_pos=centre,
            agent_energy=state.agent_energy,
            hotspot_pos=state.hotspot_pos,
            step=state.step,
            rng_key=state.rng_key,
        )
        sensors = sensor_readout(at_hotspot, wcfg)
        assert float(sensors[0]) > 0.5, "Food sensor should be high at hotspot centre"

    def test_food_sensor_low_far_from_hotspots(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        corner = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=state.agent_energy,
            hotspot_pos=jnp.full((wcfg.n_food_types, wcfg.n_food, 2), wcfg.arena_size * 0.75),
            step=state.step,
            rng_key=state.rng_key,
        )
        sensors = sensor_readout(corner, wcfg)
        assert float(sensors[0]) < 0.1, "Food sensor should be low far from hotspots"

    def test_two_type_food_sensors_are_independent(self, wcfg2):
        """Type A sensor responds to type A hotspots, type B to type B."""
        state = reset_world(jax.random.PRNGKey(0), wcfg2)
        # Place agent at a type-A hotspot, far from all type-B hotspots
        centre_a = state.hotspot_pos[0, 0]
        hotspots = state.hotspot_pos.at[1].set(
            jnp.full((wcfg2.n_food, 2), wcfg2.arena_size)
        )
        at_a = WorldState(
            agent_pos=centre_a,
            agent_energy=state.agent_energy,
            hotspot_pos=hotspots,
            step=state.step,
            rng_key=state.rng_key,
        )
        sensors = sensor_readout(at_a, wcfg2)
        # sensors: [food_A, food_B, energy_A, energy_B]
        assert float(sensors[0]) > 0.5, "Type-A food sensor should be high at type-A hotspot"
        assert float(sensors[1]) < 0.1, "Type-B food sensor should be low far from type-B hotspots"


# ── step_world ────────────────────────────────────────────────────────────────

class TestStepWorld:
    def test_step_counter_increments(self, state, wcfg):
        action = jnp.zeros(2)
        s2 = step_world(state, action, wcfg)
        assert int(s2.step) == int(state.step) + 1

    def test_stationary_agent_loses_energy(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        barren = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.full((wcfg.n_food_types,), 0.5),
            hotspot_pos=jnp.full((wcfg.n_food_types, wcfg.n_food, 2), wcfg.arena_size),
            step=state.step,
            rng_key=state.rng_key,
        )
        s2 = step_world(barren, jnp.zeros(2), wcfg)
        assert jnp.all(s2.agent_energy < 0.5), "Stationary agent in barren area must lose energy"

    def test_eating_restores_energy(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        centre = state.hotspot_pos[0, 0]  # first type, first hotspot
        at_hotspot = WorldState(
            agent_pos=centre,
            agent_energy=jnp.full((wcfg.n_food_types,), 0.1),
            hotspot_pos=state.hotspot_pos,
            step=state.step,
            rng_key=state.rng_key,
        )
        s2 = step_world(at_hotspot, jnp.zeros(2), wcfg)
        assert float(s2.agent_energy[0]) > 0.1, "Agent at type-A hotspot must gain type-A energy"

    def test_type_separation_eating(self, wcfg2):
        """Eating type-A food only replenishes type-A energy."""
        state = reset_world(jax.random.PRNGKey(0), wcfg2)
        centre_a = state.hotspot_pos[0, 0]
        # Put type-B hotspots far away
        hotspots = state.hotspot_pos.at[1].set(
            jnp.full((wcfg2.n_food, 2), wcfg2.arena_size)
        )
        at_a = WorldState(
            agent_pos=centre_a,
            agent_energy=jnp.array([0.1, 0.5]),
            hotspot_pos=hotspots,
            step=state.step,
            rng_key=state.rng_key,
        )
        s2 = step_world(at_a, jnp.zeros(2), wcfg2)
        # Type-A energy should increase (eating), type-B should decrease (metabolism only)
        assert float(s2.agent_energy[0]) > 0.1, "Type-A energy should increase at type-A hotspot"
        assert float(s2.agent_energy[1]) < 0.5, "Type-B energy should decrease (no type-B food nearby)"

    def test_agent_dies_if_any_energy_zero(self, wcfg2):
        """With 2 food types, agent is dead if either energy hits 0."""
        state = reset_world(jax.random.PRNGKey(0), wcfg2)
        # One energy type at 0, other healthy
        near_dead = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.array([0.5, 0.001]),  # type-B about to die
            hotspot_pos=jnp.full((wcfg2.n_food_types, wcfg2.n_food, 2), wcfg2.arena_size),
            step=state.step,
            rng_key=state.rng_key,
        )
        s2 = step_world(near_dead, jnp.zeros(2), wcfg2)
        alive = jnp.all(s2.agent_energy > 0.0)
        assert not bool(alive), "Agent should be dead when type-B energy reaches 0"

    def test_energy_capped_at_max(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        full_energy = WorldState(
            agent_pos=state.hotspot_pos[0, 0],
            agent_energy=jnp.full((wcfg.n_food_types,), wcfg.max_energy),
            hotspot_pos=state.hotspot_pos,
            step=state.step,
            rng_key=state.rng_key,
        )
        for _ in range(10):
            full_energy = step_world(full_energy, jnp.zeros(2), wcfg)
        assert jnp.all(full_energy.agent_energy <= wcfg.max_energy + 1e-5)

    def test_energy_never_negative(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        empty = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.zeros((wcfg.n_food_types,)),
            hotspot_pos=jnp.full((wcfg.n_food_types, wcfg.n_food, 2), wcfg.arena_size),
            step=state.step,
            rng_key=state.rng_key,
        )
        s2 = step_world(empty, jnp.zeros(2), wcfg)
        assert jnp.all(s2.agent_energy >= 0.0)

    def test_movement_costs_energy(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        barren = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.full((wcfg.n_food_types,), 0.5),
            hotspot_pos=jnp.full((wcfg.n_food_types, wcfg.n_food, 2), wcfg.arena_size),
            step=state.step,
            rng_key=state.rng_key,
        )
        stationary = step_world(barren, jnp.zeros(2), wcfg)
        moving     = step_world(barren, jnp.ones(2),  wcfg)
        assert jnp.all(moving.agent_energy < stationary.agent_energy), \
            "Moving agent should spend more energy than stationary agent"

    def test_boundary_reflection(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        at_edge = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.full((wcfg.n_food_types,), 0.5),
            hotspot_pos=state.hotspot_pos,
            step=state.step,
            rng_key=state.rng_key,
        )
        for _ in range(10):
            at_edge = step_world(at_edge, jnp.array([-1.0, -1.0]), wcfg)
        assert jnp.all(at_edge.agent_pos >= 0.0), "Agent escaped arena lower bound"
        assert jnp.all(at_edge.agent_pos <= wcfg.arena_size), "Agent escaped arena upper bound"

    def test_deterministic_given_same_key(self, state, wcfg):
        action = jnp.array([0.3, -0.5])
        s2a = step_world(state, action, wcfg)
        s2b = step_world(state, action, wcfg)
        assert jnp.array_equal(s2a.agent_pos,    s2b.agent_pos)
        assert jnp.array_equal(s2a.agent_energy, s2b.agent_energy)
        assert jnp.array_equal(s2a.hotspot_pos,  s2b.hotspot_pos)

    def test_hotspot_drift_over_time(self, wcfg):
        T = 500
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        initial_pos = state.hotspot_pos.copy()
        for _ in range(T):
            state = step_world(state, jnp.zeros(2), wcfg)
        displacement = state.hotspot_pos - initial_pos
        msd = float(jnp.mean(displacement ** 2))
        expected_msd = T * wcfg.hotspot_drift ** 2
        assert msd > 0.0, "Hotspots did not drift at all"
        assert msd < expected_msd * 3, f"Hotspots drifted far more than expected (msd={msd:.3f})"


# ── Energy economics unit check ───────────────────────────────────────────────

class TestEnergyEconomics:
    def test_metabolism_rate_matches_config(self, wcfg):
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        barren = WorldState(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_energy=jnp.full((wcfg.n_food_types,), 0.5),
            hotspot_pos=jnp.full((wcfg.n_food_types, wcfg.n_food, 2), wcfg.arena_size * 10),
            step=state.step,
            rng_key=state.rng_key,
        )
        s2 = step_world(barren, jnp.zeros(2), wcfg)
        expected = 0.5 - wcfg.metabolism
        assert float(s2.agent_energy[0]) == pytest.approx(expected, abs=1e-4)

    def test_lower_metabolism_extends_starvation_budget(self):
        """
        Halving metabolism should roughly double the starvation budget.
        With metabolism=0.005, a stationary agent at init_energy=0.5 with
        no food nearby should survive at least 80 steps (vs ~38 at 0.01).
        """
        wcfg = WorldConfig(metabolism=0.005)
        state = reset_world(jax.random.PRNGKey(0), wcfg)
        barren = WorldState(
            agent_pos=jnp.array([50.0, 50.0]),
            agent_energy=jnp.full((wcfg.n_food_types,), 0.5),
            hotspot_pos=jnp.full((wcfg.n_food_types, wcfg.n_food, 2), wcfg.arena_size * 10),
            step=state.step,
            rng_key=state.rng_key,
        )
        steps = 0
        for _ in range(120):
            state = step_world(barren, jnp.zeros(2), wcfg)
            barren = state
            if float(state.agent_energy[0]) <= 0.0:
                break
            steps += 1
        assert steps >= 80, (
            f"With metabolism=0.005 agent should survive ≥80 steps, got {steps}"
        )

    def test_metabolism_drain_scales_with_n_food_types(self):
        """
        Each energy pool drains independently at the SAME shared rate
        regardless of n_food_types, and survival requires every pool to stay
        above zero -- so raising n_food_types multiplies difficulty instead
        of just adding independent sub-tasks.  The per-type drain should
        instead be divided by n_food_types, holding the total metabolic
        budget roughly constant as T grows: half the per-type drain at T=2,
        a third at T=3.
        """
        def drain_per_type(n_food_types: int) -> float:
            wcfg = WorldConfig(n_food_types=n_food_types)
            state = reset_world(jax.random.PRNGKey(0), wcfg)
            barren = WorldState(
                agent_pos=jnp.array([0.0, 0.0]),
                agent_energy=jnp.full((wcfg.n_food_types,), 0.5),
                hotspot_pos=jnp.full(
                    (wcfg.n_food_types, wcfg.n_food, 2), wcfg.arena_size * 10),
                step=state.step,
                rng_key=state.rng_key,
            )
            s2 = step_world(barren, jnp.zeros(2), wcfg)
            return 0.5 - float(s2.agent_energy[0])

        d1 = drain_per_type(1)
        d2 = drain_per_type(2)
        d3 = drain_per_type(3)

        assert d2 == pytest.approx(d1 / 2, rel=1e-4), (
            f"2 food types should halve per-type drain: d1={d1}, d2={d2}"
        )
        assert d3 == pytest.approx(d1 / 3, rel=1e-4), (
            f"3 food types should divide per-type drain by 3: d1={d1}, d3={d3}"
        )

    def test_movement_cost_drain_scales_with_n_food_types(self):
        """The move_cost component of drain must also divide by
        n_food_types, not just metabolism -- it's a shared search cost, not
        attributable to any one food type."""
        def moving_drain_per_type(n_food_types: int) -> float:
            wcfg = WorldConfig(n_food_types=n_food_types)
            state = reset_world(jax.random.PRNGKey(0), wcfg)
            barren = WorldState(
                agent_pos=jnp.array([50.0, 50.0]),
                agent_energy=jnp.full((wcfg.n_food_types,), 0.5),
                hotspot_pos=jnp.full(
                    (wcfg.n_food_types, wcfg.n_food, 2), wcfg.arena_size * 10),
                step=state.step,
                rng_key=state.rng_key,
            )
            s2 = step_world(barren, jnp.ones(2), wcfg)
            return 0.5 - float(s2.agent_energy[0])

        d1 = moving_drain_per_type(1)
        d2 = moving_drain_per_type(2)

        assert d2 == pytest.approx(d1 / 2, rel=1e-4), (
            f"2 food types should halve total (metabolism + move_cost) "
            f"per-type drain: d1={d1}, d2={d2}"
        )

    def test_smaller_hotspot_sigma_reduces_coverage(self):
        """
        A smaller sigma should give significantly less food reward at the
        same distance from the hotspot centre.
        At distance = 2*sigma the reward should drop by a factor of e^2 ≈ 7.4x.
        """
        dist = 6.0  # fixed distance from hotspot
        wcfg_large = WorldConfig(hotspot_sigma=5.0)
        wcfg_small = WorldConfig(hotspot_sigma=3.0)
        hotspot = jnp.array([[50.0, 50.0], [50.0, 50.0], [50.0, 50.0]])
        pos_near = jnp.array([50.0 + dist, 50.0])
        f_large = float(food_at(pos_near, hotspot, wcfg_large))
        f_small = float(food_at(pos_near, hotspot, wcfg_small))
        assert f_large > f_small, "Larger sigma should give more reward at same distance"
        ratio = f_large / (f_small + 1e-9)
        assert ratio > 3.0, (
            f"Smaller sigma should substantially reduce reward at distance={dist}: "
            f"ratio={ratio:.2f} (expected >3x)"
        )


# ── Controllers ───────────────────────────────────────────────────────────────

class TestControllers:
    def test_random_walk_action_shape(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        action  = random_walk(jax.random.PRNGKey(0), sensors, state, wcfg)
        assert action.shape == (2,)

    def test_random_walk_action_in_range(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        for i in range(10):
            action = random_walk(jax.random.PRNGKey(i), sensors, state, wcfg)
            assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)

    def test_nearest_hotspot_action_shape(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        action  = nearest_hotspot(jax.random.PRNGKey(0), sensors, state, wcfg)
        assert action.shape == (2,)

    def test_nearest_hotspot_action_in_range(self, state, wcfg):
        sensors = sensor_readout(state, wcfg)
        action  = nearest_hotspot(jax.random.PRNGKey(0), sensors, state, wcfg)
        assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)

    def test_nearest_hotspot_moves_toward_food(self, wcfg):
        wcfg_single = WorldConfig(n_food_types=1, n_food=1, hotspot_drift=0.0)
        state = reset_world(jax.random.PRNGKey(0), wcfg_single)
        hotspot = state.hotspot_pos[0, 0]  # type 0, hotspot 0
        offset_pos = jnp.clip(hotspot + jnp.array([-20.0, 0.0]), 0.0, wcfg_single.arena_size)
        at_offset = WorldState(
            agent_pos=offset_pos,
            agent_energy=state.agent_energy,
            hotspot_pos=state.hotspot_pos,
            step=state.step,
            rng_key=state.rng_key,
        )
        sensors = sensor_readout(at_offset, wcfg_single)
        action  = nearest_hotspot(jax.random.PRNGKey(0), sensors, at_offset, wcfg_single)
        assert float(action[0]) > 0.0, "Gradient follower should move toward hotspot"

    def test_nearest_hotspot_two_types(self, wcfg2):
        """nearest_hotspot works with multi-type state."""
        state = reset_world(jax.random.PRNGKey(0), wcfg2)
        sensors = sensor_readout(state, wcfg2)
        action = nearest_hotspot(jax.random.PRNGKey(0), sensors, state, wcfg2)
        assert action.shape == (2,)
        assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)


# ── Validation gates ──────────────────────────────────────────────────────────

class TestValidationGates:
    def test_nearest_hotspot_survives_full_episode(self):
        wcfg = WorldConfig(episode_steps=500)
        _, steps = run_episode(jax.random.PRNGKey(0), nearest_hotspot, wcfg)
        assert steps == wcfg.episode_steps, (
            f"Gradient follower starved at step {steps}/{wcfg.episode_steps}"
        )

    def test_random_walk_dies_before_episode_end(self):
        wcfg  = WorldConfig(episode_steps=2000)
        seeds = [10, 11, 12, 13, 14]
        steps_list = [
            run_episode(jax.random.PRNGKey(s), random_walk, wcfg)[1]
            for s in seeds
        ]
        max_steps = max(steps_list)
        assert max_steps < wcfg.episode_steps, (
            f"Random walker survived the full episode ({max_steps} steps). "
            "World may be too easy."
        )

    def test_difficulty_band(self):
        wcfg  = WorldConfig(episode_steps=1000)
        seeds = [20, 21, 22]
        gf_steps = [run_episode(jax.random.PRNGKey(s), nearest_hotspot, wcfg)[1] for s in seeds]
        rw_steps = [run_episode(jax.random.PRNGKey(s), random_walk,        wcfg)[1] for s in seeds]
        mean_gf  = np.mean(gf_steps)
        mean_rw  = np.mean(rw_steps)
        assert mean_gf > mean_rw * 2, (
            f"Difficulty band too narrow: nearest_hotspot mean={mean_gf:.0f}, "
            f"random_walk mean={mean_rw:.0f}.  Gap should be at least 2x."
        )

    def test_two_type_world_nearest_hotspot_survives(self):
        """
        Urgency×proximity controller must navigate the 2-food-type world well.

        Strict strip separation makes inter-type travel necessary, so 100%
        survival on every seed is not guaranteed.  We verify instead that the
        mean survival fraction over multiple seeds is clearly above random
        (random walk dies in <<50% of steps) and above 85%.
        """
        wcfg  = WorldConfig(n_food_types=2, episode_steps=500)
        seeds = [0, 2, 4, 6, 8]
        steps_list = [
            int(run_episode(jax.random.PRNGKey(s), nearest_hotspot, wcfg)[1])
            for s in seeds
        ]
        mean_frac = np.mean(steps_list) / wcfg.episode_steps
        assert mean_frac >= 0.85, (
            f"Gradient follower mean survival {mean_frac:.2f} < 0.85 "
            f"in two-food-type world: {steps_list}"
        )

    def test_hotspot_strips_no_cross_type_overlap(self):
        """
        At init, type-i hotspots must lie in the Y-axis strip
        [i/T, (i+1)/T] * arena_size.  Checked across multiple seeds.
        For n_food_types=1 the strip is the full arena — trivially satisfied.
        """
        for n_types in (2, 3, 4):
            wcfg = WorldConfig(n_food_types=n_types, arena_size=100.0)
            strip_h = wcfg.arena_size / n_types
            for seed in range(10):
                state = reset_world(jax.random.PRNGKey(seed), wcfg)
                hs = np.array(state.hotspot_pos)   # [n_types, n_food, 2]
                for ti in range(n_types):
                    lo = ti * strip_h
                    hi = (ti + 1) * strip_h
                    ys = hs[ti, :, 1]  # Y coordinates for this type
                    assert np.all(ys >= lo - 1e-4) and np.all(ys <= hi + 1e-4), (
                        f"n_types={n_types}, type {ti}: Y={ys} not in [{lo:.1f}, {hi:.1f}]"
                    )

    def test_hotspot_strips_single_type_full_arena(self):
        """With n_food_types=1 the strip is the whole arena — no constraint on Y."""
        wcfg  = WorldConfig(n_food_types=1, arena_size=100.0)
        states = [reset_world(jax.random.PRNGKey(s), wcfg) for s in range(20)]
        ys = np.array([np.array(s.hotspot_pos)[0, :, 1] for s in states]).ravel()
        # Should span at least half the arena height across seeds
        assert float(ys.max() - ys.min()) > 40.0, "Single-type strips too restricted"
