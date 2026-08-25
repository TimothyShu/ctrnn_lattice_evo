from __future__ import annotations

import jax
import jax.numpy as jnp

from .world import WorldConfig, WorldState


def random_walk(
    key: jax.Array,
    sensors: jnp.ndarray,
    state: WorldState,
    wcfg: WorldConfig,
) -> jnp.ndarray:
    """
    Uniform random velocity each step.

    Baseline controller that should reliably starve — the expected energy
    gain from random movement is less than the metabolism + movement cost
    at the default WorldConfig parameters.
    """
    return jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0)


def nearest_hotspot(
    key: jax.Array,
    sensors: jnp.ndarray,
    state: WorldState,
    wcfg: WorldConfig,
) -> jnp.ndarray:
    """
    Move toward the hotspot with the best urgency × proximity score.

    Score for each food type = (1 / energy) × (1 / nearest_hotspot_distance).
    This causes the controller to proactively interleave visits between types —
    going to a nearby type even before it fully depletes — which prevents one
    energy channel from draining to zero while the agent is eating another.

    Validation tool only — reads exact hotspot coordinates from WorldState,
    which evolved agents cannot access.  Sensors are ignored entirely.

    With n_food_types=1 the score reduces to pure proximity, identical to the
    original nearest-hotspot behaviour.
    """
    def _score(energy_i: jnp.ndarray, hotspots_i: jnp.ndarray) -> jnp.ndarray:
        diff      = hotspots_i - state.agent_pos[None, :]          # [n_food, 2]
        distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))          # [n_food]
        min_dist  = jnp.min(distances)
        return (1.0 / (energy_i + 0.01)) * (1.0 / (min_dist + 1.0))

    scores     = jax.vmap(_score)(state.agent_energy, state.hotspot_pos)  # [n_types]
    best_type  = jnp.argmax(scores)
    hotspots_t = state.hotspot_pos[best_type]
    diff       = hotspots_t - state.agent_pos[None, :]
    distances  = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    nearest    = jnp.argmin(distances)
    direction  = diff[nearest] / (distances[nearest] + 1e-8)
    return direction
