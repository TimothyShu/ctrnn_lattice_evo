from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp


# ── WorldConfig ───────────────────────────────────────────────────────────────

@dataclass
class WorldConfig:
    # Arena
    arena_size:    float = 100.0

    # Food
    n_food_types:  int   = 1     # distinct food types; each has n_food hotspots
    n_food:        int   = 3     # hotspots per type
    hotspot_sigma: float = 5.0   # Gaussian spread (units)
    hotspot_drift: float = 0.6   # std of per-step random walk of hotspot centres

    # Energy economics
    init_energy:   float = 0.5
    metabolism:    float = 0.010  # passive drain per world step (all energy types equally)
    move_cost:     float = 0.003  # additional drain per unit of speed (all energy types)
    eat_rate:      float = 0.08   # energy gain = eat_rate * clipped_food_density
    max_energy:    float = 1.0

    # Agent physics
    max_speed:     float = 3.0   # units per world step

    # Episode
    episode_steps: int   = 2000

    # Sensors — whether to append normalised (x, y) position to the sensor vector.
    # Must match Config.position_sensors so that n_in is consistent.
    # Default False for backward compatibility with all existing runs.
    position_sensors: bool = False


# ── WorldState ────────────────────────────────────────────────────────────────

@dataclass
class WorldState:
    """
    All mutable world state for one episode step.

    Registered as a JAX pytree so jit / lax.scan can look inside.

    agent_energy : [n_food_types]            one energy resource per food type
    hotspot_pos  : [n_food_types, n_food, 2] hotspot centres per type
    """
    agent_pos:    jnp.ndarray  # [2]                       float32 — position in [0, arena_size]²
    agent_energy: jnp.ndarray  # [n_food_types]            float32 — each in [0, max_energy]
    hotspot_pos:  jnp.ndarray  # [n_food_types, n_food, 2] float32 — hotspot centres
    step:         jnp.ndarray  # []                        int32
    rng_key:      jax.Array    # PRNG key for stochastic drift


jax.tree_util.register_pytree_node(
    WorldState,
    lambda s: (
        [s.agent_pos, s.agent_energy, s.hotspot_pos, s.step, s.rng_key],
        None,
    ),
    lambda _, children: WorldState(*children),
)


# ── Food field ────────────────────────────────────────────────────────────────

def food_at(
    pos: jnp.ndarray,
    hotspot_pos: jnp.ndarray,
    wcfg: WorldConfig,
) -> jnp.ndarray:
    """
    Raw food density at pos from one type's hotspots: sum of Gaussians.

    hotspot_pos : [n_food, 2] — centres for a single food type.
    Returns values in [0, n_food].  Clip to [0, 1] for sensor / energy use.
    """
    diff    = pos[None, :] - hotspot_pos                              # [n_food, 2]
    sq_dist = jnp.sum(diff ** 2, axis=-1)                             # [n_food]
    return jnp.sum(jnp.exp(-sq_dist / (2.0 * wcfg.hotspot_sigma ** 2)))


# ── Sensor readout ────────────────────────────────────────────────────────────

def sensor_readout(state: WorldState, wcfg: WorldConfig) -> jnp.ndarray:
    """
    Returns the sensor vector fed to the CTRNN input neurons.

    Base (always present):
        [food_0, ..., food_{T-1}, energy_0, ..., energy_{T-1}]
        length = 2 * n_food_types, all normalised to [0, 1].

    With wcfg.position_sensors=True, two additional values are appended:
        [..., x / arena_size, y / arena_size]
        length = 2 * n_food_types + 2

    With n_food_types=1 and position_sensors=False this is [food_density, energy_level],
    identical to the original two-sensor interface.

    Position sensors give the agent proprioceptive awareness of its location,
    which is essential when it starts far from any hotspot and the food signal
    is zero — without them it has no gradient to follow and runs open-loop.
    """
    food_norms = jax.vmap(
        lambda hpos: jnp.clip(food_at(state.agent_pos, hpos, wcfg), 0.0, 1.0)
    )(state.hotspot_pos)                                    # [n_food_types]
    energy_norms = state.agent_energy / wcfg.max_energy    # [n_food_types]
    sensors = jnp.concatenate([food_norms, energy_norms])  # [2 * n_food_types]

    if wcfg.position_sensors:
        pos_norm = state.agent_pos / wcfg.arena_size        # [2] in [0, 1]
        sensors  = jnp.concatenate([sensors, pos_norm])

    return sensors


# ── World step ────────────────────────────────────────────────────────────────

def step_world(
    state: WorldState,
    action: jnp.ndarray,
    wcfg: WorldConfig,
) -> WorldState:
    """
    Advance the world by one step.

    action: [v_x, v_y] in [-1, 1]; scaled by max_speed internally.
    Boundary: reflective clamp (agent cannot leave the arena).

    Each energy type is replenished only by its own hotspots.
    Metabolism and movement cost drain all energy types equally.
    The agent is alive only while ALL energy types are > 0.
    """
    # --- Physics ---
    v       = jnp.clip(action, -1.0, 1.0) * wcfg.max_speed
    speed   = jnp.sqrt(jnp.sum(v ** 2))
    new_pos = jnp.clip(state.agent_pos + v, 0.0, wcfg.arena_size)

    # --- Energy (per type via vmap) ---
    lost = wcfg.metabolism + wcfg.move_cost * speed  # shared drain applied to each type

    def update_energy(energy_i: jnp.ndarray, hotspot_pos_i: jnp.ndarray) -> jnp.ndarray:
        food_norm = jnp.clip(food_at(new_pos, hotspot_pos_i, wcfg), 0.0, 1.0)
        gained    = wcfg.eat_rate * food_norm
        return jnp.clip(energy_i + gained - lost, 0.0, wcfg.max_energy)

    new_energy = jax.vmap(update_energy)(state.agent_energy, state.hotspot_pos)

    # --- Hotspot drift (all types) ---
    key, subkey  = jax.random.split(state.rng_key)
    noise        = jax.random.normal(subkey, state.hotspot_pos.shape) * wcfg.hotspot_drift
    new_hotspots = jnp.clip(state.hotspot_pos + noise, 0.0, wcfg.arena_size)

    return WorldState(
        agent_pos=new_pos,
        agent_energy=new_energy,
        hotspot_pos=new_hotspots,
        step=state.step + 1,
        rng_key=key,
    )


# ── Episode initialisation ────────────────────────────────────────────────────

def reset_world(key: jax.Array, wcfg: WorldConfig) -> WorldState:
    """Initialise a fresh episode with random agent position and hotspot layout.

    Hotspots are placed in non-overlapping Y-axis strips: food type i is
    confined to [i/T, (i+1)/T] * arena_size at episode start.  X coordinates
    are drawn from the full arena width.  This guarantees cross-type spatial
    separation, creating genuine modular task structure.  With n_food_types=1
    the strip spans the full arena and behaviour is identical to uniform sampling.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Raw uniform samples in [0, 1]^3: [n_food_types, n_food, 2]
    raw = jax.random.uniform(k2, (wcfg.n_food_types, wcfg.n_food, 2))

    # Y-strip boundaries per type: shape [n_food_types, 1, 1]
    T       = wcfg.n_food_types
    strip_h = wcfg.arena_size / T
    lo_y    = (jnp.arange(T) * strip_h)[:, None, None]   # [T, 1, 1]

    xs = raw[..., 0:1] * wcfg.arena_size          # full width
    ys = lo_y + raw[..., 1:2] * strip_h           # clamped to type's strip

    hotspot_pos = jnp.concatenate([xs, ys], axis=-1)  # [T, n_food, 2]

    return WorldState(
        agent_pos=jax.random.uniform(k1, (2,)) * wcfg.arena_size,
        agent_energy=jnp.full((wcfg.n_food_types,), wcfg.init_energy, dtype=jnp.float32),
        hotspot_pos=hotspot_pos,
        step=jnp.array(0, dtype=jnp.int32),
        rng_key=k3,
    )


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    key: jax.Array,
    controller_fn,
    wcfg: WorldConfig,
) -> tuple[WorldState, jnp.ndarray]:
    """
    Run a full episode with controller_fn for wcfg.episode_steps steps.

    controller_fn(key, sensors, state, wcfg) -> action [2] in [-1, 1]

    Returns (final_state, steps_survived) where steps_survived counts the
    number of steps on which ALL energy types were > 0 after the step.
    Surviving the full episode gives steps_survived == episode_steps.
    """
    k1, k2 = jax.random.split(key)
    state   = reset_world(k1, wcfg)

    def body(carry, _):
        state, k = carry
        k, ctrl_key = jax.random.split(k)
        sensors   = sensor_readout(state, wcfg)
        action    = controller_fn(ctrl_key, sensors, state, wcfg)
        new_state = step_world(state, action, wcfg)
        alive     = jnp.all(new_state.agent_energy > 0.0)
        return (new_state, k), alive

    (final_state, _), alive_mask = jax.lax.scan(
        body, (state, k2), None, length=wcfg.episode_steps
    )
    steps_survived = jnp.sum(alive_mask.astype(jnp.int32))
    return final_state, steps_survived
