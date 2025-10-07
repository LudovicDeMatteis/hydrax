from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class ArmReaching(Task):
    """Reaching task for the Franka-Emika panda arm."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/panda/mjx_scene.xml"
        )
        super().__init__(
            mj_model,
        )

        # Get end_effector frame id
        self.ee_id = mj_model.body("link7").id

        # Set the target position
        self.target_pos = jnp.array([0.4, 0.4, 0.6])

        # Initial configuration
        self.qinit = jnp.array([0.0, -0.785, 0.0, -2.35, 0.0, 1.57, 0.785])

    def _get_ee_pos(self, state: mjx.Data) -> jax.Array:
        """Get the position of the end effector."""
        return state.xpos[self.ee_id - 1]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost l(x_t, u_t)."""
        target_error = jnp.sum(
            jnp.square(self._get_ee_pos(state) - self.target_pos)
        )
        config_reg = jnp.sum(jnp.square(state.qpos - self.qinit))
        control_reg = jnp.sum(jnp.square(control))
        return 10.0 * target_error + 0.1 * config_reg + 0.001 * control_reg

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost phi(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly perturb the measured base position and velocities."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.01 * jax.random.normal(q_rng, (7,))
        v_err = 0.01 * jax.random.normal(v_rng, (6,))

        qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
        qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

        return {"qpos": qpos, "qvel": qvel}
