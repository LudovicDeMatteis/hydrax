from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class H1Imitation(Task):
    """Reaching task for the Franka-Emika panda arm."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/h1_2/scene/scene_27dof.xml"
        )
        super().__init__(
            mj_model,
        )

        # Initial configuration
        self.qinit = jnp.array(mj_model.keyframe("home").qpos)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """
            The running cost l(x_t, u_t)
            The running cost is defined as l(x_t, u_t) = ||r||_2^2.
        """
        res = self.running_residuals(state, control)
        cost = (1/2)*jnp.dot(res, res)
        return cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost phi(x_T)."""
        res = self.terminal_residuals(state)
        cost = (1/2)*jnp.dot(res, res)
        return cost

    def running_residuals(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """
            The running residuals r(x_t, u_t)
            The running cost is defined as l(x_t, u_t) = ||r||_2^2.
        """
        config_reg = state.qpos - self.qinit
        control_reg = control
        residuals = jnp.concatenate(
            [0.1*config_reg, 0.03*control_reg]
        )
        return residuals

    def terminal_residuals(self, state: mjx.Data) -> jax.Array:
        """The terminal residual r_T(x_T)."""
        return self.running_residuals(state, jnp.zeros(self.model.nu))

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
