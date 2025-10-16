import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class CartPole(Task):
    """A cart-pole swingup task."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/cart_pole/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["tip"])

    def _distance_to_upright(self, state: mjx.Data) -> jax.Array:
        """Get a measure of distance to the upright position."""
        theta = state.qpos[1] + jnp.pi
        theta_err = jnp.array([jnp.cos(theta) - 1, jnp.sin(theta)])
        return theta_err

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        res = self.running_residuals(state, control)
        cost = (1/2)*jnp.dot(res, res)
        return cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        res = self.terminal_residuals(state)
        cost = (1/2)*jnp.dot(res, res)
        return cost

    def running_residuals(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost residual r(xₜ, uₜ)."""
        theta_residual = 5 * self._distance_to_upright(state).flatten()
        centering_residual = jnp.array(state.qpos[0]).flatten()
        velocity_residual = 0.1 * state.qvel.flatten()
        control_residual = 0.1 * control.flatten()
        return jnp.concatenate([
            theta_residual,
            centering_residual,
            velocity_residual,
            control_residual,
        ])

    def terminal_residuals(self, state: mjx.Data) -> jax.Array:
        theta_residual = 5 * self._distance_to_upright(state).flatten()
        centering_residual = jnp.array(state.qpos[0]).flatten()
        velocity_residual = 0.1 * state.qvel.flatten()
        return jnp.concatenate([
            theta_residual,
            centering_residual,
            velocity_residual,
        ])
