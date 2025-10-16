from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.alg_base import SamplingParams
from hydrax.alg_residual_base import SamplingBasedResidualController, TrajectoryResiduals
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class RSParams(SamplingParams):
    """Policy parameters for randomized smoothing control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """

class RSGaussNewton(SamplingBasedResidualController):
    """
      Randomized Smoothing control using residuals instead of cost (to use Gauss Newton)
    """
    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            alpha: The step size for the descent.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.noise_level = noise_level
        self.num_samples = num_samples

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> RSParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        return RSParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    def sample_knots(self, params: RSParams) -> Tuple[jax.Array, RSParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_level * noise
        controls = jnp.concatenate([params.mean[None], controls], axis=0)
        return controls, params.replace(rng=rng)

    def update_params(
        self, state:mjx.Data, params: RSParams, rollouts: TrajectoryResiduals
    ) -> RSParams:
        """Update the mean with the estimated gradient through randomized smoothing"""
        def _linesearch(params, direction):
            rng, dr_rng = jax.random.split(params.rng)
            alphas = jnp.array([2**i for i in range(3, -10, -1)])
            candidates = params.mean + alphas[:, None, None] * direction
            rollouts = self.rollout_with_randomizations(
                state, params.tk, candidates, dr_rng
            )
            params = params.replace(rng=rng)
            residuals = jnp.reshape(rollouts.residuals, (candidates.shape[0], -1))
            costs = jnp.sum(residuals**2, axis=1)
            best = jnp.argmin(costs)
            alpha = alphas[best]
            return alpha

        residuals = jnp.reshape(rollouts.residuals, (self.num_samples + 1, -1))
        res_diff = residuals[1:] - residuals[0]
        noise_knots = (rollouts.knots[1:] - params.mean[None])
        estim_res_grad = jnp.einsum("ni,njk->ikj", res_diff, noise_knots) / (self.noise_level * self.num_samples)
        estim_res_grad = jnp.reshape(estim_res_grad, (estim_res_grad.shape[0], -1))
        approx_hessian = (estim_res_grad.T @ estim_res_grad) + 1e-6 * jnp.eye(estim_res_grad.shape[1])
        direction = -jnp.linalg.inv(approx_hessian) @ estim_res_grad.T @ residuals[0]
        direction = jnp.reshape(direction, params.mean.shape)
        a = _linesearch(params, direction)
        mean = params.mean + a * direction
        mean = jnp.clip(
            mean, self.task.u_min, self.task.u_max
        )
        return params.replace(mean=mean)
