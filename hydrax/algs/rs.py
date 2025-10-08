from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
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
    alpha = 0.5

class RS(SamplingBasedController):
    """
      Randomized Smoothing control.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
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
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
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
        self.temperature = temperature

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
        controls = jnp.concatenate([controls, params.mean[None]], axis=0)
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: RSParams, rollouts: Trajectory
    ) -> RSParams:
        """Update the mean with the estimated gradient through randomized smoothing"""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        costs_diff = costs[:-1] - costs[-1]
        noise_knots = (rollouts.knots - params.mean[None])[:-1]
        estim_grad = jnp.mean((costs_diff[:, None, None] * noise_knots), axis=0) / self.noise_level
        mean = params.mean - params.alpha * estim_grad
        return params.replace(mean=mean)
