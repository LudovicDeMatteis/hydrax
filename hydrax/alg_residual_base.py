from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import AverageCost, RiskStrategy
from hydrax.task_base import Task
from hydrax.utils.spline import get_interp_func

@dataclass
class TrajectoryResiduals:
    """Data class for storing rollout data.

    Throughout, H denotes the number of control steps (given by the times at
    which the control spline is interpolated).

    Attributes:
        controls: Control actions of shape (num_rollouts, H, nu).
        knots: Control spline knots of shape (num_rollouts, num_knots, nu).
        residuals: Residuals of shape (num_rollouts, H+1, r_dim).
        trace_sites: Positions of trace sites of shape (num_rollouts, H+1, 3).
    """
    
    knots: jax.Array
    controls: jax.Array
    trace_sites: jax.Array
    residuals: jax.Array

    def __len__(self):
        """Return the number of time steps in the trajectory"""
        return self.controls.shape[1]


class SamplingBasedResidualController(SamplingBasedController):
    """An abstract class based on SamplingBasedController for using residuals instead of costs."""
    
    def __init__(
        self,
        task: Task,
        num_randomizations: int,
        risk_strategy: RiskStrategy,
        seed: int,
        plan_horizon: float,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and residuals.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining residuals from different randomizations.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        self.task = task
        self.num_randomizations = max(num_randomizations, 1)

        # Risk strategy defaults to average cost
        if risk_strategy is None:
            risk_strategy = AverageCost()
        self.risk_strategy = risk_strategy

        # time-related variables
        # NOTE: we always interpret self.task.model as the controller's
        # internal model, not the model used for simulation. dt is the
        # time between spline queries.
        self.plan_horizon = plan_horizon
        self.dt = self.task.dt
        self.ctrl_steps = int(round(self.plan_horizon / self.dt))

        # Spline setup for control interpolation
        self.spline_type = spline_type
        self.num_knots = num_knots
        self.interp_func = get_interp_func(spline_type)

        # Use a single model (no domain randomization) by default
        self.model = task.model
        self.randomized_axes = None

        # Number of optimization iterations
        if iterations < 1:
            raise ValueError("iterations must be greater than 0!")

        self.iterations = iterations

        if self.num_randomizations > 1:
            # Make domain randomized models
            rng = jax.random.key(seed)
            rng, subrng = jax.random.split(rng)
            subrngs = jax.random.split(subrng, num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_model)(subrngs)
            self.model = self.task.model.tree_replace(randomizations)

            # Keep track of which elements of the model have randomization
            self.randomized_axes = jax.tree.map(lambda x: None, self.task.model)
            self.randomized_axes = self.randomized_axes.tree_replace(
                {key: 0 for key in randomizations.keys()}
            )

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Warm-start spline by advancing knot times by sim dt, then recomputing
        # the mean knots by evaluating the old spline at those times
        tk = params.tk
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)

        def _optimize_scan_body(params: Any, _: Any):
            # Sample random control sequences from spline knots
            knots, params = self.sample_knots(params)
            knots = jnp.clip(
                knots, self.task.u_min, self.task.u_max
            )  # (num_rollouts, num_knots, nu)

            # Roll out the control sequences, applying domain randomizations and
            # combining costs using self.risk_strategy.
            rng, dr_rng = jax.random.split(params.rng)
            rollouts = self.rollout_with_randomizations(
                state, new_tk, knots, dr_rng
            )
            params = params.replace(rng=rng)

            # Update the policy parameters based on the combined costs
            params = self.update_params(state, params, rollouts)

            return params, rollouts

        params, rollouts = jax.lax.scan(
            f=_optimize_scan_body, init=params, xs=jnp.arange(self.iterations)
        )

        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)

        return params, rollouts_final

    def rollout_with_randomizations(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        rng: jax.Array,
    ) -> Trajectory:
        """Compute rollout residuals, applying domain randomizations.

        Args:
            state: The initial state x₀.
            tk: The knot times of the control spline, (num_knots,).
            knots: The control spline knots, (num rollouts, num_knots, nu).
            rng: The random number generator key for randomizing initial states.

        Returns:
            A Trajectory object containing the control, residuals, and trace sites.
            Costs are aggregated over domains using the given risk strategy.
        """
        # Set the initial state for each rollout.
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )

        if self.num_randomizations > 1:
            # Randomize the initial states for each domain randomization
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(
                states, subrngs
            )
            states = states.tree_replace(randomizations)

        # compute the control sequence from the knots
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, knots)  # (num_rollouts, H, nu)

        # Apply the control sequences, parallelized over both rollouts and
        # domain randomizations.
        _, rollouts = jax.vmap(
            self.eval_rollouts, in_axes=(self.randomized_axes, 0, None, None)
        )(self.model, states, controls, knots)

        # Combine the residuals from different domain randomizations using the
        # specified risk strategy.
        residuals = self.risk_strategy.combine_residuals(rollouts.residuals)
        controls = rollouts.controls[0]  # identical over randomizations
        knots = rollouts.knots[0]  # identical over randomizations
        trace_sites = rollouts.trace_sites[0]  # visualization only, take 1st
        return rollouts.replace(
            residuals=residuals, controls=controls, knots=knots, trace_sites=trace_sites
        )

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
    ) -> Tuple[mjx.Data, Trajectory]:
        """Rollout control sequences (in parallel) and compute the residuals.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, (num rollouts, H, nu).
            knots: The control spline knots, (num rollouts, num_knots, nu).

        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, residuals, and trace sites.
        """

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the residuals and observation, then advance the state."""
            x = x.replace(ctrl=u)
            x = mjx.step(model, x)  # step model + compute site positions
            residuals = self.dt * self.task.running_residuals(x, u)
            sites = self.task.get_trace_sites(x)
            return x, (x, residuals, sites)

        final_state, (states, residuals, trace_sites) = jax.lax.scan(
            _scan_fn, state, controls
        )
        final_residuals = self.task.terminal_residuals(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)

        residuals = jnp.append(residuals, final_residuals)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return states, TrajectoryResiduals(
            controls=controls,
            knots=knots,
            residuals=residuals,
            trace_sites=trace_sites,
        )

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> Any:
        """Initialize the policy parameters, U = [u₀, u₁, ... ] ~ π(params).

        Args:
            initial_knots: The initial knots of the control spline.
            seed: The random seed for initializing the policy parameters.

        Returns:
            The initial policy parameters.
        """
        rng = jax.random.key(seed)
        mean = (
            initial_knots
            if initial_knots is not None
            else jnp.zeros((self.num_knots, self.task.model.nu))
        )
        assert mean.shape == (self.num_knots, self.task.model.nu), (
            f"Initial knots must have shape (num_knots, nu), got {mean.shape}"
        )
        tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots)
        return SamplingParams(tk=tk, mean=mean, rng=rng)
