import time
from typing import Sequence
import os

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController
from hydrax import ROOT
from hydrax.utils.video import VideoRecorder

"""
Tools solving the OCP, without the simulator running
"""

def ocp(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    initial_knots: jax.Array = None,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
) -> None:
    """Run an controller to solve the OCP.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        mj_data: A MuJoCo data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        initial_knots: The initial knot points for the control spline at t=0
        trace_width: The width of the trace lines (in pixels).
        trace_color: The RGBA color of the trace lines.
        reference: The reference trajectory (qs) to visualize.
        reference_fps: The frame rate of the reference trajectory.
    """
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(initial_knots=initial_knots)

    # Set the start state for the controller
    mjx_data = mjx_data.replace(
        qpos=jnp.array(mj_data.qpos),
        qvel=jnp.array(mj_data.qvel),
        mocap_pos=jnp.array(mj_data.mocap_pos),
        mocap_quat=jnp.array(mj_data.mocap_quat),
        time=mj_data.time,
    )

    # Do a replanning step
    plan_start = time.time()
    policy_params, rollouts = controller.optimize(mjx_data, policy_params)
    plan_time = time.time() - plan_start

    print(f"Problem solved in {plan_time:.3f} seconds")
    # Preserve the last printout
    print("")

    return policy_params, rollouts
