import time
import os
import tqdm
from typing import Sequence
from contextlib import nullcontext
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController
from hydrax import ROOT
from hydrax.utils.video import VideoRecorder
from hydrax.utils.log import plot_solver_metrics

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""


def run_interactive(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    stop_time: float = -1.0,
    headless: bool = False, 
    log_path: Path = "",
) -> None:
    """Run an interactive simulation with the MPC controller.

    This is a deterministic simulation, with the controller and simulation
    running in the same thread. This is useful for repeatability, but is less
    realistic than asynchronous simulation.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        mj_data: A MuJoCo data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        initial_knots: The initial knot points for the control spline at t=0
        fixed_camera_id: The camera ID to use for the fixed camera view.
        show_traces: Whether to show traces for the site positions.
        max_traces: The maximum number of traces to show at once.
        trace_width: The width of the trace lines (in pixels).
        trace_color: The RGBA color of the trace lines.
        reference: The reference trajectory (qs) to visualize.
        reference_fps: The frame rate of the reference trajectory.
        lstop_time: TODO
        headless: TODO
        log_path: TODO
    """
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts, metrics = jit_optimize(mjx_data, policy_params)
    policy_params, rollouts, metrics = jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)
    print(f"Time to jit: {time.time() - st:.3f} seconds")
    num_traces = min(rollouts.controls.shape[1], max_traces)

    # Initialize video recording if enabled
    recorder = None
    if log_path:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir=os.path.join(log_path, "videos"),
            width=width,
            height=height,
            fps=int(1/mj_model.opt.timestep),
        )
        # Ensure model visual offscreen buffer is compatible with video recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        recorder.start()
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
    
    if log_path:
        metrics_log = {"time": []}
        
        logger = {
            "time": [],
            "qpos": [],
            "qvel": [],
            "ctrl": [],
            "qacc": [],
            "tau": [],
        }

    # Start the simulation
    viewer_context = mujoco.viewer.launch_passive(mj_model, mj_data) if not headless else nullcontext()
    
    with viewer_context as viewer:
        if not headless:
            viewer.opt.sitegroup[5] = 1
            cam = viewer.cam
            opt = viewer.opt
        
            # Set up rollout traces
            if show_traces:
                num_trace_sites = len(controller.task.trace_site_ids)
                for i in range(
                    num_trace_sites * num_traces * controller.ctrl_steps
                ):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_LINE,
                        size=np.zeros(3),
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.array(trace_color),
                    )
                    viewer.user_scn.ngeom += 1
        else:
            cam = mujoco.MjvCamera()
            opt = mujoco.MjvOption()
            opt.sitegroup[5] = 1 
            show_traces = False
        
        cam.trackbodyid = 0
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.distance = 3.0
        cam.elevation = -20
        cam.azimuth = 135

        # Add geometry for the ghost reference
        if reference is not None:
            n_sites = len(reference[0])
            site_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f"marker{i+1}") for i in range(n_sites)]

        if hasattr(controller.task, "stop_time"):
            pbar = tqdm.tqdm(total=controller.task.stop_time)
        while True:
            if hasattr(controller.task, "stop_time"):
                pbar.update(step_dt)
            if not headless and not viewer.is_running():
                break
            
            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                time=mj_data.time,
            )

            # Do a replanning step
            policy_params, rollouts, metrics = jit_optimize(mjx_data, policy_params)
            
            if log_path:   
                # Extract scalar values from JAX arrays for logging
                metrics_log["time"].append(mj_data.time)
                
                for key, val_history in metrics.items():
                    if key not in metrics_log:
                        metrics_log[key] = []
                    # Extract last iteration value as float
                    metrics_log[key].append(float(val_history[-1]))

            # Visualize the rollouts
            if show_traces:
                ii = 0
                for k in range(num_trace_sites):
                    for i in range(num_traces):
                        for j in range(controller.ctrl_steps):
                            mujoco.mjv_connector(
                                viewer.user_scn.geoms[ii],
                                mujoco.mjtGeom.mjGEOM_LINE,
                                trace_width,
                                rollouts.trace_sites[i, j, k],
                                rollouts.trace_sites[i, j + 1, k],
                            )
                            ii += 1

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time

            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)

            # simulate the system between spline replanning steps
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[i])
                mujoco.mj_step(mj_model, mj_data)
                # Update the ghost reference
                if reference is not None:
                    t_ref = mj_data.time * reference_fps
                    i_ref = int(t_ref)
                    i_ref = min(i_ref, reference.shape[0] - 1)
                    ref_positions = reference[i_ref]

                    for i, site_id in enumerate(site_ids):
                        mj_data.site_xpos[site_id] = ref_positions[i]
                        
                if not headless:
                    viewer.sync()

                # Capture frame if recording
                if log_path and recorder.is_recording:
                    renderer.update_scene(mj_data, cam, opt)
                    frame = renderer.render()
                    recorder.add_frame(frame.tobytes())

                if log_path:
                    logger["time"].append(mj_data.time)
                    logger["qpos"].append(np.array(mj_data.qpos))
                    logger["qvel"].append(np.array(mj_data.qvel))
                    logger["ctrl"].append(np.array(mj_data.ctrl))
                    logger["qacc"].append(np.array(mj_data.qacc))
                    logger["tau"].append(np.array(mj_data.actuator_force))

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            if (stop_time > 0.0 and mj_data.time >= stop_time) or (stop_time < 0.0 and hasattr(controller.task, "stop_time") and mj_data.time >= controller.task.stop_time):
                break
        if hasattr(controller.task, "stop_time"):
            pbar.close()

    # Preserve the last printout
    print("")

    if log_path:
        if recorder is not None:
            recorder.stop()
    
        traj_output_dir = os.path.join(log_path, "trajectories") 
        if not os.path.exists(traj_output_dir):
            os.makedirs(traj_output_dir)
        np.savez_compressed(
            os.path.join(traj_output_dir, "trajectory.npz"),
            time=np.array(logger["time"]),
            qpos=np.array(logger["qpos"]),
            qvel=np.array(logger["qvel"]),
            ctrl=np.array(logger["ctrl"]),
            qacc=np.array(logger["qacc"]),
            tau=np.array(logger["tau"])
        )
        
        metrics_output_dir = os.path.join(log_path, "metrics") 
        if not os.path.exists(metrics_output_dir):
            os.makedirs(metrics_output_dir)
        plot_solver_metrics(metrics_log, metrics_output_dir)
        
