# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to generate log outputs while the simulation plays.
It accompanies the tutorial on docker usage.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import os
import torch

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating logs from within the docker container.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# add tutorial-specific arguments
parser.add_argument(
    "--sim-duration",
    type=float,
    default=10.0,
    help="시뮬레이션 실행 시간(초). 기본값 10초. 0 이하이면 무제한으로 실행.",
)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""
    # Specify that the logs must be in logs/docker_tutorial
    log_dir_path = os.path.join("logs")
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    # In the container, the absolute path will be
    # /workspace/isaaclab/logs/docker_tutorial, because
    # all python execution is done through /workspace/isaaclab/isaaclab.sh
    # and the calling process' path will be /workspace/isaaclab
    log_dir_path = os.path.abspath(os.path.join(log_dir_path, "docker_tutorial"))
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    print(f"[INFO] Logging experiment to directory: {log_dir_path}")

    # Initialize the simulation context
    # Resolve device: prefer CLI value, otherwise CUDA if available, else CPU
    device = getattr(args_cli, "device", None)
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 비활성(PyTorch CPU 빌드). CPU로 강제 전환합니다.")
        device = "cpu"

    sim_cfg = SimulationCfg(dt=0.01, device=device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Prepare to count sim_time
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    max_sim_time = float(args_cli.sim_duration)
    step_count = 0

    # Open logging file
    with open(os.path.join(log_dir_path, "log.txt"), "w") as log_file:
        # Simulate physics
        while simulation_app.is_running():
            if max_sim_time > 0.0 and sim_time >= max_sim_time:
                break
            log_file.write(f"{sim_time:.4f}\n")
            log_file.flush()  # Ensure data is written immediately
            
            # perform step
            sim.step()
            sim_time += sim_dt
            step_count += 1
            
            # Print progress every 60 steps (1 second at 60 FPS)
            if step_count % 60 == 0:
                print(f"[INFO] Simulation time: {sim_time:.2f}s")
    
    print(f"[INFO] Simulation completed. Total time: {sim_time:.2f}s, Steps: {step_count}")
    print(f"[INFO] Log file written to: {os.path.join(log_dir_path, 'log.txt')}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
