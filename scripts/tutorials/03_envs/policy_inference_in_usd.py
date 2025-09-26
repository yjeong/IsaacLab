# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment.

In this example, we use a locomotion policy to control the H1 robot. The robot was trained
using Isaac-Velocity-Rough-H1-v0. The robot is commanded to move forward at a constant velocity.

.. code-block:: bash

    # Run with trained checkpoint (if available)
    ./isaaclab.sh -p scripts/tutorials/03_envs/policy_inference_in_usd.py --checkpoint /path/to/jit/checkpoint.pt
    
    # Run with dummy random policy (for demonstration without trained model)
    ./isaaclab.sh -p scripts/tutorials/03_envs/policy_inference_in_usd.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", default=None)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
from typing import Callable

import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY


def _is_cuda_driver_issue(error: Exception) -> bool:
    """Return True if the exception indicates a CUDA driver incompatibility."""

    message = str(error)
    keywords = [
        "Warp CUDA error",
        "CUDA error 36",
        "cuDeviceGetUuid",
        "API call is not supported in the installed CUDA driver",
    ]
    return any(keyword in message for keyword in keywords)


def _create_env(device_str: str) -> tuple[ManagerBasedRLEnv, int]:
    """Create the environment on the requested device and return it with the action dimension."""

    env_cfg = H1RoughEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    )
    env_cfg.sim.device = device_str
    if device_str.startswith("cpu"):
        env_cfg.sim.use_fabric = False

    env = ManagerBasedRLEnv(cfg=env_cfg)
    action_dim = env.action_manager.total_action_dim
    return env, action_dim


def _load_policy(expected_action_dim: int, device_str: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Load the trained policy if provided, otherwise return a dummy policy."""

    target_device = torch.device(device_str) if not device_str.startswith("cpu") else torch.device("cpu")

    if args_cli.checkpoint is not None:
        policy_path = os.path.abspath(args_cli.checkpoint)
        try:
            # try Omniverse client first (for remote files)
            file_content = omni.client.read_file(policy_path)[2]  # type: ignore[attr-defined]
            file = io.BytesIO(memoryview(file_content).tobytes())
            policy_module = torch.jit.load(file, map_location=target_device)
        except (AttributeError, RuntimeError):
            # fallback to standard file loading for local files
            policy_module = torch.jit.load(policy_path, map_location=target_device)
        try:
            policy_module.to(target_device)
        except AttributeError:
            pass
        print(f"[INFO] 학습된 정책을 로드했습니다: {policy_path}")
        return policy_module

    print("[INFO] 체크포인트가 제공되지 않아 더미 랜덤 정책을 사용합니다.")

    class DummyPolicy:
        def __init__(self, dim: int, device: torch.device):
            self.action_dim = dim
            self.device = device

        def __call__(self, obs: torch.Tensor) -> torch.Tensor:
            if not isinstance(obs, torch.Tensor):
                raise TypeError("Observation tensor expected for dummy policy.")
            batch_size = obs.shape[0]
            return torch.randn(batch_size, self.action_dim, device=self.device) * 0.1

    return DummyPolicy(expected_action_dim, target_device)


def _compute_action(
    policy_fn: Callable[[torch.Tensor], torch.Tensor],
    policy_obs: torch.Tensor,
    action_dim: int,
    default_device: torch.device,
) -> torch.Tensor:
    """Run the policy and sanity-check the resulting action tensor."""

    action = policy_fn(policy_obs)
    if isinstance(action, tuple):
        action = action[0]
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action)
    if action.ndim == 1:
        action = action.unsqueeze(0)

    target_device = default_device
    if isinstance(policy_obs, torch.Tensor):
        target_device = policy_obs.device

    action = action.to(target_device)

    if action.shape[-1] != action_dim:
        raise ValueError(
            f"정책이 반환한 액션 차원이 잘못되었습니다. 기대값: {action_dim}, 실제값: {action.shape[-1]}."
        )
    return action


def _run_inference(env: ManagerBasedRLEnv, policy: Callable[[torch.Tensor], torch.Tensor], action_dim: int, device_str: str):
    """Reset the environment and execute inference until the app stops."""

    default_device = torch.device(device_str) if not device_str.startswith("cpu") else torch.device("cpu")
    obs, _ = env.reset()

    with torch.inference_mode():
        while simulation_app.is_running():
            action = _compute_action(policy, obs["policy"], action_dim, default_device)
            obs, _, _, _, _ = env.step(action)


def main():
    """Main function."""

    preferred_device = args_cli.device
    device_candidates = [preferred_device]
    if not preferred_device.startswith("cpu"):
        device_candidates.append("cpu")

    last_error: Exception | None = None

    for device_str in device_candidates:
        env: ManagerBasedRLEnv | None = None
        try:
            env, action_dim = _create_env(device_str)
            policy = _load_policy(action_dim, device_str)
            print(f"[INFO] '{device_str}' 장치에서 추론을 실행합니다.")
            _run_inference(env, policy, action_dim, device_str)
            return
        except Exception as err:  # noqa: BLE001
            if not device_str.startswith("cpu") and _is_cuda_driver_issue(err):
                print(
                    "[WARN] GPU 모드에서 CUDA 드라이버 문제가 감지되어 CPU 모드로 자동 전환합니다."
                )
                last_error = err
                if env is not None:
                    env.close()
                continue
            raise

    raise RuntimeError(
        "GPU 및 CPU 장치에서 환경 초기화에 실패했습니다. CUDA 드라이버 또는 Isaac Sim 설치를 확인하세요."
    ) from last_error


if __name__ == "__main__":
    main()
    simulation_app.close()
