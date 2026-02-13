# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Minimal ground environment for GR00T-WBC wholebody testing (G1 29DoF + Dex3)."""

import gymnasium as gym
from . import minimal_ground_g1_29dof_dex3_wh_env_cfg

gym.register(
    id="Isaac-MinimalGround-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": minimal_ground_g1_29dof_dex3_wh_env_cfg.MinimalGroundG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
