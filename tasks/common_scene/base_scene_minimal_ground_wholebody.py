# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Minimal ground plane scene for GR00T-WBC testing.
Only contains: ground plane, light, and robot. No tables, objects, or room walls.
Matches the simplicity of MuJoCo's scene_29dof.xml (floor + robot only).
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass


@configclass
class MinimalGroundSceneCfgWH(InteractiveSceneCfg):
    """Minimal ground plane scene configuration.
    
    Contains only:
    - Ground plane (like MuJoCo's floor geom)
    - Dome light
    - Robot (to be set by the task env cfg)
    - Contact sensor
    
    No room walls, tables, objects, or cameras.
    """

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )
