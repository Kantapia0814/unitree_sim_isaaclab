# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Minimal ground environment for GR00T-WBC wholebody testing.
Ground plane + robot + cameras. No tables, objects, or room walls.
Matches MuJoCo's scene_29dof.xml simplicity.
"""
import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from . import mdp

from tasks.common_config import G1RobotPresets, CameraPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

# Import minimal scene
from tasks.common_scene.base_scene_minimal_ground_wholebody import MinimalGroundSceneCfgWH


##
# Scene definition
##
@configclass
class MinimalGroundSceneCfg(MinimalGroundSceneCfgWH):
    """Minimal scene: ground plane + robot + contact sensor + cameras."""
    
    # Robot at origin, facing forward, standing on ground
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=(0.0, 0.0, 0.8),
        init_rot=(1.0, 0.0, 0.0, 0.0),  # No rotation - face forward
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    # Camera configuration
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()
    robot_camera = CameraPresets.g1_world_camera()


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Joint position control actions."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observations for the policy."""
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gripper_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    pass


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    pass


@configclass
class MinimalGroundG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal ground environment config for GR00T-WBC testing.
    
    Physics settings match the move_cylinder_wholebody task (tuned for MuJoCo compatibility):
    - friction: static=0.8, dynamic=0.6, combine_mode="multiply"
    - depenetration_velocity=10.0 (in robot config)
    - solver iterations: 8/4 (in robot config)
    """

    # Scene
    scene: MinimalGroundSceneCfg = MinimalGroundSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization - physics settings tuned for GR00T-WBC compatibility."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 60.0  # Longer episode for testing
        
        # Simulation settings
        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Physics material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        # Event manager (minimal - just scene reset)
        self.event_manager = SimpleEventManager()
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device),
            )
        ))
