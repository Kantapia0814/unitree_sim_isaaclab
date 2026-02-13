# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from action_provider.action_base import ActionProvider
from typing import Optional
import torch
from dds.dds_master import dds_manager
import os
import csv
import math
import onnxruntime as ort
from dds.sharedmemorymanager import SharedMemoryManager
import time
import threading
from datetime import datetime
from isaaclab.utils.buffers import CircularBuffer,DelayBuffer
import os
import ast
project_root = os.environ.get("PROJECT_ROOT")
class DDSRLActionProvider(ActionProvider):
    """Action provider based on DDS"""
    
    def __init__(self,env, args_cli):
        super().__init__("DDSActionProvider")
        self.enable_robot = args_cli.robot_type
        self.enable_gripper = args_cli.enable_dex1_dds
        self.enable_dex3 = args_cli.enable_dex3_dds
        self.enable_inspire = args_cli.enable_inspire_dds
        self.wh = args_cli.enable_wholebody_dds
        self.full_body_mode = getattr(args_cli, 'enable_fullbody_dds', False)
        self.policy_path = f"{project_root}/"+args_cli.model_path
        self.env = env
        # Initialize DDS communication
        self.robot_dds = None
        self.gripper_dds = None
        self.dex3_dds = None
        self.inspire_dds = None
        self.run_command = None
        self._setup_dds()
        self._setup_joint_mapping()
        self.policy = self.load_policy(self.policy_path)
        
        # GR00T-WBC full body mode setup (TORQUE MODE)
        # Faithfully reproduces real robot / MuJoCo actuator model:
        #   torque = tau + kp*(q_target - q_current) + kd*(dq_target - dq_current)
        if self.full_body_mode:
            self._fb_dds_connected = False
            self._fb_step_count = 0
            self._fb_pd_zeroed = False
            # Full body joint mapping: GR00T-WBC DDS motor index -> Isaac Sim joint name
            self._fb_joint_mapping = {
                "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
                "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
                "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
                "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
                "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
                "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16, "left_shoulder_yaw_joint": 17,
                "left_elbow_joint": 18, "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20, "left_wrist_yaw_joint": 21,
                "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23, "right_shoulder_yaw_joint": 24,
                "right_elbow_joint": 25, "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27, "right_wrist_yaw_joint": 28,
            }
            device = self.env.device
            
            # --- Motor effort (torque) limits from MuJoCo g1_gear_wbc.xml actuatorfrcrange ---
            self._motor_effort_limit = torch.tensor([
                88.0, 139.0, 88.0, 139.0, 50.0, 50.0,   # left leg  (pitch, roll, yaw, knee, ankle_p, ankle_r)
                88.0, 139.0, 88.0, 139.0, 50.0, 50.0,   # right leg (pitch, roll, yaw, knee, ankle_p, ankle_r)
                88.0, 50.0, 50.0,                          # waist    (yaw, roll, pitch)
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,  # left arm
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0   # right arm
            ], device=device, dtype=torch.float32)
            
            # --- Joint position limits (for sanity clamping q_target) ---
            self._motor_pos_lower = torch.tensor([
                -2.5307, -0.5236, -2.7576, -0.0873, -0.8727, -0.2618,
                -2.5307, -2.9671, -2.7576, -0.0873, -0.8727, -0.2618,
                -2.618,  -0.52,   -0.52,
                -3.0892, -1.5882, -2.618,  -1.0472, -1.972,  -1.614, -1.614,
                -3.0892, -2.2515, -2.618,  -1.0472, -1.972,  -1.614, -1.614
            ], device=device, dtype=torch.float32)
            self._motor_pos_upper = torch.tensor([
                2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
                2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
                2.618,  0.52,   0.52,
                2.6704, 2.2515, 2.618,  2.0944, 1.972,  1.614, 1.614,
                2.6704, 1.5882, 2.618,  2.0944, 1.972,  1.614, 1.614
            ], device=device, dtype=torch.float32)
            
            self._wbc_sanity_reject_count = 0
            
            # Build index mapping tensors
            fb_targets = []
            fb_sources = []
            for jname, motor_idx in self._fb_joint_mapping.items():
                if jname in self.joint_to_index:
                    fb_targets.append(self.joint_to_index[jname])
                    fb_sources.append(motor_idx)
            self._fb_target_idx_t = torch.tensor(fb_targets, dtype=torch.long, device=device)
            self._fb_source_idx_t = torch.tensor(fb_sources, dtype=torch.long, device=device)
            
            # Effort limit tensor in Isaac Sim joint index order
            num_all_joints = len(self.env.scene["robot"].data.joint_names)
            self._effort_limit_sim = torch.full((num_all_joints,), 50.0, device=device, dtype=torch.float32)
            for jname, motor_idx in self._fb_joint_mapping.items():
                if jname in self.joint_to_index:
                    self._effort_limit_sim[self.joint_to_index[jname]] = self._motor_effort_limit[motor_idx]
            
            # DDS field buffers (29 motors each)
            self._dds_q = torch.zeros(29, device=device, dtype=torch.float32)
            self._dds_dq = torch.zeros(29, device=device, dtype=torch.float32)
            self._dds_tau = torch.zeros(29, device=device, dtype=torch.float32)
            self._dds_kp = torch.zeros(29, device=device, dtype=torch.float32)
            self._dds_kd = torch.zeros(29, device=device, dtype=torch.float32)
            
            # === Sim-to-sim gap compensation ===
            # PhysX rigid contacts lack MuJoCo's implicit damping/compliance.
            # However, filtering position commands fights the WBC policy (the policy
            # observes the real robot state, so delayed commands cause a positive
            # feedback loop).  Instead, we use only a very mild damping boost.
            #
            # KEY LESSON: Do NOT delay/filter WBC position targets — the policy
            # expects instantaneous command application.  Only add modest damping.
            self._kd_multiplier = 1.0      # damping multiplier (1.0 = no boost)
            self._extra_vel_damping = 0.0  # additional velocity damping (0.0 = disabled)
            
            # Torque buffer in Isaac Sim joint index order
            self._torque_buf = torch.zeros(num_all_joints, device=device, dtype=torch.float32)
            
            # Pre-compute motor-to-sim index mapping for efficient torque computation
            # _motor_to_sim_idx[motor_idx] = sim_joint_idx
            self._motor_to_sim_idx = torch.full((29,), -1, dtype=torch.long, device=device)
            for i in range(len(self._fb_source_idx_t)):
                motor_idx = self._fb_source_idx_t[i].item()
                sim_idx = self._fb_target_idx_t[i].item()
                self._motor_to_sim_idx[motor_idx] = sim_idx
            
            print(f"[{self.name}] GR00T-WBC full body TORQUE mode: mapped {len(fb_targets)}/29 joints")
            print(f"[{self.name}] Actuator model: torque = tau + kp*(q-q_cur) + kd*(dq-dq_cur)")
            print(f"[{self.name}] Waiting for GR00T-WBC DDS commands (default pose until connected)")
            # Initialize CSV debug logger
            self._init_debug_logger()
        
        # 预计算索引张量与复用缓冲
        device = self.env.device
        if hasattr(self, "arm_joint_mapping") and self.arm_joint_mapping:
            self._arm_target_indices = [self.joint_to_index[name] for name in self.arm_joint_mapping.keys()]
            self._arm_source_indices = [idx + 15 for idx in self.arm_joint_mapping.values()]
            self._arm_target_idx_t = torch.tensor(self._arm_target_indices, dtype=torch.long, device=device)
            self._arm_source_idx_t = torch.tensor(self._arm_source_indices, dtype=torch.long, device=device)
        if self.enable_gripper:
            self._gripper_target_indices = [self.joint_to_index[name] for name in self.gripper_joint_mapping.keys()]
            self._gripper_source_indices = [idx for idx in self.gripper_joint_mapping.values()]
            self._gripper_target_idx_t = torch.tensor(self._gripper_target_indices, dtype=torch.long, device=device)
            self._gripper_source_idx_t = torch.tensor(self._gripper_source_indices, dtype=torch.long, device=device)
        if self.enable_dex3:
            self._left_hand_target_indices = [self.joint_to_index[name] for name in self.left_hand_joint_mapping.keys()]
            self._left_hand_source_indices = [idx for idx in self.left_hand_joint_mapping.values()]
            self._right_hand_target_indices = [self.joint_to_index[name] for name in self.right_hand_joint_mapping.keys()]
            self._right_hand_source_indices = [idx for idx in self.right_hand_joint_mapping.values()]
            self._left_hand_target_idx_t = torch.tensor(self._left_hand_target_indices, dtype=torch.long, device=device)
            self._left_hand_source_idx_t = torch.tensor(self._left_hand_source_indices, dtype=torch.long, device=device)
            self._right_hand_target_idx_t = torch.tensor(self._right_hand_target_indices, dtype=torch.long, device=device)
            self._right_hand_source_idx_t = torch.tensor(self._right_hand_source_indices, dtype=torch.long, device=device)
        if self.enable_inspire:
            self._inspire_target_indices = [self.joint_to_index[name] for name in self.inspire_hand_joint_mapping.keys()]
            self._inspire_source_indices = [idx for idx in self.inspire_hand_joint_mapping.values()]
            self._inspire_special_target_indices = [self.joint_to_index[name] for name in self.special_joint_mapping.keys()]
            self._inspire_special_source_indices = [spec[0] for spec in self.special_joint_mapping.values()]
            self._inspire_special_scales = torch.tensor([spec[1] for spec in self.special_joint_mapping.values()], dtype=torch.float32)
            self._inspire_target_idx_t = torch.tensor(self._inspire_target_indices, dtype=torch.long, device=device)
            self._inspire_source_idx_t = torch.tensor(self._inspire_source_indices, dtype=torch.long, device=device)
            self._inspire_special_target_idx_t = torch.tensor(self._inspire_special_target_indices, dtype=torch.long, device=device)
            self._inspire_special_source_idx_t = torch.tensor(self._inspire_special_source_indices, dtype=torch.long, device=device)
            self._inspire_special_scales_t = self._inspire_special_scales.to(device)
        
        self._full_action_buf = torch.zeros(len(self.all_joint_names), device=device, dtype=torch.float32)
        self._positions_buf = torch.empty(29, device=device, dtype=torch.float32)
        if self.enable_gripper:
            self._gripper_buf = torch.empty(2, device=device, dtype=torch.float32)
        if self.enable_dex3:
            self._left_hand_buf = torch.empty(len(self._left_hand_source_indices), device=device, dtype=torch.float32)
            self._right_hand_buf = torch.empty(len(self._right_hand_source_indices), device=device, dtype=torch.float32)
        if self.enable_inspire:
            self._inspire_buf = torch.empty(12, device=device, dtype=torch.float32)
        
    def _init_debug_logger(self):
        """Initialize CSV debug logger for fullbody torque mode diagnostics."""
        log_dir = os.path.join(project_root, "debug_logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._debug_log_path = os.path.join(log_dir, f"wbc_torque_debug_{timestamp}.csv")
        
        # 6 left-leg joint names matching DDS motor order (pitch, roll, yaw)
        self._log_leg_names = [
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
            "left_knee", "left_ankle_pitch", "left_ankle_roll",
        ]
        # Build CSV header - includes all 5 DDS fields + computed torques
        header = ["step", "timestamp_ms", "mode", "wbc_connected", "has_wbc_cmd"]
        for n in self._log_leg_names:
            header.append(f"wbc_q_{n}")
        for n in self._log_leg_names:
            header.append(f"wbc_dq_{n}")
        for n in self._log_leg_names:
            header.append(f"wbc_tau_{n}")
        for n in self._log_leg_names:
            header.append(f"wbc_kp_{n}")
        for n in self._log_leg_names:
            header.append(f"wbc_kd_{n}")
        for n in self._log_leg_names:
            header.append(f"robot_pos_{n}")
        for n in self._log_leg_names:
            header.append(f"robot_vel_{n}")
        for n in self._log_leg_names:
            header.append(f"computed_torque_{n}")
        header += ["root_height", "root_pitch_deg"]
        
        self._debug_csv_file = open(self._debug_log_path, "w", newline="")
        self._debug_csv_writer = csv.writer(self._debug_csv_file)
        self._debug_csv_writer.writerow(header)
        self._debug_fall_detected = False
        self._debug_start_ms = int(time.time() * 1000)
        print(f"[{self.name}] Debug CSV logger: {self._debug_log_path}")
    
    def _log_step(self, mode: str, has_wbc_cmd: bool, dds_fields: dict, computed_torques):
        """Write one row to the debug CSV. Called every step in fullbody torque mode.
        
        Args:
            mode: control mode string
            has_wbc_cmd: whether valid WBC command was received
            dds_fields: dict with 'q','dq','tau','kp','kd' lists (or None)
            computed_torques: tensor of computed torques in sim joint order (or None)
        """
        if not hasattr(self, "_debug_csv_writer") or self._debug_csv_writer is None:
            return
        try:
            robot = self.env.scene["robot"]
            cur_pos = robot.data.joint_pos[0]
            cur_vel = robot.data.joint_vel[0]
            now_ms = int(time.time() * 1000) - self._debug_start_ms
            
            # WBC DDS fields for left leg (motor indices 0-5)
            def fmt_dds(field_name, count=6):
                vals = []
                if dds_fields and field_name in dds_fields and dds_fields[field_name] is not None:
                    for i in range(count):
                        vals.append(f"{dds_fields[field_name][i]:.5f}")
                else:
                    vals = ["nan"] * count
                return vals
            
            wbc_q_vals = fmt_dds('q')
            wbc_dq_vals = fmt_dds('dq')
            wbc_tau_vals = fmt_dds('tau')
            wbc_kp_vals = fmt_dds('kp')
            wbc_kd_vals = fmt_dds('kd')
            
            # Actual robot position and velocity for left leg joints
            robot_pos_vals = []
            robot_vel_vals = []
            for jname in list(self._fb_joint_mapping.keys())[:6]:
                idx = self.joint_to_index.get(jname, 0)
                robot_pos_vals.append(f"{cur_pos[idx].item():.5f}")
                robot_vel_vals.append(f"{cur_vel[idx].item():.5f}")
            
            # Computed torques for left leg joints
            torque_vals = []
            if computed_torques is not None:
                for jname in list(self._fb_joint_mapping.keys())[:6]:
                    idx = self.joint_to_index.get(jname, 0)
                    torque_vals.append(f"{computed_torques[idx].item():.5f}")
            else:
                torque_vals = ["nan"] * 6
            
            # Root state
            root_state = robot.data.root_state_w[0]
            root_height = f"{root_state[2].item():.4f}"
            
            # Compute pitch angle from projected gravity
            proj_grav = robot.data.projected_gravity_b[0]
            gx = proj_grav[0].item()
            gz = proj_grav[2].item()
            pitch_rad = math.atan2(-gx, -gz)
            pitch_deg = f"{math.degrees(pitch_rad):.2f}"
            
            row = [self._fb_step_count, now_ms, mode, self._fb_dds_connected, has_wbc_cmd]
            row += wbc_q_vals + wbc_dq_vals + wbc_tau_vals + wbc_kp_vals + wbc_kd_vals
            row += robot_pos_vals + robot_vel_vals + torque_vals
            row += [root_height, pitch_deg]
            
            self._debug_csv_writer.writerow(row)
            
            # Flush every 100 steps for safety
            if self._fb_step_count % 100 == 0:
                self._debug_csv_file.flush()
            
            # Fall detection: root height < 0.3m
            if float(root_height) < 0.3 and not self._debug_fall_detected:
                self._debug_fall_detected = True
                self._debug_csv_file.flush()
                print(f"\n{'!'*60}")
                print(f"[{self.name}] FALL DETECTED at step {self._fb_step_count}!")
                print(f"[{self.name}] Root height: {root_height}m, Pitch: {pitch_deg} deg")
                print(f"[{self.name}] Debug log saved: {self._debug_log_path}")
                print(f"{'!'*60}\n")
        except Exception as e:
            # Don't let logging errors crash the control loop
            pass
    
    def _close_debug_logger(self):
        """Close the CSV debug logger."""
        if hasattr(self, "_debug_csv_file") and self._debug_csv_file:
            try:
                self._debug_csv_file.flush()
                self._debug_csv_file.close()
                print(f"[{self.name}] Debug CSV saved: {self._debug_log_path}")
            except Exception:
                pass
            self._debug_csv_file = None
            self._debug_csv_writer = None

    def _setup_dds(self):
        """Setup DDS communication"""
        print(f"enable_robot: {self.enable_robot}")
        print(f"enable_gripper: {self.enable_gripper}")
        print(f"enable_dex3: {self.enable_dex3}")
        try:
            if self.enable_robot == "g129":
                self.robot_dds = dds_manager.get_object("g129")
            if self.enable_gripper:
                self.gripper_dds = dds_manager.get_object("dex1")
            elif self.enable_dex3:
                self.dex3_dds = dds_manager.get_object("dex3")
            elif self.enable_inspire:
                self.inspire_dds = dds_manager.get_object("inspire")
            if self.wh:
                self.run_command_dds = dds_manager.get_object("run_command")
            print(f"[{self.name}] DDS communication initialized")
        except Exception as e:
            print(f"[{self.name}] DDS initialization failed: {e}")
    
    def _setup_joint_mapping(self):
        """Setup joint mapping"""
        if self.wh:
            self.action_joint_names = [
            'left_hip_pitch_joint', 
            'right_hip_pitch_joint', 
            'left_hip_roll_joint', 
            'right_hip_roll_joint', 
            'left_hip_yaw_joint', 
            'right_hip_yaw_joint', 
            'left_knee_joint', 
            'right_knee_joint', 
            'left_ankle_pitch_joint',
            'right_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_ankle_roll_joint'
            ]
            self.waist_joint_mapping = [
                'waist_yaw_joint',
                'waist_roll_joint',
                'waist_pitch_joint',
            ]
            self.arm_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            # right arm (7)
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            ]
            self.old_action_joints_names = [
            'left_hip_pitch_joint', 
            'right_hip_pitch_joint', 
            'waist_yaw_joint', 
            'left_hip_roll_joint', 
            'right_hip_roll_joint', 
            'waist_roll_joint',
            'left_hip_yaw_joint', 
            'right_hip_yaw_joint', 
            'waist_pitch_joint', 
            'left_knee_joint', 
            'right_knee_joint', 
            'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint', 
            'left_ankle_pitch_joint', 
            'right_ankle_pitch_joint',
            'left_shoulder_roll_joint', 
            'right_shoulder_roll_joint', 
            'left_ankle_roll_joint', 
            'right_ankle_roll_joint', 
            'left_shoulder_yaw_joint', 
            'right_shoulder_yaw_joint', 
            'left_elbow_joint', 
            'right_elbow_joint', 
            'left_wrist_roll_joint', 
            'right_wrist_roll_joint', 
            'left_wrist_pitch_joint', 
            'right_wrist_pitch_joint', 
            'left_wrist_yaw_joint', 
            'right_wrist_yaw_joint',]
        if self.enable_robot == "g129":
            self.arm_joint_mapping = {
                "left_shoulder_pitch_joint": 0,
                "left_shoulder_roll_joint": 1,
                "left_shoulder_yaw_joint": 2,
                "left_elbow_joint": 3,
                "left_wrist_roll_joint": 4,
                "left_wrist_pitch_joint": 5,
                "left_wrist_yaw_joint": 6,
                "right_shoulder_pitch_joint": 7,
                "right_shoulder_roll_joint": 8,
                "right_shoulder_yaw_joint": 9,
                "right_elbow_joint": 10,
                "right_wrist_roll_joint": 11,
                "right_wrist_pitch_joint": 12,
                "right_wrist_yaw_joint": 13
            }
        if self.enable_gripper:
            self.gripper_joint_mapping = {
                "left_hand_Joint1_1": 1,
                "left_hand_Joint2_1": 1,
                "right_hand_Joint1_1": 0,
                "right_hand_Joint2_1": 0,
            }
        if self.enable_dex3:
            self.left_hand_joint_mapping = {
                "left_hand_thumb_0_joint":0,
                "left_hand_thumb_1_joint":1,
                "left_hand_thumb_2_joint":2,
                "left_hand_middle_0_joint":3,
                "left_hand_middle_1_joint":4,
                "left_hand_index_0_joint":5,
                "left_hand_index_1_joint":6}
            self.right_hand_joint_mapping = {
                "right_hand_thumb_0_joint":0,     
                "right_hand_thumb_1_joint":1,
                "right_hand_thumb_2_joint":2,
                "right_hand_middle_0_joint":3,
                "right_hand_middle_1_joint":4,
                "right_hand_index_0_joint":5,
                "right_hand_index_1_joint":6}
        if self.enable_inspire:
            self.inspire_hand_joint_mapping = {
                "R_pinky_proximal_joint":0,
                "R_ring_proximal_joint":1,
                "R_middle_proximal_joint":2,
                "R_index_proximal_joint":3,
                "R_thumb_proximal_pitch_joint":4,
                "R_thumb_proximal_yaw_joint":5,
                "L_pinky_proximal_joint":6,
                "L_ring_proximal_joint":7,
                "L_middle_proximal_joint":8,
                "L_index_proximal_joint":9,
                "L_thumb_proximal_pitch_joint":10,
                "L_thumb_proximal_yaw_joint":11,
            }
            self.special_joint_mapping = {
                "L_index_intermediate_joint":[9,1],
                "L_middle_intermediate_joint":[8,1],
                "L_pinky_intermediate_joint":[6,1],
                "L_ring_intermediate_joint":[7,1],
                "L_thumb_intermediate_joint":[10,1.5],
                "L_thumb_distal_joint":[10,2.4],

                "R_index_intermediate_joint":[3,1],
                "R_middle_intermediate_joint":[2,1],
                "R_pinky_intermediate_joint":[0,1],
                "R_ring_intermediate_joint":[1,1],
                "R_thumb_intermediate_joint":[4,1.5],
                "R_thumb_distal_joint":[4,2.4],
            }
        self.all_joint_names = self.env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}
        self.arm_action_pose = [self.joint_to_index[name] for name in self.arm_joint_mapping.keys()]
        self.arm_action_pose_indices = [self.arm_joint_mapping[name] for name in self.arm_joint_mapping.keys()]
        self.action_to_indices=[]
        for action_joint in self.action_joint_names:
            if action_joint in self.all_joint_names:
                self.action_to_indices.append(self.all_joint_names.index(action_joint))
            else:
                raise ValueError(f"action joint '{action_joint}' not in all joint list")
        self.waist_to_all_indices = []
        for waist_joint in self.waist_joint_mapping:
            if waist_joint in self.all_joint_names:
                self.waist_to_all_indices.append(self.all_joint_names.index(waist_joint))
            else:
                raise ValueError(f"waist joint '{waist_joint}' not in all joint list")

        self.arm_to_all_indices=[]
        for arm_joint in self.arm_joint_names:
            if arm_joint in self.all_joint_names:
                self.arm_to_all_indices.append(self.all_joint_names.index(arm_joint))
            else:
                raise ValueError(f"arm joint '{arm_joint}' not in all joint list")
        self.default_waist_positions = self.env.scene["robot"].data.default_joint_pos[:, self.waist_to_all_indices]
        self.default_action_positions = self.env.scene["robot"].data.default_joint_pos
        self.default_action_velocities = self.env.scene["robot"].data.default_joint_vel
        self.all_obs_indices = self.action_to_indices + self.arm_to_all_indices
        self.old_action_indices = []
        for old_action_joint in self.old_action_joints_names:
            if old_action_joint in self.all_joint_names:
                self.old_action_indices.append(self.all_joint_names.index(old_action_joint))
            else:
                raise ValueError(f"action joint '{old_action_joint}' not in all joint list")
        self.arm_action = []
        self.obs_scales = {"ang_vel":1.0, "projected_gravity":1.0, "commands":1.0, 
                           "joint_pos":1.0, "joint_vel":1.0, "actions":1.0}
        self.ang_vel = self.env.scene["robot"].data.root_ang_vel_b                      
        self.projected_gravity = self.env.scene["robot"].data.projected_gravity_b       
        self.joint_pos = self.env.scene["robot"].data.joint_pos
        self.joint_vel = self.env.scene["robot"].data.joint_vel
        self.actor_obs_buffer = CircularBuffer(
            max_len=10, batch_size=1, device=self.env.device
        )
        self.num_envs =1
        self.clip_obs = 100
        self.num_actions_all = self.env.scene["robot"].data.default_joint_pos[:,self.old_action_indices].shape[1]  
        self.action_buffer = DelayBuffer(
            5, self.num_envs, device=self.env.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions_all, dtype=torch.float, device=self.env.device, requires_grad=False)
        )
        self.clip_actions = 100
        self.action_scale = 0.25
        self.sim_step_counter = 0
    def load_policy(self,path):
        ext = os.path.splitext(path)[1].lower()
        if ext==".onnx":
            return self.load_onnx_policy(path)
        elif ext==".pt":
            return self.load_jit_pt_policy(path)

    def load_jit_pt_policy(self,path):
        return torch.jit.load(path)

    def load_onnx_policy(self,path):
        model = ort.InferenceSession(path)
        def run_inference(input_tensor):
            ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device=self.env.device)
        return run_inference
    def compute_current_observations(self):
        command = [0,0,0,0.8]  
        run_command = self.run_command_dds.get_run_command()
        if run_command and 'run_command' in run_command:
            run_command_data = run_command['run_command']
            
            if isinstance(run_command_data, str):
                try:
                    run_command_list = ast.literal_eval(run_command_data)
                    if isinstance(run_command_list, list) and len(run_command_list) >= 4:
                        command[0] = float(run_command_list[0])
                        command[1] = float(run_command_list[1])
                        command[2] = float(run_command_list[2])
                        command[3] = float(run_command_list[3])
                except (ValueError, SyntaxError) as e:
                    print(f"[WARNING] cannot parse run_command string: {run_command_data}, error: {e}")
            else:
                try:
                    command[0] = float(run_command_data[0])
                    command[1] = float(run_command_data[1])
                    command[2] = float(run_command_data[2])
                    command[3] = float(run_command_data[3])
                except (IndexError, TypeError) as e:
                    print(f"[WARNING] cannot parse run_command data: {run_command_data}, error: {e}")
            
            self.run_command_dds.write_run_command([0.0,0,0,0.8])
      
        # command = [0.5,0.0,0.7,0.8]
        command = torch.tensor(command, device=self.env.device, dtype=torch.float32)
        
        if command.dim() == 1:
            command = command.unsqueeze(0)  # [4] -> [1, 4]
        self.ang_vel = self.env.scene["robot"].data.root_ang_vel_b                      
        self.projected_gravity = self.env.scene["robot"].data.projected_gravity_b       
        self.joint_pos = self.env.scene["robot"].data.joint_pos
        self.joint_vel = self.env.scene["robot"].data.joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]     
        current_actor_obs = torch.cat(
        [
            self.ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity * self.obs_scales["projected_gravity"],
            command * self.obs_scales["commands"],
            (self.joint_pos[:, self.all_obs_indices] - self.default_action_positions[:, self.all_obs_indices]) * self.obs_scales["joint_pos"],
            (self.joint_vel[:, self.all_obs_indices] - self.default_action_velocities[:, self.all_obs_indices]) * self.obs_scales["joint_vel"],
            action * self.obs_scales["actions"],  # [29] -> [1, 29]
        ],
        dim=-1,
    )
        return current_actor_obs
    def compute_observations(self):

        current_actor_obs = self.compute_current_observations()

        self.actor_obs_buffer.append(current_actor_obs)
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        return actor_obs
    
    def run_policy(self):
        current_actor_obs = self.compute_observations()
        action = self.policy(current_actor_obs)
        return action
    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action from DDS"""
        try:
            # === GR00T-WBC full body mode: check for full body commands ===
            if self.full_body_mode:
                return self._get_action_fullbody_or_fallback(env)

            # === Normal wholebody mode: RL policy for legs, DDS for arms ===
            full_action = self._full_action_buf
            full_action.zero_()
            action_data = self.run_policy()

            # RL output + waist default
            full_action[self.action_to_indices] = action_data
            full_action[self.waist_to_all_indices] = self.default_waist_positions
            # Robot arm commands (if any)
            if self.enable_robot == "g129" and self.robot_dds:
                cmd_data = self.robot_dds.get_robot_command()
                if cmd_data and 'motor_cmd' in cmd_data:
                    positions = cmd_data['motor_cmd']['positions']
                    if len(positions) >= 29 and hasattr(self, "_arm_source_idx_t"):
                        self._positions_buf[:29].copy_(torch.tensor(positions[:29], dtype=torch.float32, device=self.env.device))
                        arm_vals = self._positions_buf.index_select(0, self._arm_source_idx_t)
                        full_action.index_copy_(0, self._arm_target_idx_t, arm_vals)
            # Delay/clip/scale
            delayed_actions = self.action_buffer.compute(full_action[self.old_action_indices].unsqueeze(0))
            cliped_actions = torch.clip(delayed_actions[:,self.action_to_indices], -self.clip_actions, self.clip_actions).to(self.env.device)
            full_action[self.action_to_indices] = cliped_actions * self.action_scale + self.default_action_positions[:, self.action_to_indices]
            
            # Gripper/hand commands
            self._apply_hand_commands(full_action)

            # Step simulation
            self._step_simulation(full_action)
            
        except Exception as e:
            print(f"[{self.name}] Get DDS action failed: {e}")
            return None

    def _run_rl_balance(self):
        """Run RL balance policy and return full_action tensor.
        
        Used ONLY before GR00T-WBC connects to keep the robot standing.
        Once WBC connects, this is never called again.
        """
        full_action = self._full_action_buf.clone()
        full_action.zero_()
        action_data = self.run_policy()
        full_action[self.action_to_indices] = action_data
        full_action[self.waist_to_all_indices] = self.default_waist_positions
        delayed_actions = self.action_buffer.compute(full_action[self.old_action_indices].unsqueeze(0))
        cliped_actions = torch.clip(delayed_actions[:, self.action_to_indices], -self.clip_actions, self.clip_actions).to(self.env.device)
        full_action[self.action_to_indices] = cliped_actions * self.action_scale + self.default_action_positions[:, self.action_to_indices]
        return full_action

    def _check_wbc_sanity(self, values, label="values"):
        """Check if a list of values are physically sane (no NaN/Inf/overflow).
        
        Returns:
            True if all values are valid, False if any should be rejected.
        """
        for i in range(min(len(values), 29)):
            v = values[i]
            if math.isnan(v) or math.isinf(v) or abs(v) > 1e6:
                return False
        return True

    def _zero_pd_gains_for_wbc_joints(self):
        """Zero out Isaac Sim's internal PD gains for WBC-controlled joints.
        
        This is critical: we compute torques ourselves using GR00T-WBC's formula,
        so the sim's internal PD controller must produce zero torque for these joints.
        Non-WBC joints (hands etc.) keep their original PD gains.
        """
        if self._fb_pd_zeroed:
            return
        try:
            robot = self.env.scene["robot"]
            new_stiffness = robot.data.joint_stiffness[0].clone()
            new_damping = robot.data.joint_damping[0].clone()
            
            print(f"[{self.name}] === Zeroing PD gains for WBC joints ===")
            print(f"[{self.name}] Before: stiffness[:15]={new_stiffness[:15].tolist()}")
            print(f"[{self.name}] Before: damping[:15]={new_damping[:15].tolist()}")
            
            # Zero out PD gains only for the 29 WBC-controlled joints
            for jname in self._fb_joint_mapping.keys():
                if jname in self.joint_to_index:
                    sim_idx = self.joint_to_index[jname]
                    new_stiffness[sim_idx] = 0.0
                    new_damping[sim_idx] = 0.0
            
            stiffness_2d = new_stiffness.unsqueeze(0)
            damping_2d = new_damping.unsqueeze(0)
            robot.write_joint_stiffness_to_sim(stiffness_2d)
            robot.write_joint_damping_to_sim(damping_2d)
            
            print(f"[{self.name}] After:  stiffness[:15]={new_stiffness[:15].tolist()}")
            print(f"[{self.name}] After:  damping[:15]={new_damping[:15].tolist()}")
            print(f"[{self.name}] PD gains zeroed for WBC joints -> effort-only control")
            
            self._fb_pd_zeroed = True
        except Exception as e:
            print(f"[{self.name}] WARNING: Failed to zero PD gains: {e}")
            import traceback
            traceback.print_exc()

    def _get_action_fullbody_or_fallback(self, env) -> None:
        """GR00T-WBC full body TORQUE mode.
        
        Faithfully reproduces the real robot / MuJoCo actuator model:
            torque = tau + kp*(q_target - q_current) + kd*(dq_target - dq_current)
        
        Strategy:
        - Before GR00T-WBC connects: hold default pose via position control
        - On first WBC command: zero sim PD gains, switch to effort (torque) control
        - Read all 5 DDS fields: q, dq, tau, kp, kd
        - Compute torques ourselves and apply via set_joint_effort_target
        - Sanity check all fields, clamp torques to effort limits
        - No internal RL fallback: GR00T-WBC is responsible for balance
        """
        # --- Read DDS command (all 5 fields) ---
        has_wbc_cmd = False
        dds_fields = None
        if self.enable_robot == "g129" and self.robot_dds:
            cmd_data = self.robot_dds.get_robot_command()
            if cmd_data and 'motor_cmd' in cmd_data:
                motor_cmd = cmd_data['motor_cmd']
                positions = motor_cmd.get('positions', [])
                velocities = motor_cmd.get('velocities', [])
                torques = motor_cmd.get('torques', [])
                kp = motor_cmd.get('kp', [])
                kd = motor_cmd.get('kd', [])
                
                if len(positions) >= 29:
                    q_list = list(positions[:29])
                    dq_list = list(velocities[:29]) if len(velocities) >= 29 else [0.0] * 29
                    tau_list = list(torques[:29]) if len(torques) >= 29 else [0.0] * 29
                    kp_list = list(kp[:29]) if len(kp) >= 29 else [0.0] * 29
                    kd_list = list(kd[:29]) if len(kd) >= 29 else [0.0] * 29
                    
                    # Check if positions are valid (not all near zero = no real command)
                    max_abs = max(abs(p) for p in q_list[:12])
                    if max_abs > 0.001:
                        # Sanity check ALL fields
                        if (self._check_wbc_sanity(q_list, "q") and
                            self._check_wbc_sanity(dq_list, "dq") and
                            self._check_wbc_sanity(tau_list, "tau") and
                            self._check_wbc_sanity(kp_list, "kp") and
                            self._check_wbc_sanity(kd_list, "kd")):
                            has_wbc_cmd = True
                            dds_fields = {
                                'q': q_list, 'dq': dq_list,
                                'tau': tau_list, 'kp': kp_list, 'kd': kd_list
                            }
                        else:
                            self._wbc_sanity_reject_count += 1
                            if self._wbc_sanity_reject_count % 50 == 1:
                                print(f"[{self.name}] WBC SANITY REJECT #{self._wbc_sanity_reject_count}: "
                                      f"NaN/Inf/overflow in DDS fields")

        # --- First WBC connection: zero sim PD gains for torque-only control ---
        if has_wbc_cmd and not self._fb_dds_connected:
            self._fb_dds_connected = True
            self._zero_pd_gains_for_wbc_joints()
            
            print(f"\n{'='*60}")
            print(f"[{self.name}] GR00T-WBC CONNECTED - TORQUE MODE ACTIVE!")
            print(f"[{self.name}] torque = tau + kp*(q - q_cur) + kd*(dq - dq_cur)")
            print(f"[{self.name}] kd_mult={self._kd_multiplier}x, extra_vel_damp={self._extra_vel_damping}")
            print(f"[{self.name}] WBC q (legs[:6]): {[f'{v:.4f}' for v in dds_fields['q'][:6]]}")
            print(f"[{self.name}] WBC kp (legs[:6]): {[f'{v:.1f}' for v in dds_fields['kp'][:6]]}")
            print(f"[{self.name}] WBC kd (legs[:6]): {[f'{v:.1f}' for v in dds_fields['kd'][:6]]}")
            print(f"[{self.name}] Isaac Sim PD gains ZEROED for WBC joints")
            print(f"{'='*60}\n")

        self._fb_step_count += 1
        
        computed_torques = None

        if not self._fb_dds_connected:
            # === RL BALANCE MODE: use internal RL policy to keep robot standing ===
            # The robot CANNOT balance with static default poses alone.
            # RL policy actively adjusts joint commands to resist gravity.
            full_action = self._run_rl_balance()
            self._apply_hand_commands(full_action)
            
            if self._fb_step_count % 500 == 1:
                root_h, pitch_d = self._get_robot_pitch_and_height()
                print(f"[{self.name}] Step #{self._fb_step_count} RL_BALANCE "
                      f"h={root_h:.3f}m pitch={pitch_d:.1f}° (waiting for WBC)")
            
            # Debug logging
            self._log_step("RL_BALANCE", False, None, None)
            # Use normal position control stepping (RL outputs position targets)
            self._step_simulation(full_action)
        else:
            # === TORQUE CONTROL MODE ===
            robot = self.env.scene["robot"]
            cur_pos = robot.data.joint_pos[0]   # current joint positions
            cur_vel = robot.data.joint_vel[0]   # current joint velocities
            
            if has_wbc_cmd:
                # Copy DDS fields directly into tensors (motor index order, 29 each)
                # NO filtering — the WBC policy expects instantaneous command application.
                # Filtering creates tracking delay that confuses the policy (it observes
                # the real robot state and expects its commands to take effect immediately).
                self._dds_q.copy_(torch.tensor(dds_fields['q'], dtype=torch.float32, device=self.env.device))
                self._dds_dq.copy_(torch.tensor(dds_fields['dq'], dtype=torch.float32, device=self.env.device))
                self._dds_tau.copy_(torch.tensor(dds_fields['tau'], dtype=torch.float32, device=self.env.device))
                self._dds_kp.copy_(torch.tensor(dds_fields['kp'], dtype=torch.float32, device=self.env.device))
                self._dds_kd.copy_(torch.tensor(dds_fields['kd'], dtype=torch.float32, device=self.env.device))
                
                # Clamp q_target to joint limits (motor index order)
                self._dds_q.clamp_(self._motor_pos_lower, self._motor_pos_upper)
                
                # Get current joint state in motor index order
                sim_indices = self._motor_to_sim_idx  # [29] tensor
                q_cur_for_motors = cur_pos[sim_indices]    # [29]
                dq_cur_for_motors = cur_vel[sim_indices]   # [29]
                
                # Compute torque: tau + kp*(q_target - q_cur) + kd*(dq_target - dq_cur)
                effective_kd = self._dds_kd * self._kd_multiplier
                motor_torques = (self._dds_tau 
                                 + self._dds_kp * (self._dds_q - q_cur_for_motors) 
                                 + effective_kd * (self._dds_dq - dq_cur_for_motors)
                                 - self._extra_vel_damping * dq_cur_for_motors)
                
                # Clamp torques to motor effort limits
                motor_torques.clamp_(-self._motor_effort_limit, self._motor_effort_limit)
                
                # Map motor torques back to Isaac Sim joint index order
                self._torque_buf.zero_()
                self._torque_buf[sim_indices] = motor_torques
                
                computed_torques = self._torque_buf.clone()
            else:
                # No new WBC command: hold last position targets with PD control
                sim_indices = self._motor_to_sim_idx
                q_cur_for_motors = cur_pos[sim_indices]
                dq_cur_for_motors = cur_vel[sim_indices]
                effective_kd = self._dds_kd * self._kd_multiplier
                motor_torques = (self._dds_kp * (self._dds_q - q_cur_for_motors)
                                 + effective_kd * (self._dds_dq - dq_cur_for_motors)
                                 - self._extra_vel_damping * dq_cur_for_motors)
                motor_torques.clamp_(-self._motor_effort_limit, self._motor_effort_limit)
                self._torque_buf.zero_()
                self._torque_buf[sim_indices] = motor_torques
                computed_torques = self._torque_buf.clone()
            
            # Diagnostic logging
            if self._fb_step_count % 200 == 1:
                root_h, pitch_d = self._get_robot_pitch_and_height()
                if has_wbc_cmd:
                    leg_q = [f'{dds_fields["q"][i]:.3f}' for i in range(6)]
                    leg_tau = [f'{dds_fields["tau"][i]:.3f}' for i in range(6)]
                    leg_kp = [f'{dds_fields["kp"][i]:.1f}' for i in range(6)]
                    leg_torque = []
                    for jname in list(self._fb_joint_mapping.keys())[:6]:
                        idx = self.joint_to_index.get(jname, 0)
                        leg_torque.append(f'{computed_torques[idx].item():.3f}')
                    print(f"[{self.name}] Step #{self._fb_step_count} TORQUE_CTRL "
                          f"h={root_h:.3f}m pitch={pitch_d:.1f}°")
                    print(f"  WBC q (legs):       {leg_q}")
                    print(f"  WBC tau_ff (legs):   {leg_tau}")
                    print(f"  WBC kp (legs):       {leg_kp}")
                    print(f"  Computed torque:     {leg_torque}")
                    if self._wbc_sanity_reject_count > 0:
                        print(f"  Sanity rejects: {self._wbc_sanity_reject_count}")
                else:
                    print(f"[{self.name}] Step #{self._fb_step_count} TORQUE_HOLD "
                          f"h={root_h:.3f}m pitch={pitch_d:.1f}°")
            
            # Apply hand commands (position control) alongside body torques.
            # Hand joints retain their PD gains (not zeroed by _zero_pd_gains_for_wbc_joints),
            # so setting position targets will make them track the Dex3 DDS commands.
            hand_pos_action = self._full_action_buf.clone()
            hand_pos_action.zero_()
            self._apply_hand_commands(hand_pos_action)

            # Debug CSV logging
            mode = "TORQUE_CTRL" if has_wbc_cmd else "TORQUE_HOLD"
            self._log_step(mode, has_wbc_cmd, dds_fields, computed_torques)
            
            # Step simulation with TORQUE control for body + POSITION control for hands
            self._step_simulation_torque(computed_torques, hand_pos_action)
        
        return None

    def _get_robot_pitch_and_height(self):
        """Get robot root height and pitch angle in degrees."""
        robot = self.env.scene["robot"]
        root_state = robot.data.root_state_w[0]
        root_height = root_state[2].item()
        proj_grav = robot.data.projected_gravity_b[0]
        gx = proj_grav[0].item()
        gz = proj_grav[2].item()
        pitch_rad = math.atan2(-gx, -gz)
        pitch_deg = math.degrees(pitch_rad)
        return root_height, pitch_deg

    def _apply_hand_commands(self, full_action):
        """Apply gripper/dex3/inspire hand commands to full_action."""
        if self.gripper_dds and hasattr(self, "_gripper_source_idx_t"):
            gripper_cmd = self.gripper_dds.get_gripper_command()
            if gripper_cmd:
                left_gripper_cmd = gripper_cmd.get('left_gripper_cmd', {})
                right_gripper_cmd = gripper_cmd.get('right_gripper_cmd', {})
                left_gripper_positions = left_gripper_cmd.get('positions', [])
                right_gripper_positions = right_gripper_cmd.get('positions', [])
                gripper_positions = right_gripper_positions + left_gripper_positions
                if len(gripper_positions) >= 2:
                    self._gripper_buf.copy_(torch.tensor(gripper_positions[:2], dtype=torch.float32, device=self.env.device))
                    gp_vals = self._gripper_buf.index_select(0, self._gripper_source_idx_t)
                    full_action.index_copy_(0, self._gripper_target_idx_t, gp_vals)
        elif self.dex3_dds and hasattr(self, "_left_hand_source_idx_t"):
            hand_cmds = self.dex3_dds.get_hand_commands()
            if hand_cmds:
                left_hand_cmd = hand_cmds.get('left_hand_cmd', {})
                right_hand_cmd = hand_cmds.get('right_hand_cmd', {})
                if left_hand_cmd and right_hand_cmd:
                    left_positions = left_hand_cmd.get('positions', [])
                    right_positions = right_hand_cmd.get('positions', [])
                    if len(left_positions) >= len(self._left_hand_buf) and len(right_positions) >= len(self._right_hand_buf):
                        self._left_hand_buf.copy_(torch.tensor(left_positions[:len(self._left_hand_buf)], dtype=torch.float32, device=self.env.device))
                        self._right_hand_buf.copy_(torch.tensor(right_positions[:len(self._right_hand_buf)], dtype=torch.float32, device=self.env.device))
                        l_vals = self._left_hand_buf.index_select(0, self._left_hand_source_idx_t)
                        r_vals = self._right_hand_buf.index_select(0, self._right_hand_source_idx_t)
                        full_action.index_copy_(0, self._left_hand_target_idx_t, l_vals)
                        full_action.index_copy_(0, self._right_hand_target_idx_t, r_vals)
        elif self.inspire_dds and hasattr(self, "_inspire_source_idx_t"):
            inspire_cmds = self.inspire_dds.get_inspire_hand_command()
            if inspire_cmds and 'positions' in inspire_cmds:
                    inspire_cmds_positions = inspire_cmds['positions']
                    if len(inspire_cmds_positions) >= 12:
                        self._inspire_buf.copy_(torch.tensor(inspire_cmds_positions[:12], dtype=torch.float32, device=self.env.device))
                        base_vals = self._inspire_buf.index_select(0, self._inspire_source_idx_t)
                        full_action.index_copy_(0, self._inspire_target_idx_t, base_vals)
                        special_vals = self._inspire_buf.index_select(0, self._inspire_special_source_idx_t) * self._inspire_special_scales_t
                        full_action.index_copy_(0, self._inspire_special_target_idx_t, special_vals)

    def _step_simulation(self, full_action):
        """Step simulation 4 times with position targets (same pattern as original)."""
        for _ in range(4):
            self.env.scene["robot"].set_joint_position_target(full_action) 
            self.env.scene.write_data_to_sim()                           
            self.env.sim.step(render=False)                              
            self.env.scene.update(dt=self.env.physics_dt)                    
        self.env.sim.render()
        self.env.observation_manager.compute()

    def _step_simulation_torque(self, torques_initial, hand_pos_action=None):
        """Step simulation 4 times, recomputing torques each sub-step.
        
        MuJoCo recomputes the impedance controller torque at every physics
        sub-step (200Hz).  We match that here: recalculate torques from the
        stored DDS targets and the *current* joint state every sub-step.
        
        Args:
            torques_initial: Effort targets for body joints (29 DOF via GR00T-WBC).
            hand_pos_action: Optional position targets for hand joints (Dex3 via xr_teleoperate).
                             Hand joints retain their PD gains, so position targets drive them.
        """
        effective_kd = self._dds_kd * self._kd_multiplier
        
        for sub_step in range(4):
            if sub_step == 0:
                # First sub-step: use the pre-computed torques
                self.env.scene["robot"].set_joint_effort_target(torques_initial)
            else:
                # Sub-steps 1-3: recompute torques using latest joint state
                robot = self.env.scene["robot"]
                cur_pos = robot.data.joint_pos[0]
                cur_vel = robot.data.joint_vel[0]
                
                sim_indices = self._motor_to_sim_idx
                q_cur = cur_pos[sim_indices]
                dq_cur = cur_vel[sim_indices]
                
                motor_torques = (self._dds_tau
                                 + self._dds_kp * (self._dds_q - q_cur)
                                 + effective_kd * (self._dds_dq - dq_cur)
                                 - self._extra_vel_damping * dq_cur)
                motor_torques.clamp_(-self._motor_effort_limit, self._motor_effort_limit)
                
                self._torque_buf.zero_()
                self._torque_buf[sim_indices] = motor_torques
                self.env.scene["robot"].set_joint_effort_target(self._torque_buf)
            
            # Set hand joint position targets (hand joints still have PD gains active)
            if hand_pos_action is not None:
                self.env.scene["robot"].set_joint_position_target(hand_pos_action)

            self.env.scene.write_data_to_sim()
            self.env.sim.step(render=False)
            self.env.scene.update(dt=self.env.physics_dt)
        self.env.sim.render()
        self.env.observation_manager.compute()
    
    def _convert_to_joint_range(self, value):
        """Convert gripper control value to joint angle"""
        input_min, input_max = 0.0, 5.6
        output_min, output_max = 0.03, -0.02
        value = max(input_min, min(input_max, value))
        return output_min + (output_max - output_min) * (value - input_min) / (input_max - input_min)
    
    def cleanup(self):
        """Clean up DDS resources"""
        # Close debug logger first
        self._close_debug_logger()
        try:
            if self.robot_dds:
                self.robot_dds.stop_communication()
            if self.gripper_dds:
                self.gripper_dds.stop_communication()
            if self.dex3_dds:
                self.dex3_dds.stop_communication()
            if self.inspire_dds:
                self.inspire_dds.stop_communication()
        except Exception as e:
            print(f"[{self.name}] Clean up DDS resources failed: {e}")