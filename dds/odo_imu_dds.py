# Copyright (c) 2025. Additional DDS publishers for GR00T-WBC integration.
# Publishes rt/odostate (OdoState_) and rt/secondary_imu (IMUState_)
"""
OdoImuDDS: Publishes floating-base odometry and torso IMU data via DDS.
Required by GR00T-WholeBodyControl's BodyStateProcessor.
"""

import numpy as np
from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import OdoState_, IMUState_
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__OdoState_,
    unitree_hg_msg_dds__IMUState_,
)


class OdoImuDDS(DDSObject):
    """DDS publisher for rt/odostate and rt/secondary_imu topics.
    
    These topics are required by GR00T-WholeBodyControl's BodyStateProcessor
    when running in sim mode (ENV_TYPE == "sim").
    """

    def __init__(self, node_name: str = "odo_imu"):
        if hasattr(self, '_initialized'):
            return

        super().__init__()
        self.node_name = node_name
        self.odo_state = unitree_hg_msg_dds__OdoState_()
        self.torso_imu_state = unitree_hg_msg_dds__IMUState_()
        self._initialized = True

        # Setup shared memory for receiving data from Isaac Lab observation
        self.setup_shared_memory(
            input_shm_name="isaac_odo_imu_state",
            input_size=2048,
            outputshm_flag=False,  # No output shared memory needed (publish only)
        )

        print(f"[{self.node_name}] OdoImu DDS node initialized")

    def setup_publisher(self) -> bool:
        """Setup publishers for rt/odostate and rt/secondary_imu"""
        try:
            self.odo_publisher = ChannelPublisher("rt/odostate", OdoState_)
            self.odo_publisher.Init()
            print(f"[{self.node_name}] OdoState publisher initialized (rt/odostate)")

            self.imu_publisher = ChannelPublisher("rt/secondary_imu", IMUState_)
            self.imu_publisher.Init()
            print(f"[{self.node_name}] IMUState publisher initialized (rt/secondary_imu)")
            return True
        except Exception as e:
            print(f"[{self.node_name}] Publisher initialization failed: {e}")
            return False

    def setup_subscriber(self) -> bool:
        """No subscriber needed (publish only)"""
        return True

    def dds_subscriber(self, msg: Any, datatype: str = None) -> None:
        """Not used - this is a publish-only object"""
        pass

    def dds_publisher(self) -> None:
        """Read data from shared memory and publish to DDS topics."""
        try:
            data = self.input_shm.read_data()
            if data is None:
                return

            # --- Publish OdoState ---
            root_pos = data.get("root_position")
            root_quat = data.get("root_orientation")  # Expected: [w, x, y, z]
            root_lin_vel = data.get("root_linear_velocity")
            root_ang_vel = data.get("root_angular_velocity")

            if root_pos is not None:
                self.odo_state.position[:] = np.asarray(root_pos, dtype=np.float32)[:3]
            if root_quat is not None:
                self.odo_state.orientation[:] = np.asarray(root_quat, dtype=np.float32)[:4]
            if root_lin_vel is not None:
                self.odo_state.linear_velocity[:] = np.asarray(root_lin_vel, dtype=np.float32)[:3]
            if root_ang_vel is not None:
                self.odo_state.angular_velocity[:] = np.asarray(root_ang_vel, dtype=np.float32)[:3]

            self.odo_state.tick += 1
            self.odo_publisher.Write(self.odo_state)

            # --- Publish Secondary IMU (torso) ---
            torso_quat = data.get("torso_quaternion")  # Expected: [w, x, y, z]
            torso_gyro = data.get("torso_gyroscope")

            if torso_quat is not None:
                self.torso_imu_state.quaternion[:] = np.asarray(torso_quat, dtype=np.float32)[:4]
            if torso_gyro is not None:
                self.torso_imu_state.gyroscope[:] = np.asarray(torso_gyro, dtype=np.float32)[:3]

            self.imu_publisher.Write(self.torso_imu_state)

        except Exception as e:
            print(f"[{self.node_name}] Error in dds_publisher: {e}")

    def write_odo_imu_state(self, root_state, torso_imu_data):
        """Write odometry and torso IMU data to shared memory.

        Args:
            root_state: [13] array = [pos(3), quat_wxyz(4), lin_vel(3), ang_vel(3)]
            torso_imu_data: [7] array = [quat_wxyz(4), gyro(3)]
        """
        if self.input_shm is None:
            return
        try:
            root_state_np = root_state.tolist() if hasattr(root_state, 'tolist') else root_state
            torso_np = torso_imu_data.tolist() if hasattr(torso_imu_data, 'tolist') else torso_imu_data

            state_data = {
                "root_position": root_state_np[:3],
                "root_orientation": root_state_np[3:7],       # [w, x, y, z]
                "root_linear_velocity": root_state_np[7:10],
                "root_angular_velocity": root_state_np[10:13],
                "torso_quaternion": torso_np[:4],              # [w, x, y, z]
                "torso_gyroscope": torso_np[4:7],
            }
            self.input_shm.write_data(state_data)
        except Exception as e:
            print(f"[{self.node_name}] Error writing odo_imu state: {e}")
