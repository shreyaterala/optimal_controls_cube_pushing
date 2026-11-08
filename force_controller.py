import mujoco
import numpy as np

class SO101ForceController:
    def __init__(
        self,
        model: str | mujoco.MjModel,
        data: mujoco.MjData | str | None = None,
        contact_site_name: str = "gripperframe",
    ):
        """
        Interfaces a high-level force-based trajectory planner with a MuJoCo SO-101 simulation.
        """
        if isinstance(model, mujoco.MjModel):
            if not isinstance(data, mujoco.MjData):
                raise ValueError("data must be provided when model is an MjModel.")
            self.model = model
            self.data = data
        else:
            if isinstance(data, str):
                if contact_site_name != "gripperframe":
                    raise ValueError("contact_site_name was provided more than once.")
                contact_site_name = data
                data = None
            self.model = mujoco.MjModel.from_xml_path(model)
            self.data = data if data is not None else mujoco.MjData(self.model)
            if not isinstance(self.data, mujoco.MjData) or self.data.model is not self.model:
                raise ValueError("data must belong to the loaded MuJoCo model.")
        
        # Resolve actuator joints explicitly: free objects may precede the arm in qvel.
        self.nv = self.model.nv  # Degrees of freedom
        actuator_joint_ids = self.model.actuator_trnid[:, 0]
        self.arm_dof_ids = []
        self.arm_actuator_ids = []
        for actuator_id, joint_id in enumerate(actuator_joint_ids):
            if joint_id < 0:
                raise ValueError("SO101ForceController requires joint-transmission actuators.")
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)
            )
            if joint_name == "gripper":
                continue
            self.arm_dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
            self.arm_actuator_ids.append(actuator_id)
        self.arm_dof_ids = np.asarray(self.arm_dof_ids, dtype=int)
        self.arm_actuator_ids = np.asarray(self.arm_actuator_ids, dtype=int)
        if self.arm_dof_ids.size == 0:
            raise ValueError("No arm joints were found in the model actuators.")

        # Position servos would resist the Jacobian-transpose torque command.
        self.model.actuator_gainprm[self.arm_actuator_ids] = 0.0
        self.model.actuator_biasprm[self.arm_actuator_ids] = 0.0
        
        # Find the end-effector or contact site ID
        self.site_name = contact_site_name
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, contact_site_name)
        if self.site_id == -1:
            raise ValueError(f"Site '{contact_site_name}' not found in the MJCF model.")

    def get_object_state(self, object_body_name: str):
        """
        Helper to return (x_t, x_dot_t) of the pushed object for your high-level controller.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
        # 3D position and orientation quaternion
        x = self.data.xpos[body_id].copy() 
        # Linear and angular velocity
        x_dot = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, x_dot, 0)
        return x, x_dot

    def apply_force_command(self, F_t: np.ndarray, frame: str = "world", object_body_name: str = None):
        """
        Apply a 6D force/torque command as joint torques using the Jacobian transpose.
        
        Parameters:
            F_t: np.ndarray of shape (6,) -> [Fx, Fy, Fz, Tx, Ty, Tz]
            frame: "world" or "local" (tells the controller how to interpret F_t)
            object_body_name: Optional, required only if frame is "local"
        """
        # 1. Synchronize and update kinematics/Jacobians based on current state
        mujoco.mj_forward(self.model, self.data)
        
        # 2. Transform force vector to world frame if it's provided in a local frame
        if frame == "local":
            if not object_body_name:
                raise ValueError("object_body_name must be provided for local frame forces.")
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
            # Get 3x3 rotation matrix for the body
            R = self.data.xmat[body_id].reshape(3, 3)
            
            # Rotate both the force (linear) and torque (angular) components
            F_world = np.zeros(6)
            F_world[:3] = R @ F_t[:3]
            F_world[3:] = R @ F_t[3:]
        else:
            F_world = F_t

        # 3. Compute the 6D Geometric Jacobian at the contact point
        jac_pos = np.zeros((3, self.nv))
        jac_rot = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.site_id)
        J = np.vstack([jac_pos, jac_rot])
        
        # Isolate only the actuated arm joints, regardless of model joint ordering.
        J_arm = J[:, self.arm_dof_ids]
        
        # 4. Map task-space force to joint torques: tau = J^T * F
        tau_force = J_arm.T @ F_world
        
        # 5. Extract passive forces (Gravity + Coriolis terms)
        tau_passive = self.data.qfrc_bias[self.arm_dof_ids]
        
        # 6. Sum them for total feedforward command
        tau_total = tau_force + tau_passive
        
        # 7. Apply directly to the simulation
        # Note: If using Method A (Actuators), write to self.data.ctrl instead
        self.data.qfrc_applied[self.arm_dof_ids] = tau_total

    def execute_force_command(self, F_t: np.ndarray, frame: str = "world", object_body_name: str = None):
        """Apply a force command and advance the simulation by one physics step."""
        self.apply_force_command(F_t, frame, object_body_name)
        mujoco.mj_step(self.model, self.data)
        return self.data.time
