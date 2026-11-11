import mujoco
import numpy as np


class ResolvedRateController:
    """Convert task-space force commands into joint-position targets through a Jacobian."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names: tuple[str, ...],
        site_name: str = "gripperframe",
        admittance: float = 0.03,
        damping: float = 0.01,
        position_gain: float = 4.0,
        nullspace_gain: float = 0.0,
        max_cartesian_speed: float = 0.08,
        max_joint_speed: float = 1.5,
        orientation_gain: float = 4.0,
        force_gain: float | None = None,
    ):
        if data.model is not model:
            raise ValueError("data must belong to model.")
        if len(joint_names) != 5:
            raise ValueError("Resolved-rate control requires the arm's five joints.")
        selected_force_gain = admittance if force_gain is None else force_gain
        if not np.all(
            np.isfinite(
                [
                    selected_force_gain,
                    damping,
                    position_gain,
                    nullspace_gain,
                    orientation_gain,
                    max_cartesian_speed,
                    max_joint_speed,
                ]
            )
        ):
            raise ValueError("Controller gains and velocity limits must be finite.")
        if (
            selected_force_gain < 0
            or damping <= 0
            or position_gain < 0
            or nullspace_gain < 0
            or orientation_gain < 0
        ):
            raise ValueError("Controller gains must be non-negative and damping positive.")
        if max_cartesian_speed <= 0 or max_joint_speed <= 0:
            raise ValueError("Velocity limits must be positive.")

        self.model = model
        self.data = data
        self.joint_names = joint_names
        self.joints = [model.joint(name) for name in joint_names]
        self.qpos_ids = np.array([joint.qposadr[0] for joint in self.joints], dtype=int)
        self.dof_ids = np.array([joint.dofadr[0] for joint in self.joints], dtype=int)
        self.joint_limits = np.array([joint.range for joint in self.joints])
        self.actuator_ids = np.empty(len(self.joints), dtype=int)
        for index, joint in enumerate(self.joints):
            actuator_id = next(
                (
                    actuator_id
                    for actuator_id, joint_id in enumerate(model.actuator_trnid[:, 0])
                    if joint_id == joint.id
                ),
                None,
            )
            if actuator_id is None:
                raise ValueError(f"No joint actuator found for {joint.name}.")
            self.actuator_ids[index] = actuator_id

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in the MJCF model.")

        self.force_gain = selected_force_gain
        self.admittance = selected_force_gain
        self.damping = damping
        self.position_gain = position_gain
        self.orientation_gain = orientation_gain
        self.nullspace_gain = nullspace_gain
        self.max_cartesian_speed = max_cartesian_speed
        self.max_joint_speed = max_joint_speed
        self.commanded_q = data.qpos[self.qpos_ids].copy()
        self.jacobian = np.zeros((3, model.nv))
        self.jacobian_rotation = np.zeros((3, model.nv))
        self.task_jacobian = np.zeros((5, len(self.joints)))

    def get_object_state(self, object_body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the body's world position and linear-then-angular velocity."""
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, object_body_name
        )
        if body_id < 0:
            raise ValueError(f"Body '{object_body_name}' not found in the MuJoCo model.")
        position = self.data.xpos[body_id].copy()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
        )
        return position, np.concatenate((velocity[3:], velocity[:3]))

    def reset(self) -> None:
        """Synchronize joint targets to the current simulated pose."""
        self.commanded_q = self.data.qpos[self.qpos_ids].copy()

    def step(
        self,
        task_force: np.ndarray,
        dt: float,
        position_reference: np.ndarray | None = None,
        motion_direction: np.ndarray | None = None,
        position_velocity_reference: np.ndarray | None = None,
        track_position_reference: bool = False,
        measured_force: np.ndarray | None = None,
        predicted_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update targets using force-error velocity control and 5D resolved rate.

        ``task_force`` is the desired force. ``measured_force`` defaults to zero
        when no force estimate is available. The translational command is
        ``v_pred + force_gain * (F_des - F_meas) + position_gain * position_error``.
        ``motion_direction`` also specifies the desired gripper-pad normal.
        """
        force = np.asarray(task_force, dtype=float)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise ValueError("task_force must be a finite 3D vector.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if measured_force is None:
            force_error = force
        else:
            force_measurement = np.asarray(measured_force, dtype=float)
            if force_measurement.shape != (3,) or not np.all(
                np.isfinite(force_measurement)
            ):
                raise ValueError("measured_force must be a finite 3D vector.")
            force_error = force - force_measurement
        reference = None
        if position_reference is not None:
            reference = np.asarray(position_reference, dtype=float)
            if reference.shape != (3,) or not np.all(np.isfinite(reference)):
                raise ValueError("position_reference must be a finite 3D vector.")

        if motion_direction is None:
            motion_projector = np.eye(3)
            hold_projector = np.zeros((3, 3))
            reference_projector = (
                np.eye(3) if track_position_reference else np.zeros((3, 3))
            )
        else:
            if reference is None:
                raise ValueError(
                    "position_reference is required when motion_direction is set."
                )
            direction = np.asarray(motion_direction, dtype=float)
            if direction.shape != (3,) or not np.all(np.isfinite(direction)):
                raise ValueError("motion_direction must be a finite 3D vector.")
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0:
                raise ValueError("motion_direction must be non-zero.")
            direction = direction / direction_norm
            motion_projector = np.outer(direction, direction)
            hold_projector = np.eye(3) - motion_projector
            reference_projector = (
                np.eye(3) if track_position_reference else hold_projector
            )

        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[self.dof_ids] = self.data.qfrc_bias[self.dof_ids]
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self.jacobian,
            self.jacobian_rotation,
            self.site_id,
        )
        arm_jacobian = self.jacobian[:, self.dof_ids]
        current_position = self.data.site_xpos[self.site_id]
        position_error = np.zeros(3)
        if position_reference is not None:
            position_error = reference_projector @ (reference - current_position)
        if position_velocity_reference is not None and predicted_velocity is not None:
            raise ValueError(
                "Provide only one of position_velocity_reference and predicted_velocity."
            )
        velocity_input = (
            predicted_velocity
            if predicted_velocity is not None
            else position_velocity_reference
        )
        velocity_prediction = np.zeros(3)
        if velocity_input is not None:
            velocity_prediction = np.asarray(velocity_input, dtype=float)
            if velocity_prediction.shape != (3,) or not np.all(
                np.isfinite(velocity_prediction)
            ):
                raise ValueError("predicted_velocity must be a finite 3D vector.")
        desired_velocity = (
            velocity_prediction
            + self.force_gain * (motion_projector @ force_error)
            + self.position_gain * position_error
        )
        speed = np.linalg.norm(desired_velocity)
        if speed > self.max_cartesian_speed:
            desired_velocity *= self.max_cartesian_speed / speed

        task_jacobian = self.task_jacobian
        task_jacobian[:3] = arm_jacobian

        if motion_direction is None:
            desired_normal = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2]
        else:
            desired_normal = direction
        desired_normal = desired_normal / np.linalg.norm(desired_normal)
        current_normal = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2]
        tangent_seed = np.eye(3)[np.argmin(np.abs(desired_normal))]
        tangent_one = np.cross(desired_normal, tangent_seed)
        tangent_one /= np.linalg.norm(tangent_one)
        tangent_two = np.cross(desired_normal, tangent_one)
        tangent_basis = np.column_stack((tangent_one, tangent_two))
        task_jacobian[3:] = tangent_basis.T @ self.jacobian_rotation[:, self.dof_ids]
        normal_cross = np.cross(current_normal, desired_normal)
        cross_norm = np.linalg.norm(normal_cross)
        normal_angle = np.arctan2(
            cross_norm, np.clip(np.dot(current_normal, desired_normal), -1.0, 1.0)
        )
        if cross_norm > 1e-8:
            orientation_error = (normal_angle / cross_norm) * normal_cross
        elif np.dot(current_normal, desired_normal) < 0:
            orientation_error = normal_angle * tangent_one
        else:
            orientation_error = np.zeros(3)
        orientation_velocity = self.orientation_gain * orientation_error
        task_velocity = np.concatenate(
            (desired_velocity, tangent_basis.T @ orientation_velocity)
        )

        regularized = task_jacobian @ task_jacobian.T + self.damping**2 * np.eye(5)
        jacobian_inverse = task_jacobian.T @ np.linalg.solve(regularized, np.eye(5))

        joint_center = self.joint_limits.mean(axis=1)
        joint_half_range = (self.joint_limits[:, 1] - self.joint_limits[:, 0]) / 2
        joint_gradient = (joint_center - self.data.qpos[self.qpos_ids]) / np.maximum(
            joint_half_range**2, 1e-8
        )
        nullspace_velocity = self.nullspace_gain * (
            np.eye(len(self.joints)) - jacobian_inverse @ task_jacobian
        ) @ joint_gradient
        joint_velocity = jacobian_inverse @ task_velocity + nullspace_velocity
        joint_velocity = np.clip(
            joint_velocity, -self.max_joint_speed, self.max_joint_speed
        )

        self.commanded_q = np.clip(
            self.commanded_q + joint_velocity * dt,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )
        self.data.ctrl[self.actuator_ids] = self.commanded_q
        return self.commanded_q.copy()
