import threading

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from tabletop_env.env import make_push_env
from force_controller import ResolvedRateController

DESIRED_DISTANCE = 0.20


def _wait_for_space(viewer, data, previous_time, continue_event, stage):
    continue_event.clear()
    viewer.set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            f"{stage} reached",
            "Press Space to continue",
        )
    )
    print(f"{stage} reached. Press Space in the viewer to continue.")

    while viewer.is_running():
        if data.time < previous_time:
            viewer.clear_texts()
            return "reset"
        if continue_event.wait(0.05):
            viewer.clear_texts()
            return "continue"

    viewer.clear_texts()
    return "closed"


def _move_to_home(world, home_joints):
    model, data = world.model, world.data
    home_target = world.to_ctrl(np.asarray(home_joints, dtype=float))
    data.ctrl[world.actuators] = home_target
    home_joint_ids = model.actuator_trnid[world.actuators, 0]
    home_dof_ids = model.jnt_dofadr[home_joint_ids]
    control_dt = model.opt.timestep * world.substeps
    max_steps = int(np.ceil(5.0 / control_dt))
    position_tolerance = np.deg2rad(1.0)
    velocity_tolerance = 0.02

    for _ in range(max_steps):
        mujoco.mj_step(model, data, nstep=world.substeps)
        position_error = np.max(np.abs(data.qpos[world.qadr] - home_target))
        max_joint_speed = np.max(np.abs(data.qvel[home_dof_ids]))
        if (
            position_error <= position_tolerance
            and max_joint_speed <= velocity_tolerance
        ):
            return

    raise TimeoutError(
        "Robot did not settle at home within 5 seconds "
        f"(position error {position_error:.4f} rad, "
        f"max joint speed {max_joint_speed:.4f} rad/s)."
    )


def _reset_to_home(robot_env):
    robot_env.reset()
    world = robot_env._ensure_world()
    _move_to_home(world, robot_env.initial_joints)
    return world


def _start_in_contact(world):
    model, data = world.model, world.data
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    if min(site_id, cube_id, cube_geom_id) < 0:
        raise ValueError("The push scene must define gripperframe, cube, and cube_geom.")

    joint_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    joints = [model.joint(name) for name in joint_names]
    qpos_ids = np.array([joint.qposadr[0] for joint in joints])
    dof_ids = np.array([joint.dofadr[0] for joint in joints])
    limits = np.array([joint.range for joint in joints])

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_id < 0:
        raise ValueError("The push scene must define the robot base body.")

    cube_rotation = data.geom_xmat[cube_geom_id].reshape(3, 3)
    cube_center = data.geom_xpos[cube_geom_id]
    arm_direction = cube_rotation.T @ (data.xpos[base_id] - cube_center)
    axis = int(np.argmax(np.abs(arm_direction[:2])))
    normal_local = np.zeros(3)
    normal_local[axis] = np.sign(arm_direction[axis]) or 1.0
    half_size = model.geom_size[cube_geom_id]

    initial_q = data.qpos[qpos_ids].copy()
    jacobian = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    established_contact = False

    # Start at the face center and move the target inward only if the gripper
    # frame's offset from the jaw prevents physical contact at that point.
    for inset in (0.002, 0.005, 0.008, 0.011, 0.014, 0.015):
        target = cube_center + cube_rotation @ (
            normal_local * (half_size[axis] - inset)
        )
        q = initial_q.copy()
        for _ in range(100):
            data.qpos[qpos_ids] = q
            mujoco.mj_forward(model, data)
            error = target - data.site_xpos[site_id]
            if np.linalg.norm(error) < 0.0005:
                break
            mujoco.mj_jacSite(model, data, jacobian, jacobian_rotation, site_id)
            arm_jacobian = jacobian[:, dof_ids]
            damping = 0.001 * np.eye(3)
            q += arm_jacobian.T @ np.linalg.solve(
                arm_jacobian @ arm_jacobian.T + damping, error
            )
            q = np.clip(q, limits[:, 0], limits[:, 1])

        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        if np.linalg.norm(target - data.site_xpos[site_id]) >= 0.001:
            continue

        established_contact = any(
            cube_geom_id in (contact.geom1, contact.geom2)
            and model.geom_bodyid[
                contact.geom2 if contact.geom1 == cube_geom_id else contact.geom1
            ]
            not in (0, cube_id)
            for contact in data.contact[: data.ncon]
        )
        if established_contact:
            break

    if not established_contact:
        raise RuntimeError("The gripper reached the cube but did not make physical contact.")
    data.ctrl[world.actuators] = data.qpos[world.qadr]
    mujoco.mj_forward(model, data)
    push_direction = -cube_rotation @ normal_local
    push_direction /= np.linalg.norm(push_direction)
    return push_direction


def _measure_push_force(world, cube_geom_id, push_direction, base_id):
    """Estimate the robot-on-cube force magnitude from MuJoCo contact wrenches."""
    model, data = world.model, world.data
    push_force_magnitude = 0.0
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if cube_geom_id not in (contact.geom1, contact.geom2):
            continue
        other_geom = contact.geom2 if contact.geom1 == cube_geom_id else contact.geom1
        body_id = int(model.geom_bodyid[other_geom])
        while body_id > 0 and body_id != base_id:
            body_id = int(model.body_parentid[body_id])
        if body_id != base_id:
            continue

        contact_wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, contact_wrench)
        contact_frame = contact.frame.reshape(3, 3)
        force_world = contact_frame.T @ contact_wrench[:3]
        push_force_magnitude += abs(float(np.dot(force_world, push_direction)))

    return push_force_magnitude * push_direction


def _start_push(world):
    model, data = world.model, world.data
    push_direction = _start_in_contact(world)
    controller = ResolvedRateController(
        model,
        data,
        joint_names=(
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ),
        site_name="gripperframe",
        force_gain=0.03,
        damping=0.01,
        position_gain=4.0,
        nullspace_gain=0.0,
        max_cartesian_speed=0.08,
        max_joint_speed=1.5,
    )
    initial_cube_pos, _ = controller.get_object_state("cube")
    initial_site_pos = data.site_xpos[controller.site_id].copy()
    contact_offset = initial_site_pos - initial_cube_pos
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    goal_pos = initial_cube_pos + DESIRED_DISTANCE * push_direction

    print(f"Initial Cube Position (XYZ): {initial_cube_pos}")
    print(f"Target Cube Position (XYZ):  {goal_pos}")
    print(f"Push Direction (XYZ):        {push_direction}")

    return (
        controller,
        initial_cube_pos,
        push_direction,
        contact_offset,
        cube_geom_id,
        base_id,
    )


def main():
    CUBE = "steel"
    envs = make_push_env(n_envs=1, variants=[CUBE])
    vec_env = envs[f"so101_push_{CUBE}"][0]

    try:
        robot_env = vec_env.envs[0].unwrapped
        world = robot_env._ensure_world()
        model = world.model
        data = world.data
        kp_pos = 2.0
        kd_vel = 0.1
        max_force = 0.1
        control_dt = model.opt.timestep * world.substeps
        continue_event = threading.Event()

        def key_callback(key):
            if key == glfw.KEY_SPACE:
                continue_event.set()

        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            while viewer.is_running():
                with viewer.lock():
                    world = _reset_to_home(robot_env)
                stage_time = data.time
                viewer.sync()
                stage = _wait_for_space(
                    viewer, data, stage_time, continue_event, "Home"
                )
                if stage == "closed":
                    break
                if stage == "reset":
                    print("Reset detected; returning to the home pose.")
                    continue

                with viewer.lock():
                    (
                        controller,
                        initial_cube_pos,
                        push_direction,
                        contact_offset,
                        cube_geom_id,
                        base_id,
                    ) = _start_push(world)
                stage_time = data.time
                viewer.sync()
                stage = _wait_for_space(
                    viewer, data, stage_time, continue_event, "Initial contact"
                )
                if stage == "closed":
                    break
                if stage == "reset":
                    print("Reset detected; returning to the home pose.")
                    continue

                previous_time = data.time
                while viewer.is_running():
                    if data.time < previous_time:
                        print("Reset detected; returning to the home pose.")
                        break

                    cube_pos, cube_vel = controller.get_object_state("cube")
                    progress = float(
                        np.dot(cube_pos - initial_cube_pos, push_direction)
                    )
                    push_speed = float(np.dot(cube_vel[:3], push_direction))
                    force_magnitude = max(
                        0.0,
                        kp_pos * (DESIRED_DISTANCE - progress) - kd_vel * push_speed,
                    )
                    force_magnitude = min(force_magnitude, max_force)
                    force_linear = force_magnitude * push_direction
                    contact_target = cube_pos + contact_offset
                    measured_force = _measure_push_force(
                        world, cube_geom_id, push_direction, base_id
                    )

                    controller.step(
                        force_linear,
                        dt=control_dt,
                        position_reference=contact_target,
                        motion_direction=push_direction,
                        predicted_velocity=cube_vel[:3],
                        track_position_reference=True,
                        measured_force=measured_force,
                    )

                    for _ in range(world.substeps):
                        mujoco.mj_step(model, data)

                    previous_time = data.time
                    viewer.sync()
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
