import mujoco
import mujoco.viewer
import numpy as np
from tabletop_env.env import make_push_env
from force_controller import SO101ForceController


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


def main():
    CUBE = "ice"
    envs = make_push_env(n_envs=1, variants=[CUBE])
    vec_env = envs[f"so101_push_{CUBE}"][0]

    try:
        robot_env = vec_env.envs[0].unwrapped
        robot_env.reset()

        world = robot_env._ensure_world()
        model = world.model
        data = world.data
        push_direction = _start_in_contact(world)
        controller = SO101ForceController(model, data, contact_site_name="gripperframe")

        initial_cube_pos, _ = controller.get_object_state("cube")
        desired_distance = 0.20
        goal_pos = initial_cube_pos + desired_distance * push_direction

        print(f"Initial Cube Position (XYZ): {initial_cube_pos}")
        print(f"Target Cube Position (XYZ):  {goal_pos}")
        print(f"Push Direction (XYZ):        {push_direction}")

        kp_pos = 8.0
        kd_vel = 0.5
        max_force = 1.0

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                cube_pos, cube_vel = controller.get_object_state("cube")
                pos_error = goal_pos - cube_pos
                vel_error = -cube_vel[:3]
                force_linear = (kp_pos * pos_error) + (kd_vel * vel_error)

                pushed_distance = np.dot(cube_pos - initial_cube_pos, push_direction)
                if pushed_distance >= desired_distance:
                    force_linear = np.zeros(3)

                force_magnitude = np.linalg.norm(force_linear)
                if force_magnitude > max_force:
                    force_linear = (force_linear / force_magnitude) * max_force

                force_command = np.zeros(6)
                force_command[:3] = force_linear
                controller.apply_force_command(force_command)

                for _ in range(world.substeps):
                    mujoco.mj_step(model, data)

                viewer.sync()
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
