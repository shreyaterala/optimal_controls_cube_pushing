import mujoco
import mujoco.viewer
from tabletop_env.env import make_push_env

# Cube variant to load: "wood", "steel", "rubber" or "ice" (see CUBE_VARIANTS in env.py).
CUBE = "ice"

# Load the single-cube push scene with that cube.
envs = make_push_env(
    n_envs=1,
    variants=[CUBE],
)

# Access the SO-101 environment
vec_env = envs[f"so101_push_{CUBE}"][0]
robot_env = vec_env.envs[0].unwrapped

# Initialize the simulation and robot
robot_env.reset()

# Access the MuJoCo model and physics state
world = robot_env._ensure_world()
model = world.model
data = world.data

print("SO-101 simulation running!")
print("Bodies:", model.nbody)
print("Joints:", model.njnt)
print("Actuators:", model.nu)

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data, nstep=world.substeps)
            viewer.sync()
finally:
    vec_env.close()
