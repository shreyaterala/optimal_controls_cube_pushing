import mujoco
import mujoco.viewer
from tabletop_env.env import make_env

# Load the project-local SO-101 tabletop environment.
envs = make_env(
    n_envs=1,
)

# Access the SO-101 environment
vec_env = envs["so101_tabletop"][0]
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
