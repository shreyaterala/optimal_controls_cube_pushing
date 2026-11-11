# optimal_controls_cube_pushing
Designing an optimal controller for a cube pushing task using the SO-101 arm in MuJoCo.

## Run

Install the runtime dependencies in the selected Python environment:

```powershell
python.exe -m pip install mujoco gymnasium numpy
python.exe setup.py
```

The shared tabletop environment lives in `tabletop_env/`. It includes `env.py`, the MuJoCo
scene, and the SO-101 robot assets, so launching does not depend on a Hugging Face cache.

## Force-to-joint control

`ResolvedRateController` in `force_controller.py` computes the translational command as
`v_pred + k_F * (F_des - F_meas) + k_p * position_error`. `force_gain` configures `k_F`
(`admittance` remains a compatibility alias). `step` accepts the desired force
as its first argument, an optional measured-force estimate, and `predicted_velocity` (or
the compatibility name `position_velocity_reference`). The demo estimates `F_meas` from
robot-cube MuJoCo contact wrenches.

The controller maps the 3D velocity command and two pad-normal alignment constraints to
the arm's five joint velocities using a damped least-squares solve of the 5-by-5 task
Jacobian built from the MuJoCo site Jacobian. `motion_direction` specifies both the
force-driven translation axis and the desired pad normal. Joint-speed and joint-position
limits are applied before writing the position-actuator targets.

`testing.py` currently supplies a proportional-derivative force command as a placeholder;
the planned LQR can supply its 3D force output to the same low-level controller.
The demo pauses at the home pose and again at initial cube contact; press Space in the
MuJoCo viewer to continue each stage. The viewer Reset button restarts the sequence.
