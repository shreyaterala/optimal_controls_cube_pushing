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
