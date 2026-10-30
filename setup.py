import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/robotstudio_so101/scene.xml")
data = mujoco.MjData(model)

print("joints:", [model.joint(i).name for i in range(model.njnt)])
print("actuators:", model.nu)

mujoco.viewer.launch(model, data)