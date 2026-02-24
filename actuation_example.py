import os
import sys
import time

import mujoco
import mujoco.viewer


def main() -> None:
    xml_path = os.path.join(os.path.dirname(__file__), "actuation_robot.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor")
    elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_motor")

    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        raise RuntimeError(
            "On macOS, MuJoCo's viewer requires running under `mjpython`.\n"
            "Try: mjpython actuation_example.py"
        )

    print("Actuators:", model.nu, "DOFs:", model.nv)
    print("Actuated joints: shoulder, elbow. Passive flex joints: flex1, flex2.")

    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            data.ctrl[shoulder_id] = 0.7 * (1.0 if int(t) % 4 < 2 else -1.0)
            data.ctrl[elbow_id] = 0.6 * (1.0 if int(t + 1) % 4 < 2 else -1.0)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
