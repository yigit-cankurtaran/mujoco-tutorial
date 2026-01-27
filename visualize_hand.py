import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the robot hand model.")
    parser.add_argument(
        "xml_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "robot_hand.xml"),
        help="Path to the MuJoCo XML file (default: ./robot_hand.xml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = args.xml_path
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        raise RuntimeError(
            "On macOS, MuJoCo's viewer requires running under `mjpython`.\n"
            f"Try: mjpython visualize_hand.py {xml_path}"
        )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
