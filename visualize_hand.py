import argparse
import os
import time

import mujoco
import mujoco.viewer

from simple_mujoco_env import SimpleMujocoEnv


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
    env = SimpleMujocoEnv(args.xml_path)
    model = env.model
    data = env.data
    env.require_mjpython("visualize_hand.py")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Frame the hand in view on launch.
        viewer.cam.lookat[:] = [0.12, 0.0, 0.9]
        viewer.cam.distance = 0.9
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        while viewer.is_running():
            env.step()
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
