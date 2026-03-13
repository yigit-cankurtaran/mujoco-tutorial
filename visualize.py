import argparse
import os
import time

import mujoco
import mujoco.viewer

from simple_mujoco_env import SimpleMujocoEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a MuJoCo XML model.")
    parser.add_argument(
        "xml_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "hello.xml"),
        help="Path to the MuJoCo XML file (default: ./hello.xml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = SimpleMujocoEnv(args.xml_path)
    model = env.model
    data = env.data
    env.require_mjpython("visualize.py")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            env.step()
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
