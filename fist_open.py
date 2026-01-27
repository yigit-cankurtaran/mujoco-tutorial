import argparse
import math
import os
import time

import mujoco
import mujoco.viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curl the hand into a fist and open back up using joint-space PD control."
    )
    parser.add_argument(
        "xml_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "robot_hand.xml"),
        help="Path to the MuJoCo XML file (default: ./robot_hand.xml).",
    )
    parser.add_argument("--kp", type=float, default=12.0, help="Proportional gain.")
    parser.add_argument("--kd", type=float, default=1.2, help="Derivative gain.")
    parser.add_argument(
        "--period",
        type=float,
        default=4.0,
        help="Seconds per full open-close-open cycle (default: 4.0).",
    )
    parser.add_argument(
        "--range_scale",
        type=float,
        default=1.0,
        help="Fraction of each joint's max range to use for the fist.",
    )
    parser.add_argument("--sleep", type=float, default=0.01, help="Loop sleep (s).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    # Drive all finger joints (MCP/PIP/DIP) to make a fist, then open.
    joint_names = [
        "index_mcp",
        "index_pip",
        "index_dip",
        "middle_mcp",
        "middle_pip",
        "middle_dip",
        "ring_mcp",
        "ring_pip",
        "ring_dip",
        "pinky_mcp",
        "pinky_pip",
        "pinky_dip",
        "thumb_mcp",
        "thumb_pip",
        "thumb_dip",
    ]
    actuator_names = [f"motor_{name}" for name in joint_names]

    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    aids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in actuator_names
    ]

    # Store baseline and max range targets.
    qpos_base = {}
    qpos_fist = {}
    for name, jid in zip(joint_names, jids):
        qpos_adr = model.jnt_qposadr[jid]
        qpos_base[name] = float(data.qpos[qpos_adr])
        jnt_range = model.jnt_range[jid]
        qpos_fist[name] = float(jnt_range[1]) * args.range_scale

    t0 = time.time()

    if os.environ.get("MJPYTHON_BIN") is None and os.sys.platform == "darwin":
        raise RuntimeError(
            "On macOS, MuJoCo's viewer requires running under `mjpython`.\n"
            f"Try: mjpython fist_open.py {args.xml_path}"
        )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Frame the hand in view on launch.
        viewer.cam.lookat[:] = [0.12, 0.0, 0.9]
        viewer.cam.distance = 0.9
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        while viewer.is_running():
            t = time.time() - t0
            # 0 -> 1 -> 0 smooth cycle (open -> fist -> open).
            phase = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / args.period))

            data.ctrl[:] = 0.0
            for name, jid, aid in zip(joint_names, jids, aids):
                qpos_adr = model.jnt_qposadr[jid]
                qvel_adr = model.jnt_dofadr[jid]
                qpos = float(data.qpos[qpos_adr])
                qvel = float(data.qvel[qvel_adr])

                qpos_target = qpos_base[name] + phase * (qpos_fist[name] - qpos_base[name])
                torque = args.kp * (qpos_target - qpos) - args.kd * qvel

                gear = float(model.actuator_gear[aid, 0])
                data.ctrl[aid] = torque / gear if gear != 0.0 else 0.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
