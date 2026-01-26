import argparse
import math
import os
import time

import mujoco
import mujoco.viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make the arm curl up and down using joint-space PD control."
    )
    parser.add_argument(
        "xml_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "example_with_actuators.xml"),
        help="Path to the MuJoCo XML file (default: ./example_with_actuators.xml).",
    )
    parser.add_argument("--kp", type=float, default=25.0, help="Proportional gain.")
    parser.add_argument("--kd", type=float, default=3.0, help="Derivative gain.")
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.8,
        help="Curl amplitude in radians (default: 0.8).",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=4.0,
        help="Seconds per full down-up cycle (default: 4.0).",
    )
    parser.add_argument("--sleep", type=float, default=0.01, help="Loop sleep (s).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    # Drive the two hinge joints that visually curl the arm.
    hinge_joint_names = ["hinge_y1", "hinge_y2"]
    hinge_actuator_names = ["motor_hinge_y1", "motor_hinge_y2"]

    hinge_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge_joint_names]
    hinge_aids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in hinge_actuator_names
    ]

    # Store baseline joint angles from the initial state.
    qpos_base = {}
    for name, jid in zip(hinge_joint_names, hinge_jids):
        qpos_adr = model.jnt_qposadr[jid]
        qpos_base[name] = float(data.qpos[qpos_adr])

    t0 = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            # Smooth oscillation between down (0) and up (+amplitude) and back.
            phase = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / args.period))
            target_offset = args.amplitude * phase

            data.ctrl[:] = 0.0
            for name, jid, aid in zip(hinge_joint_names, hinge_jids, hinge_aids):
                qpos_adr = model.jnt_qposadr[jid]
                qvel_adr = model.jnt_dofadr[jid]
                qpos = float(data.qpos[qpos_adr])
                qvel = float(data.qvel[qvel_adr])

                qpos_target = qpos_base[name] + target_offset
                torque = args.kp * (qpos_target - qpos) - args.kd * qvel

                gear = float(model.actuator_gear[aid, 0])
                data.ctrl[aid] = torque / gear if gear != 0.0 else 0.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
