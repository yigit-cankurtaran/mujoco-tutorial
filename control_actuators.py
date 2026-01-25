import argparse
import os
import time

import mujoco
import mujoco.viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PD-control hinge actuators to stabilize the arm."
    )
    parser.add_argument(
        "xml_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "example_with_actuators.xml"),
        help="Path to the MuJoCo XML file (default: ./example_with_actuators.xml).",
    )
    parser.add_argument("--kp", type=float, default=20.0, help="Proportional gain.")
    parser.add_argument("--kd", type=float, default=2.0, help="Derivative gain.")
    parser.add_argument("--sleep", type=float, default=0.01, help="Loop sleep (s).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    # Control only hinge motors to hold their initial angles.
    hinge_joint_names = ["hinge_y1", "hinge_x1", "hinge_y2", "hinge_z2"]
    hinge_actuator_names = [
        "motor_hinge_y1",
        "motor_hinge_x1",
        "motor_hinge_y2",
        "motor_hinge_z2",
    ]

    hinge_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge_joint_names]
    hinge_aids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in hinge_actuator_names
    ]

    # Store target positions from the initial state.
    qpos_target = {}
    for name, jid in zip(hinge_joint_names, hinge_jids):
        qpos_adr = model.jnt_qposadr[jid]
        qpos_target[name] = float(data.qpos[qpos_adr])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # PD control on hinge joints.
            data.ctrl[:] = 0.0
            for name, jid, aid in zip(hinge_joint_names, hinge_jids, hinge_aids):
                qpos_adr = model.jnt_qposadr[jid]
                qvel_adr = model.jnt_dofadr[jid]
                qpos = float(data.qpos[qpos_adr])
                qvel = float(data.qvel[qvel_adr])
                torque = args.kp * (qpos_target[name] - qpos) - args.kd * qvel

                # Convert torque to control signal using actuator gear.
                gear = float(model.actuator_gear[aid, 0])
                data.ctrl[aid] = torque / gear if gear != 0.0 else 0.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
