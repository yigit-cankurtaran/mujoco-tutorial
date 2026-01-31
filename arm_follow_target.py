import argparse
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curl the arm while applying an IK correction toward a movable target sphere. "
            "Arrow keys move the target (left/right = -/+x, up/down = +/-z)."
        )
    )
    parser.add_argument(
        "xml_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "basic_arm.xml"),
        help="Path to the MuJoCo XML file (default: ./basic_arm.xml).",
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
    parser.add_argument(
        "--site",
        type=str,
        default="end1",
        help="End-effector site to track (default: end1).",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Initial target position in meters (default: current site position).",
    )
    parser.add_argument(
        "--target-radius",
        type=float,
        default=0.02,
        help="Radius of the red target sphere (default: 0.02).",
    )
    parser.add_argument(
        "--move-step",
        type=float,
        default=0.02,
        help="Target move step in meters per key event (default: 0.02).",
    )
    parser.add_argument(
        "--move-speed",
        dest="move_step",
        type=float,
        default=None,
        help="Deprecated: use --move-step.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.1,
        help="Min seconds between printing target coordinates (default: 0.1).",
    )
    parser.add_argument(
        "--ik-gain",
        type=float,
        default=0.6,
        help="Scale applied to IK offset added to curl target (default: 0.6).",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=0.05,
        help="Damping term for IK (default: 0.05).",
    )
    parser.add_argument(
        "--ik-max-offset",
        type=float,
        default=0.6,
        help="Clamp for IK joint offsets in radians (default: 0.6).",
    )
    args = parser.parse_args()
    if args.move_step is None:
        args.move_step = 0.02
    return args


def require_id(model: mujoco.MjModel, obj: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj, name)
    if idx < 0:
        raise ValueError(f"Could not find {name} in model.")
    return idx


def damped_least_squares(jac: np.ndarray, err: np.ndarray, damping: float) -> np.ndarray:
    jjt = jac @ jac.T + (damping ** 2) * np.eye(3)
    return jac.T @ np.linalg.solve(jjt, err)


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        raise RuntimeError(
            "On macOS, MuJoCo's viewer requires running under `mjpython`.\n"
            "Try: mjpython arm_follow_target.py {}".format(args.xml_path)
        )

    hinge_joint_names = ["hinge_y1", "hinge_y2"]
    hinge_actuator_names = ["motor_hinge_y1", "motor_hinge_y2"]

    hinge_jids = [
        require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in hinge_joint_names
    ]
    hinge_aids = [
        require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in hinge_actuator_names
    ]

    dof_indices = [int(model.jnt_dofadr[jid]) for jid in hinge_jids]

    qpos_base = {}
    for name, jid in zip(hinge_joint_names, hinge_jids):
        qpos_adr = model.jnt_qposadr[jid]
        qpos_base[name] = float(data.qpos[qpos_adr])

    site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, args.site)
    mujoco.mj_forward(model, data)

    if args.target is None:
        target_pos = np.array(data.site_xpos[site_id], dtype=float)
    else:
        target_pos = np.array(args.target, dtype=float)

    print(
        f"Target position: x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, z={target_pos[2]:.3f}",
        flush=True,
    )

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    t0 = time.time()
    last_print_time = t0

    # GLFW key codes for arrows (MuJoCo viewer uses GLFW under the hood).
    key_left = 263
    key_right = 262
    key_up = 265
    key_down = 264

    def key_callback(keycode: int) -> None:
        nonlocal last_print_time

        delta = np.zeros(3)
        if keycode == key_left:
            delta[0] -= 1.0
        elif keycode == key_right:
            delta[0] += 1.0
        elif keycode == key_up:
            delta[2] += 1.0
        elif keycode == key_down:
            delta[2] -= 1.0
        else:
            return

        target_pos[:] = target_pos + delta * args.move_step

        now = time.time()
        if now - last_print_time >= args.print_interval:
            print(
                f"Target position: x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, z={target_pos[2]:.3f}",
                flush=True,
            )
            last_print_time = now

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        marker_mat = np.eye(3).flatten()

        while viewer.is_running():
            now = time.time()

            t = now - t0
            phase = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / args.period))
            target_offset = args.amplitude * phase

            with viewer.lock():
                mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                site_pos = data.site_xpos[site_id]
                err = target_pos - site_pos

                jac = jacp[:, dof_indices]
                dq = damped_least_squares(jac, err, args.ik_damping)
                dq = np.clip(dq, -args.ik_max_offset, args.ik_max_offset)

                data.ctrl[:] = 0.0
                for idx, name, jid, aid in zip(
                    range(len(hinge_joint_names)),
                    hinge_joint_names,
                    hinge_jids,
                    hinge_aids,
                ):
                    qpos_adr = model.jnt_qposadr[jid]
                    qvel_adr = model.jnt_dofadr[jid]
                    qpos = float(data.qpos[qpos_adr])
                    qvel = float(data.qvel[qvel_adr])

                    qpos_target = qpos_base[name] + target_offset + args.ik_gain * dq[idx]
                    torque = args.kp * (qpos_target - qpos) - args.kd * qvel

                    gear = float(model.actuator_gear[aid, 0])
                    data.ctrl[aid] = torque / gear if gear != 0.0 else 0.0

                mujoco.mj_step(model, data)

                viewer.user_scn.ngeom = 0
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[args.target_radius, 0.0, 0.0],
                    pos=target_pos,
                    mat=marker_mat,
                    rgba=[1.0, 0.0, 0.0, 1.0],
                )
                viewer.user_scn.ngeom = 1

            viewer.sync()
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
