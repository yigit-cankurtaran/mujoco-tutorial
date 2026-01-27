import argparse
import math
import os
import sys
import time

import numpy as np

import mujoco
import mujoco.viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fly the quadcopter around a smooth looping path using a simple controller."
    )
    parser.add_argument(
        "xml_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "quadcopter.xml"),
        help="Path to the MuJoCo XML file (default: ./quadcopter.xml).",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="Orbit radius (m).")
    parser.add_argument("--period", type=float, default=10.0, help="Seconds per loop.")
    parser.add_argument(
        "--bob",
        type=float,
        default=0.0,
        help="Vertical bob amplitude (m).",
    )
    parser.add_argument("--height", type=float, default=1.0, help="Flight height (m).")
    parser.add_argument("--kp-pos", type=float, default=2.0, help="Position gain.")
    parser.add_argument("--kd-pos", type=float, default=1.6, help="Velocity damping.")
    parser.add_argument("--kp-att", type=float, default=8.0, help="Attitude gain.")
    parser.add_argument("--kd-att", type=float, default=2.2, help="Angular damping.")
    parser.add_argument("--max-accel", type=float, default=3.0, help="Max accel (m/s^2).")
    parser.add_argument("--max-tilt", type=float, default=25.0, help="Max tilt (deg).")
    parser.add_argument(
        "--yaw-follow",
        action="store_true",
        help="Yaw to face the current velocity direction.",
    )
    parser.add_argument("--sleep", type=float, default=0.01, help="Loop sleep (s).")
    return parser.parse_args()


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def _clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm > max_norm:
        return v / norm * max_norm
    return v


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        raise RuntimeError(
            "On macOS, MuJoCo's viewer requires running under `mjpython`.\n"
            "Try: mjpython quadcopter_square.py {}".format(args.xml_path)
        )

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drone_free")
    dof_adr = int(model.jnt_dofadr[free_jid])

    actuator_names = ["act_fl", "act_fr", "act_bl", "act_br"]
    actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in actuator_names
    ]

    mass = float(model.body_mass[body_id])
    gravity = np.array(model.opt.gravity, dtype=float)

    arm_length = 0.18
    yaw_gain = 0.02
    mix = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [arm_length, -arm_length, arm_length, -arm_length],
            [-arm_length, -arm_length, arm_length, arm_length],
            [yaw_gain, -yaw_gain, -yaw_gain, yaw_gain],
        ],
        dtype=float,
    )

    omega = 2.0 * math.pi / max(args.period, 1e-3)
    yaw_ref = 0.0

    max_tilt = math.radians(args.max_tilt)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = body_id
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -25.0

        while viewer.is_running():
            t = float(data.time)
            pos = np.array(data.xpos[body_id], dtype=float)
            qvel = np.array(data.qvel[dof_adr : dof_adr + 6], dtype=float)
            lin_vel = qvel[:3]
            ang_vel = qvel[3:]

            phase = omega * t
            target = np.array(
                [
                    args.radius * math.cos(phase),
                    args.radius * math.sin(phase),
                    args.height + args.bob * math.sin(0.5 * phase),
                ],
                dtype=float,
            )
            target_vel = np.array(
                [
                    -args.radius * omega * math.sin(phase),
                    args.radius * omega * math.cos(phase),
                    args.bob * 0.5 * omega * math.cos(0.5 * phase),
                ],
                dtype=float,
            )
            pos_err = target - pos

            if args.yaw_follow:
                dir_xy = target_vel[:2]
                if np.linalg.norm(dir_xy) > 1e-3:
                    yaw_ref = math.atan2(dir_xy[1], dir_xy[0])
            yaw_des = yaw_ref if args.yaw_follow else 0.0

            vel_err = target_vel - lin_vel
            a_des = args.kp_pos * pos_err + args.kd_pos * vel_err
            a_des = _clamp_norm(a_des, args.max_accel)

            force_des = mass * (a_des - gravity)
            horiz = np.linalg.norm(force_des[:2])
            max_horiz = abs(force_des[2]) * math.tan(max_tilt)
            if horiz > max_horiz and horiz > 1e-6:
                force_des[:2] *= max_horiz / horiz

            z_des = _normalize(force_des)
            x_c = np.array([math.cos(yaw_des), math.sin(yaw_des), 0.0])
            y_c = np.array([-math.sin(yaw_des), math.cos(yaw_des), 0.0])
            x_des = _normalize(np.cross(y_c, z_des))
            if np.linalg.norm(x_des) < 1e-6:
                x_des = np.array([1.0, 0.0, 0.0])
            y_des = np.cross(z_des, x_des)
            r_des = np.column_stack((x_des, y_des, z_des))

            r_cur = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
            z_body = r_cur[:, 2]
            u1 = float(np.dot(force_des, z_body))
            if u1 < 0.0:
                u1 = 0.0

            err_mat = 0.5 * (r_des.T @ r_cur - r_cur.T @ r_des)
            err_rot = np.array([err_mat[2, 1], err_mat[0, 2], err_mat[1, 0]])
            tau_body = -args.kp_att * err_rot - args.kd_att * ang_vel

            u = np.array([u1, tau_body[0], tau_body[1], tau_body[2]], dtype=float)
            thrusts = np.linalg.solve(mix, u)

            data.ctrl[:] = 0.0
            for thrust, aid in zip(thrusts, actuator_ids):
                ctrl_min, ctrl_max = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = float(np.clip(thrust, ctrl_min, ctrl_max))

            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.xpos[body_id]
            viewer.sync()
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
