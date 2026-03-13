import os
import sys

import mujoco
import numpy as np


class SimpleMujocoEnv:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, action: np.ndarray | None = None, nstep: int = 1) -> np.ndarray:
        if action is not None:
            self.data.ctrl[:] = action
        for _ in range(nstep):
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def require_mjpython(self, script_name: str) -> None:
        if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
            raise RuntimeError(
                "On macOS, MuJoCo's viewer requires running under `mjpython`.\n"
                f"Try: mjpython {script_name} {self.xml_path}"
            )
