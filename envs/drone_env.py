"""
WhoopDroneEnv – MuJoCo Gymnasium environment for a 50 g whoop-class drone.

Observation (18-dim):
    [0:3]  position error   (world frame, m)      : pos - target_pos
    [3:6]  linear velocity  (world frame, m/s)
    [6:10] orientation quaternion [w, x, y, z]    (body frame)
    [10:13] angular velocity (world frame, rad/s)
    [13]   yaw error        (rad)                  : yaw − target_yaw, ∈ [−π, π]
    [14:18] previous motor actions                 ∈ [0, 1]

Action (4-dim): motor throttles [FL, FR, BL, BR] ∈ [0, 1]
    FL = front-left  (CCW),  FR = front-right (CW)
    BL = back-left   (CW),   BR = back-right  (CCW)

Task: hover at target_pos (default [0, 0, 1] m) facing target_yaw (default 0 rad)
"""

import os
from typing import Dict, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WhoopDroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # ── Physical constants (sync with whoop_drone.xml) ──────────────────────
    MASS          = 0.050   # kg
    GRAVITY       = 9.81    # m/s²
    MAX_THRUST    = 0.245   # N per motor
    ARM           = 0.0325  # m (motor→centre distance, each axis)

    # Throttle required to hover (per motor)
    HOVER_THROTTLE: float = (MASS * GRAVITY) / (4.0 * MAX_THRUST)   # ≈ 0.50

    # ── Simulation timing ───────────────────────────────────────────────────
    # MuJoCo timestep is 0.002 s (500 Hz).
    # n_substeps = 10  →  50 Hz control frequency.
    N_SUBSTEPS = 10

    # ── Termination thresholds ──────────────────────────────────────────────
    CRASH_Z       = 0.03    # m   – below this height = crash
    FLIP_W        = 0.30    # quat w – below this ≈ tilt > 107°
    MAX_RANGE_XY  = 10.0    # m   – max horizontal distance from origin
    MAX_Z         = 20.0    # m   – max altitude

    def __init__(
        self,
        render_mode: Optional[str] = None,
        target_pos: Optional[np.ndarray] = None,
        target_yaw: float = 0.0,
        max_episode_steps: int = 1000,
    ):
        super().__init__()

        # ── MuJoCo model ────────────────────────────────────────────────────
        _here = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(_here, "..", "models", "whoop_drone.xml")
        self.model = mujoco.MjModel.from_xml_path(os.path.normpath(model_path))
        self.data  = mujoco.MjData(self.model)

        # Cache body / joint ids for fast lookup
        self._drone_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drone"
        )

        # ── Task parameters ──────────────────────────────────────────────────
        self.target_pos = (
            np.array(target_pos, dtype=np.float64)
            if target_pos is not None
            else np.array([0.0, 0.0, 1.0])
        )
        self.target_yaw = float(target_yaw)
        self.max_episode_steps = max_episode_steps

        # ── Spaces ───────────────────────────────────────────────────────────
        # Observation: pos_err(3) + vel(3) + quat(4) + ang_vel(3) + prev_act(4)
        obs_low  = np.array(
            [-5,  -5,  -3,          # pos error
             -10, -10, -10,         # linear velocity
             -1,  -1,  -1,  -1,    # quaternion
             -50, -50, -50,         # angular velocity
             -np.pi,                # yaw error
              0,   0,   0,   0],    # prev action
            dtype=np.float32,
        )
        obs_high = np.array(
            [ 5,   5,   5,
              10,  10,  10,
               1,   1,   1,  1,
              50,  50,  50,
              np.pi,
               1,   1,   1,  1],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # ── Internal state ───────────────────────────────────────────────────
        self._prev_action  = np.ones(4, dtype=np.float32) * self.HOVER_THROTTLE
        self._step_count   = 0

        # ── Rendering ────────────────────────────────────────────────────────
        self.render_mode = render_mode
        self._viewer     = None
        self._renderer   = None

    # ════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ════════════════════════════════════════════════════════════════════════

    def _get_pos(self)     -> np.ndarray: return self.data.qpos[0:3]
    def _get_quat(self)    -> np.ndarray: return self.data.qpos[3:7]   # [w,x,y,z]
    def _get_vel(self)     -> np.ndarray: return self.data.qvel[0:3]
    def _get_ang_vel(self) -> np.ndarray: return self.data.qvel[3:6]

    def _get_yaw(self) -> float:
        q = self.data.qpos[3:7]  # [w, x, y, z]
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _get_obs(self) -> np.ndarray:
        pos_err = (self._get_pos() - self.target_pos).astype(np.float32)
        vel     = self._get_vel().astype(np.float32)
        quat    = self._get_quat().astype(np.float32)
        ang_vel = self._get_ang_vel().astype(np.float32)
        yaw     = self._get_yaw()
        yaw_err = np.float32(np.arctan2(
            np.sin(yaw - self.target_yaw),
            np.cos(yaw - self.target_yaw),
        ))
        return np.concatenate([pos_err, vel, quat, ang_vel, [yaw_err], self._prev_action])

    def _get_info(self) -> Dict:
        pos = self._get_pos()
        return {
            "position":          pos.copy(),
            "distance_to_target": float(np.linalg.norm(pos - self.target_pos)),
            "step_count":         self._step_count,
        }

    # ── Termination ─────────────────────────────────────────────────────────

    def _check_termination(self) -> Tuple[bool, bool]:
        """Returns (terminated, truncated)."""
        pos  = self._get_pos()
        quat = self._get_quat()

        # Crashed into ground
        if pos[2] < self.CRASH_Z:
            return True, False
        # Flipped over
        if abs(quat[0]) < self.FLIP_W:
            return True, False
        # Out of bounds – horizontal
        if np.any(np.abs(pos[:2]) > self.MAX_RANGE_XY):
            return True, False
        # Out of bounds – altitude
        if pos[2] > self.MAX_Z:
            return True, False
        # Time limit
        if self._step_count >= self.max_episode_steps:
            return False, True

        return False, False

    # ── Reward ──────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> float:
        pos     = self._get_pos()
        quat    = self._get_quat()
        vel     = self._get_vel()
        ang_vel = self._get_ang_vel()

        pos_err = pos - self.target_pos
        dist    = float(np.linalg.norm(pos_err))

        # --- Position reward: exponential so gradient doesn't vanish far away
        r_pos = float(np.exp(-2.0 * dist**2)) - 1.0           # ∈ [-1, 0]

        # --- Upright orientation reward (w=1 → level flight)
        #     r_orient =  1 when perfectly upright,  -1 when fully inverted
        w          = float(quat[0])
        r_orient   = 2.0 * w * w - 1.0                        # ∈ [-1,  1]

        # --- Velocity penalty (encourages hovering, not drifting)
        r_vel      = -0.10 * float(np.sum(vel**2))

        # --- Angular velocity penalty
        r_angvel   = -0.05 * float(np.sum(ang_vel**2))

        # --- Yaw tracking reward
        yaw_err  = float(np.arctan2(
            np.sin(self._get_yaw() - self.target_yaw),
            np.cos(self._get_yaw() - self.target_yaw),
        ))
        r_yaw    = -0.30 * yaw_err ** 2

        # --- Action smoothness (penalise sudden throttle changes)
        r_smooth   = -0.10 * float(np.sum((action - self._prev_action)**2))

        # --- Alive bonus (encourages longer episodes)
        r_alive    = 0.10

        # --- Crash penalty
        r_crash    = -100.0 if pos[2] < self.CRASH_Z else 0.0

        return r_pos + r_orient + r_vel + r_angvel + r_yaw + r_smooth + r_alive + r_crash

    # ════════════════════════════════════════════════════════════════════════
    # Gymnasium API
    # ════════════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # ── Randomised initial position (near target) ────────────────────────
        spread       = np.array([0.20, 0.20, 0.15])
        init_pos     = self.target_pos + rng.uniform(-spread, spread)
        init_pos[2]  = max(init_pos[2], 0.15)          # never start underground
        self.data.qpos[0:3] = init_pos

        # ── Random target yaw for this episode ───────────────────────────────
        self.target_yaw = float(rng.uniform(-np.pi, np.pi))

        # ── Random initial yaw + small tilt ──────────────────────────────────
        init_yaw     = float(rng.uniform(-np.pi, np.pi))
        cy, sy       = np.cos(init_yaw / 2.0), np.sin(init_yaw / 2.0)
        q_yaw        = np.array([cy, 0.0, 0.0, sy])
        tilt         = rng.uniform(-0.05, 0.05, 3)
        q_tilt       = np.zeros(4)
        q_tilt[1:4]  = tilt
        q_tilt[0]    = np.sqrt(max(0.0, 1.0 - float(np.sum(tilt ** 2))))
        q_tilt      /= np.linalg.norm(q_tilt)
        # Multiply quaternions: q_final = q_yaw ⊗ q_tilt
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_tilt
        quat = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        quat /= np.linalg.norm(quat)
        self.data.qpos[3:7] = quat

        # ── Small random initial velocity ────────────────────────────────────
        self.data.qvel[:] = rng.uniform(-0.05, 0.05, self.model.nv)

        # ── Reset tracking state ─────────────────────────────────────────────
        self._prev_action[:] = self.HOVER_THROTTLE
        self._step_count      = 0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)

        # Apply motor commands and advance simulation
        self.data.ctrl[:] = action
        for _ in range(self.N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        reward               = self._compute_reward(action)
        terminated, truncated = self._check_termination()
        self._prev_action[:] = action

        obs  = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self._viewer.sync()

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=480, width=640
                )
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
