"""
MJX + SBX GPU Training Pipeline
=================================
Stage 1 – Base hover policy         (MJXDroneEnvBase,   GPU sim, 5 M steps)
Stage 2 – Domain-randomized fine-tune (MJXDroneEnvDR,   GPU sim, 2 M steps)
Stage 3 – DR + Navigation challenge  (MJXDroneEnvDRNav, GPU sim, 10 M steps)

train_full.py ile birebir aynı mantık; farklar:
  • Fizik simülasyonu  → MuJoCo MJX  (JAX, GPU)
  • PPO ağ güncellemesi → SBX         (JAX/Flax JIT, ~20x hızlı)
  • VecNormalize        → RunningMeanStd (.pkl uyumlu)
  • Env dosyalarına hiç dokunulmaz

Gereksinimler:
    pip install "jax[cuda12]" mujoco sbx-rl tensorboardX

Kullanım:
    python cuda/train_full_cuda.py                              # 3 aşama
    python cuda/train_full_cuda.py --only-stage1
    python cuda/train_full_cuda.py --only-stage2
    python cuda/train_full_cuda.py --skip-stage1 <zip>
    python cuda/train_full_cuda.py --skip-stage2 <zip>
    python cuda/train_full_cuda.py --n-envs 256

Logs    → logs/stage1_base_mjx/  logs/stage2_dr_mjx/  logs/stage3_nav_mjx/
Models  → models/trained_mjx/
"""

import argparse
import os
import sys
import time
import pickle
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ── JAX / MJX ────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# ── SBX (SB3 + JAX) ──────────────────────────────────────────────────────────
try:
    from sbx import PPO as SBX_PPO
except ImportError:
    raise ImportError(
        "sbx-rl paketi bulunamadı.\n"
        "Kurulum: pip install sbx-rl"
    )

# ── TensorBoard ───────────────────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        SummaryWriter = None
        print("[WARNING] TensorBoard bulunamadı; log yazılmayacak.")

# ── Gymnasium ─────────────────────────────────────────────────────────────────
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# ═════════════════════════════════════════════════════════════════════════════
# MJX Ortam – Stage 1 (Temel hover, temiz sim)
# ═════════════════════════════════════════════════════════════════════════════

class MJXDroneEnvBase(gym.Env):
    """
    WhoopDroneEnv'in MJX karşılığı.

    Gözlem (18-dim)  – orijinal ile aynı:
        [0:3]   position error  (m)
        [3:6]   linear velocity (m/s)
        [6:10]  quaternion [w,x,y,z]
        [10:13] angular velocity (rad/s)
        [13]    yaw error (rad)
        [14:18] previous motor delta actions

    Aksiyon (4-dim): delta throttle [FL,FR,BL,BR] ∈ [-0.5, 0.5]
    """

    metadata = {"render_modes": [], "render_fps": 50}

    # ── Fiziksel sabitler (whoop_drone.xml ile senkron) ────────────────────
    MASS           = 0.050
    GRAVITY        = 9.81
    MAX_THRUST     = 0.245
    HOVER_THROTTLE = (MASS * GRAVITY) / (4.0 * MAX_THRUST)   # ≈ 0.50

    N_SUBSTEPS   = 10
    CRASH_Z      = 0.03
    FLIP_W       = 0.30
    MAX_RANGE_XY = 10.0
    MAX_Z        = 20.0

    def __init__(
        self,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # ── MuJoCo CPU modeli (MJX derlemesi için) ────────────────────────────
        _here      = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(_here, "..", "models", "whoop_drone.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(os.path.normpath(model_path))
        self.mj_data  = mujoco.MjData(self.mj_model)

        # ── MJX GPU modeli ────────────────────────────────────────────────────
        self.mjx_model = mjx.put_model(self.mj_model)

        self._drone_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "drone"
        )

        self.max_episode_steps = max_episode_steps
        self.render_mode       = render_mode

        # ── Spaces (orijinal WhoopDroneEnv ile aynı) ──────────────────────────
        obs_low = np.array(
            [-5, -5, -3, -10, -10, -10, -1, -1, -1, -1,
             -50, -50, -50, -np.pi, -0.5, -0.5, -0.5, -0.5],
            dtype=np.float32,
        )
        obs_high = np.array(
            [5, 5, 5, 10, 10, 10, 1, 1, 1, 1,
             50, 50, 50, np.pi, 0.5, 0.5, 0.5, 0.5],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(
            low=-0.5, high=0.5, shape=(4,), dtype=np.float32
        )

        # ── Episode state ─────────────────────────────────────────────────────
        self._mjx_data    = None
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._step_count  = 0
        self._target_pos  = np.array([0.0, 0.0, 1.0])
        self._target_yaw  = 0.0

    # ── JAX JIT simülasyon adımı ──────────────────────────────────────────────

    @staticmethod
    @jax.jit
    def _mjx_forward(mjx_model, mjx_data, ctrl):
        """N_SUBSTEPS adım GPU'da çalıştır (lax.scan → tek XLA kernel)."""
        mjx_data = mjx_data.replace(ctrl=ctrl)

        def _one_step(carry, _):
            m, d = carry
            d = mjx.step(m, d)
            return (m, d), None

        (_, mjx_data), _ = jax.lax.scan(
            _one_step,
            (mjx_model, mjx_data),
            None,
            length=MJXDroneEnvBase.N_SUBSTEPS,
        )
        return mjx_data

    # ── Gözlem ───────────────────────────────────────────────────────────────

    def _extract_obs(self, mjx_data) -> np.ndarray:
        # GPU verisini CPU'ya çek (sadece gözlem anında)
        qpos = np.array(jax.device_get(mjx_data.qpos))
        qvel = np.array(jax.device_get(mjx_data.qvel))

        pos     = qpos[0:3]
        quat    = qpos[3:7]   # [w, x, y, z]
        vel     = qvel[0:3]
        ang_vel = qvel[3:6]

        pos_err = (pos - self._target_pos).astype(np.float32)

        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        yaw     = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_err = np.float32(np.arctan2(
            np.sin(yaw - self._target_yaw),
            np.cos(yaw - self._target_yaw),
        ))

        return np.concatenate([
            pos_err,
            vel.astype(np.float32),
            quat.astype(np.float32),
            ang_vel.astype(np.float32),
            [yaw_err],
            self._prev_action,
        ]).astype(np.float32)

    # ── Ödül (WhoopDroneEnv._compute_reward ile birebir aynı) ────────────────

    def _compute_reward(self, mjx_data, action: np.ndarray) -> float:
        qpos = np.array(jax.device_get(mjx_data.qpos))
        qvel = np.array(jax.device_get(mjx_data.qvel))

        pos     = qpos[0:3]
        quat    = qpos[3:7]
        vel     = qvel[0:3]
        ang_vel = qvel[3:6]

        dist     = float(np.linalg.norm(pos - self._target_pos))
        r_pos    = float(np.exp(-2.0 * dist**2)) - 1.0

        w        = float(quat[0])
        r_orient = 2.0 * w * w - 1.0

        r_vel    = -0.10 * float(np.sum(vel**2))
        r_angvel = -0.05 * float(np.sum(ang_vel**2))

        yaw     = np.arctan2(2.0 * (w * quat[3] + quat[1] * quat[2]),
                             1.0 - 2.0 * (quat[2]**2 + quat[3]**2))
        yaw_err = float(np.arctan2(
            np.sin(yaw - self._target_yaw),
            np.cos(yaw - self._target_yaw),
        ))
        r_yaw    = -0.30 * yaw_err ** 2
        r_smooth = -0.10 * float(np.sum((action - self._prev_action)**2))
        r_alive  = 0.10
        r_crash  = -100.0 if pos[2] < self.CRASH_Z else 0.0

        return r_pos + r_orient + r_vel + r_angvel + r_yaw + r_smooth + r_alive + r_crash

    # ── Termination (WhoopDroneEnv._check_termination ile aynı) ──────────────

    def _check_termination(self, mjx_data) -> Tuple[bool, bool]:
        qpos = np.array(jax.device_get(mjx_data.qpos))
        pos  = qpos[0:3]
        quat = qpos[3:7]

        if pos[2] < self.CRASH_Z:                          return True, False
        if abs(quat[0]) < self.FLIP_W:                    return True, False
        if np.any(np.abs(pos[:2]) > self.MAX_RANGE_XY):   return True, False
        if pos[2] > self.MAX_Z:                            return True, False
        if self._step_count >= self.max_episode_steps:     return False, True
        return False, False

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Rastgele başlangıç pozisyonu (hedef etrafında ±spread)
        spread      = np.array([0.20, 0.20, 0.15])
        init_pos    = self._target_pos + rng.uniform(-spread, spread)
        init_pos[2] = max(init_pos[2], 0.15)
        self.mj_data.qpos[0:3] = init_pos

        # Rastgele hedef yaw
        self._target_yaw = float(rng.uniform(-np.pi, np.pi))

        # Rastgele başlangıç yaw + küçük tilt
        init_yaw    = float(rng.uniform(-np.pi, np.pi))
        cy, sy      = np.cos(init_yaw / 2.0), np.sin(init_yaw / 2.0)
        q_yaw       = np.array([cy, 0.0, 0.0, sy])
        tilt        = rng.uniform(-0.05, 0.05, 3)
        q_tilt      = np.zeros(4)
        q_tilt[1:4] = tilt
        q_tilt[0]   = np.sqrt(max(0.0, 1.0 - float(np.sum(tilt**2))))
        q_tilt      /= np.linalg.norm(q_tilt)
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_tilt
        quat = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        quat /= np.linalg.norm(quat)
        self.mj_data.qpos[3:7] = quat
        self.mj_data.qvel[:]   = rng.uniform(-0.05, 0.05, self.mj_model.nv)

        mujoco.mj_forward(self.mj_model, self.mj_data)

        # CPU → GPU
        self._mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        self._prev_action[:] = 0.0
        self._step_count     = 0

        return self._extract_obs(self._mjx_data), {}

    def step(self, action: np.ndarray):
        action   = np.clip(np.asarray(action, dtype=np.float32), -0.5, 0.5)
        throttle = np.clip(self.HOVER_THROTTLE + action, 0.0, 1.0)

        # GPU simülasyon adımı
        self._mjx_data = MJXDroneEnvBase._mjx_forward(
            self.mjx_model,
            self._mjx_data,
            jnp.array(throttle, dtype=jnp.float32),
        )

        self._step_count += 1

        reward              = self._compute_reward(self._mjx_data, action)
        terminated, truncated = self._check_termination(self._mjx_data)
        self._prev_action[:] = action

        obs  = self._extract_obs(self._mjx_data)
        pos  = np.array(jax.device_get(self._mjx_data.qpos[0:3]))
        info = {
            "distance_to_target": float(np.linalg.norm(pos - self._target_pos)),
            "step_count": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


# ═════════════════════════════════════════════════════════════════════════════
# MJX Ortam – Stage 2 (Domain Randomization)
# ═════════════════════════════════════════════════════════════════════════════

class MJXDroneEnvDR(MJXDroneEnvBase):
    """
    WhoopDroneEnvDR'nin MJX karşılığı.
    Her episode: thrust ±%15, mass ±%10, wind ≤0.8 m/s, obs noise σ=0.01
    """

    DR_THRUST_RANGE  = (0.85, 1.15)
    DR_MASS_RANGE    = (0.90, 1.10)
    DR_WIND_MAX      = 0.8
    DR_OBS_NOISE_STD = 0.01

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nom_thrust = self.mj_model.actuator_gear[:, 2].copy()
        self._nom_torque = self.mj_model.actuator_gear[:, 5].copy()
        self._nom_mass   = float(self.mj_model.body_mass[self._drone_body_id])
        self._dr_rng     = np.random.default_rng()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        # Gürültü RNG'sini base reset'ten önce kur
        self._dr_rng = np.random.default_rng(seed)

        obs, info = super().reset(seed=seed, options=options)

        dr_rng = np.random.default_rng(None if seed is None else seed + 0xDEAD)

        # Thrust / yaw-torque randomizasyonu
        for i in range(4):
            scale = dr_rng.uniform(*self.DR_THRUST_RANGE)
            self.mj_model.actuator_gear[i, 2] = self._nom_thrust[i] * scale
            self.mj_model.actuator_gear[i, 5] = self._nom_torque[i] * scale

        # Kütle randomizasyonu
        mass_scale = dr_rng.uniform(*self.DR_MASS_RANGE)
        self.mj_model.body_mass[self._drone_body_id] = self._nom_mass * mass_scale

        # Rüzgar randomizasyonu
        self.mj_model.opt.wind[:] = dr_rng.uniform(
            -self.DR_WIND_MAX, self.DR_WIND_MAX, 3
        )

        # Güncel model parametrelerini GPU'ya yansıt
        self.mjx_model = mjx.put_model(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        return self._extract_obs(self._mjx_data), info

    def _extract_obs(self, mjx_data) -> np.ndarray:
        obs   = super()._extract_obs(mjx_data)
        noise = self._dr_rng.normal(0.0, self.DR_OBS_NOISE_STD, obs.shape)
        return np.clip(
            obs + noise.astype(np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )


# ═════════════════════════════════════════════════════════════════════════════
# MJX Ortam – Stage 3 (DR + Navigation)
# ═════════════════════════════════════════════════════════════════════════════

class MJXDroneEnvDRNav(MJXDroneEnvDR):
    """
    WhoopDroneEnvDRNav'ın MJX karşılığı.
    DR etkilerine ek olarak mid-episode rastgele waypoint komutu ekler.
    """

    NAV_CMD_INTERVAL = 150
    NAV_CMD_RADIUS   = 1.5
    NAV_MIN_Z        = 0.4
    NAV_MAX_DZ       = 1.5
    NAV_SETTLE_DIST  = 0.20
    NAV_SETTLE_STEPS = 30
    YAW_REWARD_SCALE = 1.0   # base: 0.30, burada artırıldı

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self._target_pos = np.array([0.0, 0.0, 1.0])
        self._target_yaw = 0.0

        obs, info = super().reset(seed=seed, options=options)

        self._nav_rng = np.random.default_rng(
            None if seed is None else seed + 0xBEEF
        )
        self._nav_steps_since_cmd = 0
        self._nav_settled_steps   = 0

        self._target_pos, self._target_yaw = self._sample_nav_target()
        return self._extract_obs(self._mjx_data), info

    def _sample_nav_target(self) -> Tuple[np.ndarray, float]:
        angle = self._nav_rng.uniform(0.0, 2.0 * np.pi)
        r     = self._nav_rng.uniform(0.0, self.NAV_CMD_RADIUS) ** 0.5
        x     = r * np.cos(angle)
        y     = r * np.sin(angle)
        z     = self._nav_rng.uniform(self.NAV_MIN_Z, self.NAV_MIN_Z + self.NAV_MAX_DZ)
        yaw   = self._nav_rng.uniform(-np.pi, np.pi)
        return np.array([x, y, z]), float(yaw)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)

        if not (terminated or truncated):
            dist = info.get("distance_to_target", float("inf"))

            self._nav_settled_steps = (
                self._nav_settled_steps + 1 if dist < self.NAV_SETTLE_DIST else 0
            )
            self._nav_steps_since_cmd += 1

            if (self._nav_settled_steps >= self.NAV_SETTLE_STEPS or
                    self._nav_steps_since_cmd >= self.NAV_CMD_INTERVAL):
                self._target_pos, self._target_yaw = self._sample_nav_target()
                self._nav_steps_since_cmd = 0
                self._nav_settled_steps   = 0

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, mjx_data, action: np.ndarray) -> float:
        reward = super()._compute_reward(mjx_data, action)

        qpos    = np.array(jax.device_get(mjx_data.qpos))
        w       = float(qpos[3])
        yaw     = np.arctan2(
            2.0 * (w * qpos[6] + qpos[4] * qpos[5]),
            1.0 - 2.0 * (qpos[5]**2 + qpos[6]**2),
        )
        yaw_err = float(np.arctan2(
            np.sin(yaw - self._target_yaw),
            np.cos(yaw - self._target_yaw),
        ))
        # Yaw ağırlığını 0.30 → YAW_REWARD_SCALE'e çıkar
        reward -= (self.YAW_REWARD_SCALE - 0.30) * yaw_err ** 2
        return reward


# ═════════════════════════════════════════════════════════════════════════════
# RunningMeanStd – VecNormalize'ın JAX uyumlu karşılığı
# ═════════════════════════════════════════════════════════════════════════════

class RunningMeanStd:
    """
    Welford online algoritması ile running mean/var tutar.
    train_full.py'deki VecNormalize işlevinin karşılığı.
    .pkl formatında kayıt/yükleme → aşamalar arası taşıma için.
    """

    def __init__(self, shape, epsilon: float = 1e-8, clip: float = 10.0):
        self.mean    = np.zeros(shape, dtype=np.float64)
        self.var     = np.ones(shape, dtype=np.float64)
        self.count   = epsilon
        self.epsilon = epsilon
        self.clip    = clip

    def update(self, x: np.ndarray):
        x           = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        batch_mean  = np.mean(x, axis=0)
        batch_var   = np.var(x, axis=0)
        batch_count = x.shape[0]
        tot_count   = self.count + batch_count
        delta       = batch_mean - self.mean
        new_mean    = self.mean + delta * batch_count / tot_count
        m2          = (self.var * self.count
                       + batch_var * batch_count
                       + delta**2 * self.count * batch_count / tot_count)
        self.mean  = new_mean
        self.var   = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        normed = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean, "var": self.var, "count": self.count}, f)

    @classmethod
    def load(cls, path: str, shape, **kwargs) -> "RunningMeanStd":
        rms = cls(shape, **kwargs)
        with open(path, "rb") as f:
            d = pickle.load(f)
        rms.mean  = d["mean"]
        rms.var   = d["var"]
        rms.count = d["count"]
        return rms


# ═════════════════════════════════════════════════════════════════════════════
# Aşama konfigürasyonları  (train_full.py ile aynı mantık)
# ═════════════════════════════════════════════════════════════════════════════

STAGE1 = dict(
    run_name        = "stage1_base_mjx",
    EnvClass        = MJXDroneEnvBase,
    total_timesteps = 5_000_000,
    n_envs          = 256,
    seed            = 42,
    learning_rate   = 3e-4,
    n_steps         = 2048,
    batch_size      = 1024,
    n_epochs        = 10,
    clip_range      = 0.2,
    ent_coef        = 0.001,
    log_dir         = "cuda/model_results/logs",
    save_dir        = "cuda/model_results",
    checkpoint_freq = 250_000,
    eval_freq       = 200_000,
)

STAGE2 = dict(
    run_name        = "stage2_dr_mjx",
    EnvClass        = MJXDroneEnvDR,
    total_timesteps = 10_000_000,
    n_envs          = 256,
    seed            = 123,
    learning_rate   = 1e-4,
    n_steps         = 2048,
    batch_size      = 1024,
    n_epochs        = 5,
    clip_range      = 0.15,
    ent_coef        = 0.0005,
    log_dir         = "cuda/model_results/logs",
    save_dir        = "cuda/model_results",
    checkpoint_freq = 200_000,
    eval_freq       = 100_000,
)

STAGE3 = dict(
    run_name        = "stage3_nav_mjx",
    EnvClass        = MJXDroneEnvDRNav,
    total_timesteps = 20_000_000,
    n_envs          = 256,
    seed            = 456,
    learning_rate   = 5e-5,
    n_steps         = 2048,
    batch_size      = 1024,
    n_epochs        = 5,
    clip_range      = 0.10,
    ent_coef        = 0.0005,
    log_dir         = "cuda/model_results/logs",
    save_dir        = "cuda/model_results",
    checkpoint_freq = 200_000,
    eval_freq       = 100_000,
)


# ═════════════════════════════════════════════════════════════════════════════
# VecEnv yardımcıları
# ═════════════════════════════════════════════════════════════════════════════

def _env_fn(EnvClass, rank: int, seed: int):
    def _init():
        env = EnvClass()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def _build_vec_env(
    EnvClass,
    n_envs: int,
    seed: int,
    obs_rms: Optional[RunningMeanStd] = None,
) -> Tuple[DummyVecEnv, RunningMeanStd]:
    fns = [_env_fn(EnvClass, i, seed) for i in range(n_envs)]
    vec = DummyVecEnv(fns)
    if obs_rms is None:
        obs_rms = RunningMeanStd(vec.observation_space.shape)
    return vec, obs_rms


# ═════════════════════════════════════════════════════════════════════════════
# Değerlendirme yardımcısı
# ═════════════════════════════════════════════════════════════════════════════

def _evaluate(
    model,
    vec_env: DummyVecEnv,
    obs_rms: RunningMeanStd,
    n_episodes: int = 5,
) -> float:
    rewards  = []
    obs      = vec_env.reset()
    ep_rew   = 0.0
    ep_count = 0

    while ep_count < n_episodes:
        action, _ = model.predict(obs_rms.normalize(obs), deterministic=True)
        obs, r, dones, _ = vec_env.step(action)
        ep_rew += float(r[0])
        if dones[0]:
            rewards.append(ep_rew)
            ep_rew   = 0.0
            ep_count += 1
            obs = vec_env.reset()

    return float(np.mean(rewards)) if rewards else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Ana aşama eğitim fonksiyonu  (run_stage)
# ═════════════════════════════════════════════════════════════════════════════

def run_stage(
    cfg: dict,
    resume_path:  Optional[str] = None,
    obs_rms_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Bir eğitim aşamasını çalıştırır.

    Parameters
    ----------
    cfg          : STAGE1 / STAGE2 / STAGE3
    resume_path  : önceki aşamadan .zip model yolu
    obs_rms_path : önceki aşamadan RunningMeanStd .pkl yolu

    Returns
    -------
    (model_zip_path, obs_rms_pkl_path)
    """
    run_name        = cfg["run_name"]
    n_envs          = cfg["n_envs"]
    seed            = cfg["seed"]
    log_dir         = cfg["log_dir"]
    save_dir        = cfg["save_dir"]
    checkpoint_freq = cfg["checkpoint_freq"]
    eval_freq       = cfg["eval_freq"]

    best_dir        = os.path.join(save_dir, "best")
    best_model_path = os.path.join(best_dir, "best_model")
    best_rms_path   = os.path.join(best_dir, "obs_rms.pkl")

    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, run_name), exist_ok=True)

    # ── Normalisation ─────────────────────────────────────────────────────────
    obs_rms = None
    if obs_rms_path and os.path.exists(obs_rms_path):
        obs_rms = RunningMeanStd.load(obs_rms_path, (18,))
        print(f"[{run_name}] RunningMeanStd yüklendi → {obs_rms_path}")

    train_vec, obs_rms = _build_vec_env(cfg["EnvClass"], n_envs, seed, obs_rms)
    eval_vec,  _       = _build_vec_env(cfg["EnvClass"], 1, seed + 9999, obs_rms)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_writer = None
    if SummaryWriter is not None:
        tb_writer = SummaryWriter(os.path.join(log_dir, run_name))

    # ── Policy kwargs (train_full.py ile aynı ağ mimarisi) ───────────────────
    policy_kwargs = dict(net_arch=[256, 256], log_std_init=-1.5)

    # ── Model oluştur / yükle ─────────────────────────────────────────────────
    if resume_path and os.path.exists(resume_path):
        print(f"\n[{run_name}] Resuming from  {resume_path}")
        model = SBX_PPO.load(
            resume_path,
            env           = train_vec,
            learning_rate = cfg["learning_rate"],
            clip_range    = cfg["clip_range"],
            ent_coef      = cfg["ent_coef"],
            n_epochs      = cfg["n_epochs"],
            tensorboard_log = log_dir,
            verbose       = 1,
        )
    else:
        model = SBX_PPO(
            "MlpPolicy",
            train_vec,
            learning_rate   = cfg["learning_rate"],
            n_steps         = cfg["n_steps"],
            batch_size      = cfg["batch_size"],
            n_epochs        = cfg["n_epochs"],
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = cfg["clip_range"],
            ent_coef        = cfg["ent_coef"],
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = log_dir,
            verbose         = 1,
            seed            = seed,
        )

    print(f"\n{'═'*58}")
    print(f"  {run_name.upper()}")
    print(f"  Env       : {cfg['EnvClass'].__name__}")
    print(f"  n_envs    : {n_envs}")
    print(f"  steps     : {cfg['total_timesteps']:,}")
    print(f"  lr        : {cfg['learning_rate']}")
    print(f"  device    : {jax.default_backend()}  {jax.devices()}")
    print(f"{'═'*58}\n")

    # ── Eğitim döngüsü ────────────────────────────────────────────────────────
    # SBX'in learn() fonksiyonu periyodik ara işlem desteklemediğinden
    # toplam adımı küçük parçalara bölerek elle eval + checkpoint yapıyoruz.
    # Bu tam olarak train_full.py'deki EvalCallback + CheckpointCallback mantığı.

    total_steps      = cfg["total_timesteps"]
    chunk            = min(eval_freq, checkpoint_freq)
    steps_done       = 0
    best_mean_reward = -float("inf")
    start_time       = time.time()

    while steps_done < total_steps:
        learn_steps = min(chunk, total_steps - steps_done)

        model.learn(
            total_timesteps     = learn_steps,
            reset_num_timesteps = (steps_done == 0 and resume_path is None),
            progress_bar        = True,
            tb_log_name         = run_name,
        )
        steps_done += learn_steps

        # RunningMeanStd güncelle (rollout buffer gözlemlerinden)
        try:
            buf_obs = np.array(model.rollout_buffer.observations)
            if buf_obs.ndim == 3:
                buf_obs = buf_obs.reshape(-1, buf_obs.shape[-1])
            obs_rms.update(buf_obs)
        except AttributeError:
            pass

        # ── Eval ─────────────────────────────────────────────────────────────
        mean_reward = _evaluate(model, eval_vec, obs_rms, n_episodes=5)
        elapsed     = time.time() - start_time
        fps         = steps_done / max(elapsed, 1.0)
        print(f"[{run_name}] steps={steps_done:,}  "
              f"mean_reward={mean_reward:.2f}  fps={fps:.0f}")

        if tb_writer is not None:
            tb_writer.add_scalar("eval/mean_reward", mean_reward, steps_done)
            tb_writer.add_scalar("train/fps",         fps,         steps_done)

        # En iyi modeli kaydet (EvalCallback davranışı)
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            model.save(best_model_path)
            obs_rms.save(best_rms_path)
            if model.verbose:
                print(f"[{run_name}] ✓ Yeni en iyi ({mean_reward:.2f}) "
                      f"→ {best_model_path}.zip")

        # ── Checkpoint (CheckpointCallback davranışı) ─────────────────────────
        if steps_done % checkpoint_freq < chunk:
            ckpt = os.path.join(save_dir, f"{run_name}_{steps_done}")
            model.save(ckpt)
            obs_rms.save(os.path.join(save_dir,
                                      f"obs_rms_{run_name}_{steps_done}.pkl"))

    # ── Final kayıt ───────────────────────────────────────────────────────────
    final_model = os.path.join(save_dir, f"{run_name}_final")
    final_rms   = os.path.join(save_dir, f"obs_rms_{run_name}.pkl")
    model.save(final_model)
    obs_rms.save(final_rms)

    if tb_writer is not None:
        tb_writer.close()

    # En iyi modeli bir sonraki aşamaya aktar (train_full.py ile aynı tercih)
    if os.path.exists(f"{best_model_path}.zip") and os.path.exists(best_rms_path):
        out_model = f"{best_model_path}.zip"
        out_rms   = best_rms_path
        print(f"\n[{run_name}] ✓ Best model → {out_model}")
        print(f"[{run_name}] ✓ Best RMS   → {out_rms}")
    else:
        out_model = f"{final_model}.zip"
        out_rms   = final_rms
        print(f"\n[{run_name}] (best checkpoint yok – final kullanılıyor)")
        print(f"[{run_name}] Model → {out_model}")
        print(f"[{run_name}] RMS   → {out_rms}")

    train_vec.close()
    eval_vec.close()

    return out_model, out_rms


# ═════════════════════════════════════════════════════════════════════════════
# Yardımcı: RMS yolu bul  (_find_vn karşılığı)
# ═════════════════════════════════════════════════════════════════════════════

def _find_rms(save_dir: str, run_name: str) -> Optional[str]:
    for candidate in [
        os.path.join(save_dir, f"obs_rms_{run_name}.pkl"),
        os.path.join(save_dir, "obs_rms.pkl"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Giriş noktası  (train_full.py:main() ile aynı yapı)
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MJX + SBX üç aşamalı PPO eğitimi (GPU)"
    )
    parser.add_argument("--s1-steps",    type=int, default=5_000_000,
                        help="Stage 1 toplam adım (varsayılan 5M)")
    parser.add_argument("--s2-steps",    type=int, default=2_000_000,
                        help="Stage 2 toplam adım (varsayılan 2M)")
    parser.add_argument("--s3-steps",    type=int, default=10_000_000,
                        help="Stage 3 toplam adım (varsayılan 10M)")
    parser.add_argument("--n-envs",      type=int, default=64,
                        help="Paralel env sayısı (MJX önerisi: 64-512)")
    parser.add_argument("--skip-stage1", default=None, metavar="ZIP",
                        help="Stage 1'i atla, .zip yolu ver")
    parser.add_argument("--skip-stage2", default=None, metavar="ZIP",
                        help="Stage 1+2'yi atla, Stage-2 .zip yolu ver")
    parser.add_argument("--only-stage1", action="store_true",
                        help="Sadece Stage 1'i çalıştır")
    parser.add_argument("--only-stage2", action="store_true",
                        help="Stage 1 + Stage 2'yi çalıştır (Stage 3 atla)")
    args = parser.parse_args()

    # ── JAX cihaz bilgisi ─────────────────────────────────────────────────────
    backend = jax.default_backend()
    print(f"\n[MJX] JAX backend : {backend}")
    print(f"[MJX] Cihazlar    : {jax.devices()}")
    if backend == "cpu":
        print("[MJX] UYARI: JAX CPU modunda çalışıyor.\n"
              "       GPU hızlandırması için: pip install 'jax[cuda12]'\n")

    STAGE1["total_timesteps"] = args.s1_steps
    STAGE1["n_envs"]          = args.n_envs
    STAGE2["total_timesteps"] = args.s2_steps
    STAGE2["n_envs"]          = args.n_envs
    STAGE3["total_timesteps"] = args.s3_steps
    STAGE3["n_envs"]          = args.n_envs

    save_dir = STAGE1["save_dir"]

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.skip_stage2:
        s2_zip = args.skip_stage2
        s2_rms = _find_rms(os.path.dirname(os.path.abspath(s2_zip)), "stage2_dr_mjx")
        print(f"[main] Stage 1+2 atlanıyor → {s2_zip}")
        if s2_rms:
            print(f"[main] RMS bulundu → {s2_rms}")
        else:
            print("[main] UYARI: Stage 2 RMS bulunamadı.")
        s1_zip = s1_rms = None
    elif args.skip_stage1:
        s1_zip = args.skip_stage1
        s1_rms = _find_rms(os.path.dirname(os.path.abspath(s1_zip)), "stage1_base_mjx")
        print(f"[main] Stage 1 atlanıyor → {s1_zip}")
        if s1_rms:
            print(f"[main] RMS bulundu → {s1_rms}")
        else:
            print("[main] UYARI: Stage 1 RMS bulunamadı.")
        s2_zip = s2_rms = None
    else:
        s1_zip, s1_rms = run_stage(STAGE1)
        s2_zip = s2_rms = None

    if args.only_stage1:
        print("\n[main] --only-stage1 tamamlandı.")
        return

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if not args.skip_stage2:
        s2_zip, s2_rms = run_stage(STAGE2, resume_path=s1_zip, obs_rms_path=s1_rms)

    if args.only_stage2:
        print("\n[main] --only-stage2 tamamlandı.")
        print(f"[main] Deploy model : {s2_zip}")
        print(f"[main] RMS          : {s2_rms}")
        return

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    s3_zip, s3_rms = run_stage(STAGE3, resume_path=s2_zip, obs_rms_path=s2_rms)

    print("\n[main] Tüm aşamalar tamamlandı.")
    print(f"[main] Deploy model : {s3_zip}")
    print(f"[main] RMS          : {s3_rms}")


if __name__ == "__main__":
    main()
