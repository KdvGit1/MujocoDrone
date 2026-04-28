"""
WhoopDroneEnvDR – Domain Randomization wrapper
================================================
Subclass of WhoopDroneEnv.  Each episode re-randomizes:
  • Motor thrust      ±15 % per motor  (motor wear / prop imbalance)
  • Total mass        ±10 %            (battery / payload variation)
  • Constant wind     ≤0.8 m/s in any direction
  • Observation noise Gaussian σ=0.01  (IMU noise)

Use this for Stage 2 (sim-to-real hardening) fine-tuning after the
base policy has already learned to hover in clean simulation.
"""

from typing import Dict, Optional

import mujoco
import numpy as np

from .drone_env import WhoopDroneEnv


class WhoopDroneEnvDR(WhoopDroneEnv):
    # ── Randomization ranges ─────────────────────────────────────────────────
    DR_THRUST_RANGE  = (0.85, 1.15)   # ±15% per motor
    DR_MASS_RANGE    = (0.90, 1.10)   # ±10% total mass
    DR_WIND_MAX      = 0.8            # m/s max per axis
    DR_OBS_NOISE_STD = 0.01           # Gaussian std added to every obs dim

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Store nominal (XML-defined) values so we can scale from them
        self._nom_thrust = self.model.actuator_gear[:, 2].copy()   # shape (4,)
        self._nom_torque = self.model.actuator_gear[:, 5].copy()   # shape (4,)
        self._nom_mass   = float(self.model.body_mass[self._drone_body_id])

        # RNG for per-step observation noise (initialised here; replaced in reset)
        self._dr_rng = np.random.default_rng()

    # ── Gymnasium API overrides ───────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        # Initialise noise RNG BEFORE super().reset() calls _get_obs()
        self._dr_rng = np.random.default_rng(seed)

        _, info = super().reset(seed=seed, options=options)

        # Use a separate RNG derived from seed for DR params so noise RNG
        # and parameter RNG don't share state.
        dr_param_rng = np.random.default_rng(
            None if seed is None else seed + 0xDEAD
        )

        # ── Per-motor thrust & yaw-torque randomisation ───────────────────────
        for i in range(4):
            scale = dr_param_rng.uniform(*self.DR_THRUST_RANGE)
            self.model.actuator_gear[i, 2] = self._nom_thrust[i] * scale
            self.model.actuator_gear[i, 5] = self._nom_torque[i] * scale

        # ── Mass randomisation ────────────────────────────────────────────────
        mass_scale = dr_param_rng.uniform(*self.DR_MASS_RANGE)
        self.model.body_mass[self._drone_body_id] = self._nom_mass * mass_scale

        # ── Wind randomisation ────────────────────────────────────────────────
        self.model.opt.wind[:] = dr_param_rng.uniform(
            -self.DR_WIND_MAX, self.DR_WIND_MAX, 3
        )

        # Re-forward with new params (state is unchanged; forces are updated)
        mujoco.mj_forward(self.model, self.data)

        # Return fresh obs (with noise) built on the new model state
        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        obs   = super()._get_obs()
        noise = self._dr_rng.normal(0.0, self.DR_OBS_NOISE_STD, obs.shape)
        return np.clip(
            obs + noise.astype(np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )
