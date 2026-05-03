"""
WhoopDroneEnvDRNav – Domain Randomization + Navigation Challenge
=================================================================
Extends WhoopDroneEnvDR with mid-episode random waypoint commands so the
policy learns to fly to targets, not just hover at a fixed point.

Each episode:
  • Drone spawns near [0, 0, 1] (origin) regardless of the first target.
  • A random 3-D waypoint (position + yaw) is issued immediately after reset.
  • A new waypoint is issued when the drone settles (dist < 0.20 m for 30
    consecutive steps) OR after NAV_CMD_INTERVAL steps — whichever is first.
  • All Stage-2 DR effects (thrust ±15%, mass ±10%, wind, IMU noise) are
    fully inherited from WhoopDroneEnvDR.

Reward changes vs base env:
  • Yaw weight increased 0.30 → 1.0 so the policy learns to track heading
    explicitly rather than ignoring it and focusing only on XYZ position.

Use this for Stage 3 fine-tuning after the base DR policy (Stage 2) has
learned robust hovering.
"""

from typing import Dict, Optional

import numpy as np

from .drone_env_dr import WhoopDroneEnvDR


class WhoopDroneEnvDRNav(WhoopDroneEnvDR):

    # ── Navigation waypoint parameters ──────────────────────────────────────
    NAV_CMD_INTERVAL = 300   # steps between forced new waypoints (150→300: give drone time to reach target)
    NAV_CMD_RADIUS   = 0.8   # m  – max XY radius of random targets (1.5→0.8: start with closer targets)
    NAV_MIN_Z        = 0.4   # m  – minimum target altitude
    NAV_MAX_DZ       = 1.5   # m  – target Z ∈ [MIN_Z, MIN_Z + MAX_DZ]
    NAV_SETTLE_DIST  = 0.30  # m  – "on target" threshold (0.20→0.30: slightly more tolerant)
    NAV_SETTLE_STEPS = 30    # consecutive steps within settle dist → new cmd

    # ── Reward adjustment ────────────────────────────────────────────────────
    # Base env uses -0.30 * yaw_err². We increase it to -0.50 here (was 1.0).
    # Position reward must dominate; yaw is secondary.
    YAW_REWARD_SCALE = 0.5

    # ────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        # Force spawn near [0, 0, 1] by temporarily setting target to origin.
        # After super().reset() we replace with the real navigation target so
        # the pos_err in the first observation is meaningful.
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0

        # super().reset() handles DR randomisation + drone placement near target_pos
        _, info = super().reset(seed=seed, options=options)

        # Separate RNG for waypoints (independent from DR noise RNG)
        self._nav_rng = np.random.default_rng(
            None if seed is None else seed + 0xBEEF
        )
        self._nav_steps_since_cmd = 0
        self._nav_settled_steps   = 0

        # Issue the first navigation target; recompute obs with it
        self.target_pos, self.target_yaw = self._sample_nav_target()
        return self._get_obs(), info

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _sample_nav_target(self):
        """Sample a random (target_pos, target_yaw) within the nav arena."""
        angle = self._nav_rng.uniform(0.0, 2.0 * np.pi)
        r     = self._nav_rng.uniform(0.0, self.NAV_CMD_RADIUS) ** 0.5  # uniform area
        x     = r * np.cos(angle)
        y     = r * np.sin(angle)
        z     = self._nav_rng.uniform(self.NAV_MIN_Z, self.NAV_MIN_Z + self.NAV_MAX_DZ)
        yaw   = self._nav_rng.uniform(-np.pi, np.pi)
        return np.array([x, y, z]), float(yaw)

    # ── Gymnasium API overrides ──────────────────────────────────────────────

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Update waypoint logic only while episode is still running
        if not (terminated or truncated):
            dist = info.get("distance_to_target", float("inf"))

            # Count consecutive steps close enough to declare "settled"
            if dist < self.NAV_SETTLE_DIST:
                self._nav_settled_steps += 1
            else:
                self._nav_settled_steps = 0

            self._nav_steps_since_cmd += 1

            # Issue new target if settled OR time-limit elapsed
            if (self._nav_settled_steps >= self.NAV_SETTLE_STEPS or
                    self._nav_steps_since_cmd >= self.NAV_CMD_INTERVAL):
                self.target_pos, self.target_yaw = self._sample_nav_target()
                self._nav_steps_since_cmd = 0
                self._nav_settled_steps   = 0

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray) -> float:
        reward = super()._compute_reward(action)

        # The base env has r_yaw = -0.30 * yaw_err².
        # We boost it to -YAW_REWARD_SCALE * yaw_err² by adding the difference.
        yaw_err = float(np.arctan2(
            np.sin(self._get_yaw() - self.target_yaw),
            np.cos(self._get_yaw() - self.target_yaw),
        ))
        reward -= (self.YAW_REWARD_SCALE - 0.30) * yaw_err ** 2
        return reward
