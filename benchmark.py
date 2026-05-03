"""
Drone model comprehensive benchmark.

Metrics reported:
  - Survival rate + termination breakdown (ground crash / flip / OOB / timeout)
  - Mean/median episode length
  - Position RMSE, mean ± std, median
  - % steps within 0.20 m / 0.50 m / 1.00 m of target
  - First time drone reaches <0.20 m ("stabilisation step")
  - RMS linear + angular velocity  (flight smoothness)
  - % time upright (|quat_w| > 0.9  ≈  tilt < 26°)

Uses the raw env directly so terminal state is always accessible
(avoids SB3 DummyVecEnv auto-reset masking the crash frame).

Usage:
    python benchmark.py
    python benchmark.py --model models/trained/stage3_nav_final.zip
    python benchmark.py --episodes 50 --max-steps 1000
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.drone_env import WhoopDroneEnv

ROOT             = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL    = os.path.join(ROOT, "models", "trained", "best", "best_model.zip")
DEFAULT_VEC_NORM = os.path.join(ROOT, "models", "trained", "best", "vec_normalize.pkl")


# ── Termination reason classifier ────────────────────────────────────────────

def _term_reason(env: WhoopDroneEnv) -> str:
    """Classify why the episode terminated from current env state."""
    pos  = env._get_pos()
    quat = env._get_quat()
    if pos[2] < env.CRASH_Z:
        return "ground_crash"
    if abs(float(quat[0])) < env.FLIP_W:
        return "flip"
    if np.any(np.abs(pos[:2]) > env.MAX_RANGE_XY):
        return "oob_xy"
    if pos[2] > env.MAX_Z:
        return "oob_z"
    return "timeout"


# ── Benchmark core ────────────────────────────────────────────────────────────

def run_benchmark(
    model_path:    str,
    vec_norm_path: str,
    n_episodes:    int = 30,
    max_steps:     int = 1000,
):
    # Load VecNormalize stats (obs normalisation only; no env stepping needed)
    _dummy_vec = DummyVecEnv([lambda: WhoopDroneEnv(render_mode=None)])
    vec_norm   = VecNormalize.load(vec_norm_path, _dummy_vec)
    vec_norm.training    = False
    vec_norm.norm_reward = False
    print(f"[bench] VecNormalize:  {vec_norm_path}")

    model = PPO.load(model_path, device="cpu")
    print(f"[bench] Model:         {model_path}")
    print(f"[bench] Episodes:      {n_episodes}  |  max steps: {max_steps}\n")

    # Raw env — full state always accessible, no auto-reset surprises
    env = WhoopDroneEnv(render_mode=None, max_episode_steps=max_steps)

    # Accumulators
    ep_lengths      = []
    ep_rewards      = []
    ep_reasons      = []
    ep_first_stable = []   # step when dist first < 0.20 m  (-1 = never)

    # Per-step metrics (extended across all episodes)
    all_dists    = []
    all_vel_mag  = []
    all_av_mag   = []
    all_w_abs    = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done       = False
        total_r    = 0.0
        steps      = 0
        first_stab = -1
        ep_dists   = []

        while not done:
            # ── Normalise obs and predict ──────────────────────────────────
            norm_obs        = vec_norm.normalize_obs(obs[np.newaxis, :])  # (1, obs_dim)
            action, _state  = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])

            total_r += float(reward)
            steps   += 1
            done     = terminated or truncated

            # ── Per-step raw metrics ───────────────────────────────────────
            dist = float(info["distance_to_target"])
            vel  = env._get_vel()
            av   = env._get_ang_vel()
            w    = abs(float(env._get_quat()[0]))

            ep_dists.append(dist)
            all_dists.append(dist)
            all_vel_mag.append(float(np.linalg.norm(vel)))
            all_av_mag.append(float(np.linalg.norm(av)))
            all_w_abs.append(w)

            if first_stab < 0 and dist < 0.20:
                first_stab = steps

        # ── Episode summary ────────────────────────────────────────────────
        reason  = _term_reason(env) if terminated else "timeout"
        ep_mean = float(np.mean(ep_dists)) if ep_dists else float("nan")

        ep_lengths.append(steps)
        ep_rewards.append(total_r)
        ep_reasons.append(reason)
        ep_first_stable.append(first_stab)

        tag = "✓ SURVIVED" if reason == "timeout" else f"✗ {reason}"
        print(
            f"  Ep {ep+1:3d}: {tag:18s}  steps={steps:5d}  "
            f"reward={total_r:9.2f}  mean_dist={ep_mean:.3f} m"
        )

    env.close()
    _dummy_vec.close()

    # ── Aggregate statistics ──────────────────────────────────────────────────
    all_dists   = np.array(all_dists)
    all_vel_mag = np.array(all_vel_mag)
    all_av_mag  = np.array(all_av_mag)
    all_w_abs   = np.array(all_w_abs)

    n_survived = sum(1 for r in ep_reasons if r == "timeout")
    reason_counts: dict = {}
    for r in ep_reasons:
        reason_counts[r] = reason_counts.get(r, 0) + 1

    rmse       = float(np.sqrt(np.mean(all_dists ** 2)))
    pct_02     = 100.0 * float(np.mean(all_dists < 0.20))
    pct_05     = 100.0 * float(np.mean(all_dists < 0.50))
    pct_10     = 100.0 * float(np.mean(all_dists < 1.00))
    pct_upr    = 100.0 * float(np.mean(all_w_abs > 0.90))
    rms_vel    = float(np.sqrt(np.mean(all_vel_mag ** 2)))
    rms_av     = float(np.sqrt(np.mean(all_av_mag ** 2)))
    stab_times = [t for t in ep_first_stable if t >= 0]

    # ── Print report ──────────────────────────────────────────────────────────
    bar = "─" * 62
    print(f"\n{bar}")
    print(f"  BENCHMARK REPORT  ({n_episodes} ep × {max_steps} max steps)")
    print(bar)
    print(f"  Survival rate     : {n_survived}/{n_episodes}"
          f"  ({100. * n_survived / n_episodes:.1f}%)")
    print(f"  Mean ep length    : {np.mean(ep_lengths):.1f} steps"
          f"  (median {int(np.median(ep_lengths))})")
    print(f"  Mean reward       : {np.mean(ep_rewards):.2f}"
          f" ± {np.std(ep_rewards):.2f}")

    print(bar)
    print("  Termination breakdown:")
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        bar_vis = "█" * cnt
        print(f"    {reason:<16s}: {cnt:3d}  ({100.*cnt/n_episodes:5.1f}%)  {bar_vis}")

    print(bar)
    print("  Position accuracy  (all steps):")
    print(f"    RMSE              : {rmse:.3f} m")
    print(f"    Mean ± std        : {np.mean(all_dists):.3f} ± {np.std(all_dists):.3f} m")
    print(f"    Median dist       : {np.median(all_dists):.3f} m")
    print(f"    % steps < 0.20 m  : {pct_02:6.2f}%   (precise hover target)")
    print(f"    % steps < 0.50 m  : {pct_05:6.2f}%   (good hover)")
    print(f"    % steps < 1.00 m  : {pct_10:6.2f}%   (rough hover)")
    if stab_times:
        print(f"    1st reach <0.2 m  : avg {np.mean(stab_times):.1f} steps"
              f"  ({len(stab_times)}/{n_episodes} eps ever reached it)")
    else:
        print(f"    1st reach <0.2 m  : never  (0/{n_episodes} eps)")

    print(bar)
    print("  Flight stability   (all steps):")
    print(f"    RMS linear vel    : {rms_vel:.3f} m/s")
    print(f"    RMS angular vel   : {rms_av:.3f} rad/s")
    print(f"    % upright |w|>0.9 : {pct_upr:.1f}%   (tilt < 26°)")
    print(bar)

    return {
        "survival_rate":  n_survived / n_episodes,
        "mean_ep_length": float(np.mean(ep_lengths)),
        "rmse_m":         rmse,
        "pct_02m":        pct_02,
        "pct_05m":        pct_05,
        "rms_vel":        rms_vel,
        "rms_av":         rms_av,
        "reason_counts":  reason_counts,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone model comprehensive benchmark")
    parser.add_argument("--model",     default=DEFAULT_MODEL,    help="Path to .zip model")
    parser.add_argument("--vec-norm",  default=DEFAULT_VEC_NORM, help="Path to vec_normalize.pkl")
    parser.add_argument("--episodes",  type=int, default=30,     help="Number of episodes (default 30)")
    parser.add_argument("--max-steps", type=int, default=1000,   help="Max steps per episode (default 1000)")
    args = parser.parse_args()

    run_benchmark(
        model_path    = args.model,
        vec_norm_path = args.vec_norm,
        n_episodes    = args.episodes,
        max_steps     = args.max_steps,
    )
