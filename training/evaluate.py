"""
Evaluate a trained PPO model on WhoopDroneEnv with MuJoCo viewer.

Usage:
    # Visualise 5 episodes with the MuJoCo viewer
    python training/evaluate.py models/trained/whoop_hover_ppo_final.zip

    # Headless, 20 episodes
    python training/evaluate.py models/trained/best/best_model.zip --episodes 20 --no-render

    # Save an MP4 video (requires ffmpeg)
    python training/evaluate.py models/trained/best/best_model.zip --video out.mp4

    # Random-command challenge mode (position + yaw targets change every N steps)
    python training/evaluate.py models/trained/stage2_dr_final.zip --challenge
    python training/evaluate.py models/trained/stage2_dr_final.zip --challenge --cmd-interval 100 --cmd-radius 1.5
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.drone_env import WhoopDroneEnv


# ─────────────────────────────────────────────────────────────────────────────
# Random command generator
# ─────────────────────────────────────────────────────────────────────────────

def _random_command(rng: np.random.Generator, radius: float, min_z: float = 0.4) -> tuple:
    """
    Generate a random (target_pos, target_yaw) pair.
    XY sampled uniformly within a circle of given radius.
    Z sampled between min_z and min_z+1.5 m (never too low to crash).
    Yaw sampled uniformly in [-π, π].
    """
    angle  = rng.uniform(0.0, 2.0 * np.pi)
    r      = rng.uniform(0.0, radius) ** 0.5   # sqrt for uniform area sampling
    x      = r * np.cos(angle)
    y      = r * np.sin(angle)
    z      = rng.uniform(min_z, min_z + 1.5)
    yaw    = rng.uniform(-np.pi, np.pi)
    return np.array([x, y, z]), float(yaw)


def evaluate(
    model_path:    str,
    n_episodes:    int   = 5,
    render:        bool  = True,
    video_path:    str   = None,
    challenge:     bool  = False,
    cmd_interval:  int   = 150,    # steps between random commands
    cmd_radius:    float = 1.5,    # m – max XY radius of target waypoints
    vec_norm_override: str = None, # explicit path to vec_normalize.pkl
):
    # ── Resolve vec normalise stats ───────────────────────────────────────────
    import glob as _glob
    model_dir = os.path.dirname(model_path)

    def _find_vec_norm(model_dir: str, model_path: str) -> str:
        """Search for VecNormalize stats using multiple heuristics."""
        candidates = []

        # 1. Exact legacy name
        candidates.append(os.path.join(model_dir, "vec_normalize.pkl"))
        candidates.append(os.path.join(os.path.dirname(model_dir), "vec_normalize.pkl"))

        # 2. Infer from model filename: "stage2_dr_final.zip" → "vec_normalize_stage2_dr.pkl"
        stem = os.path.splitext(os.path.basename(model_path))[0]  # e.g. "stage2_dr_final"
        for suffix in ("_final", "_best"):
            if stem.endswith(suffix):
                run_name = stem[: -len(suffix)]
                candidates.append(os.path.join(model_dir, f"vec_normalize_{run_name}.pkl"))

        # 3. Checkpoint naming: "stage2_dr_5000000_steps.zip" → "stage2_dr_vecnormalize_5000000_steps.pkl"
        import re as _re
        m = _re.match(r"^(.+)_(\d+)_steps$", stem)
        if m:
            run_name, steps = m.group(1), m.group(2)
            candidates.append(
                os.path.join(model_dir, f"{run_name}_vecnormalize_{steps}_steps.pkl")
            )

        # 4. Glob fallback: any vec_normalize*.pkl in the model directory
        for p in sorted(_glob.glob(os.path.join(model_dir, "vec_normalize*.pkl"))):
            candidates.append(p)

        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    vec_norm_path = _find_vec_norm(model_dir, model_path)
    if vec_norm_override:
        vec_norm_path = vec_norm_override

    # ── Build environment ─────────────────────────────────────────────────────
    render_mode = "human" if render and video_path is None else (
        "rgb_array" if video_path else None
    )
    raw_env = WhoopDroneEnv(render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: raw_env])

    if vec_norm_path is not None:
        print(f"[eval] Loading VecNormalize from {vec_norm_path}")
        vec_env            = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training   = False
        vec_env.norm_reward = False
    else:
        print("[eval] WARNING: No vec_normalize.pkl found – running without normalisation.")
        print("[eval]          Pass --vec-normalize <path.pkl> to fix this.")

    # ── Load model ────────────────────────────────────────────────────────────
    model = PPO.load(model_path, device="cpu")
    print(f"[eval] Model loaded from {model_path}")
    if challenge:
        print(
            f"[eval] CHALLENGE MODE – new target every {cmd_interval} steps, "
            f"XY radius {cmd_radius:.1f} m"
        )

    # ── Video writer setup ────────────────────────────────────────────────────
    writer = None
    if video_path:
        try:
            import imageio
            writer = imageio.get_writer(video_path, fps=50)
            print(f"[eval] Recording video to {video_path}")
        except ImportError:
            print("[eval] imageio not installed – skipping video export.")
            writer = None

    # ── Evaluation loop ───────────────────────────────────────────────────────
    episode_rewards  = []
    episode_lengths  = []
    episode_dists    = []
    # Challenge-mode stats
    cmd_reach_counts  = []   # how many commands were reached within 0.20 m
    cmd_total_counts  = []

    rng = np.random.default_rng(42)

    for ep in range(n_episodes):
        # Reset target to origin before each episode so the drone spawns
        # near [0,0,1] regardless of where the previous episode's challenge
        # target ended up (avoids cross-episode spawn-position drift).
        if challenge:
            raw_env.target_pos = np.array([0.0, 0.0, 1.0])
            raw_env.target_yaw = 0.0

        obs     = vec_env.reset()
        done    = False
        total_r = 0.0
        steps   = 0
        dists   = []

        # Per-episode challenge tracking
        cmd_issued  = 0
        cmd_reached = 0
        settled_steps = 0   # consecutive steps within 0.20 m

        # Start with a hover command (stay near reset position for first interval)
        if challenge:
            tgt_pos, tgt_yaw = _random_command(rng, cmd_radius)
            raw_env.target_pos = tgt_pos
            raw_env.target_yaw = tgt_yaw
            cmd_issued += 1
            print(
                f"\n  [Ep {ep+1}] cmd {cmd_issued}: "
                f"pos=[{tgt_pos[0]:+.2f},{tgt_pos[1]:+.2f},{tgt_pos[2]:.2f}]  "
                f"yaw={np.degrees(tgt_yaw):+.0f}°"
            )

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)

            total_r += float(reward[0])
            steps   += 1
            done     = bool(done_arr[0])

            dist = info_arr[0].get("distance_to_target", None)
            if dist is not None:
                dists.append(dist)

            # ── Challenge-mode logic ─────────────────────────────────────────
            if challenge and not done:
                # Count consecutive steps "close enough" to current target
                if dist is not None and dist < 0.20:
                    settled_steps += 1
                else:
                    settled_steps = 0

                # Issue new command if:
                #   a) reached settling criterion (drone is on target), OR
                #   b) the fixed time interval has elapsed
                new_cmd = (
                    settled_steps >= 30                    # 30 steps ≈ 0.6 s settled
                    or (steps % cmd_interval == 0)
                )
                if new_cmd:
                    if settled_steps >= 30:
                        cmd_reached += 1
                    settled_steps = 0

                    tgt_pos, tgt_yaw = _random_command(rng, cmd_radius)
                    raw_env.target_pos = tgt_pos
                    raw_env.target_yaw = tgt_yaw
                    cmd_issued += 1
                    print(
                        f"  [Ep {ep+1}] cmd {cmd_issued}: "
                        f"pos=[{tgt_pos[0]:+.2f},{tgt_pos[1]:+.2f},{tgt_pos[2]:.2f}]  "
                        f"yaw={np.degrees(tgt_yaw):+.0f}°  "
                        f"(prev reached={cmd_reached}/{cmd_issued-1})",
                        end="\n",
                    )

            if writer is not None:
                frame = raw_env.render()
                if frame is not None:
                    writer.append_data(frame)

        # Final settle check for last command
        if challenge and settled_steps >= 30:
            cmd_reached += 1

        mean_dist = float(np.mean(dists)) if dists else float("nan")
        episode_rewards.append(total_r)
        episode_lengths.append(steps)
        episode_dists.append(mean_dist)

        cmd_reach_counts.append(cmd_reached)
        cmd_total_counts.append(max(cmd_issued, 1))

        success_rate = 100.0 * cmd_reached / max(cmd_issued, 1) if challenge else None

        if challenge:
            print(
                f"  Episode {ep+1:2d}:  reward={total_r:8.2f}  "
                f"steps={steps:5d}  mean_dist={mean_dist:.3f} m  "
                f"cmds={cmd_reached}/{cmd_issued} ({success_rate:.0f}% reached)"
            )
        else:
            print(
                f"  Episode {ep+1:2d}:  reward={total_r:8.2f}  "
                f"steps={steps:5d}  mean_dist={mean_dist:.3f} m"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(
        f"  Mean reward    : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(
        f"  Mean ep length : {np.mean(episode_lengths):.0f} steps"
    )
    print(
        f"  Mean dist2goal : {np.nanmean(episode_dists):.3f} m"
    )
    if challenge:
        total_reached = sum(cmd_reach_counts)
        total_issued  = sum(cmd_total_counts)
        print(
            f"  CMD success    : {total_reached}/{total_issued} "
            f"({100.*total_reached/max(total_issued,1):.1f}%)  "
            f"[threshold: dist<0.20 m for ≥30 steps]"
        )
    print("─" * 60)

    if writer is not None:
        writer.close()

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path",     help="Path to trained model (.zip)")
    parser.add_argument("--episodes",     type=int,   default=5,    help="Number of evaluation episodes")
    parser.add_argument("--no-render",    action="store_true",       help="Disable MuJoCo viewer")
    parser.add_argument("--video",        default=None,              help="Save video to this path (e.g. out.mp4)")
    parser.add_argument("--vec-normalize",  default=None,              help="Path to VecNormalize .pkl (auto-detected if omitted)")
    parser.add_argument("--challenge",    action="store_true",       help="Enable random-command challenge mode")
    parser.add_argument("--cmd-interval", type=int,   default=150,   help="[challenge] Steps between forced new commands")
    parser.add_argument("--cmd-radius",   type=float, default=1.5,   help="[challenge] Max XY radius of random targets (m)")
    args = parser.parse_args()

    evaluate(
        model_path   = args.model_path,
        n_episodes   = args.episodes,
        render       = not args.no_render,
        video_path   = args.video,
        challenge    = args.challenge,
        cmd_interval = args.cmd_interval,
        cmd_radius   = args.cmd_radius,
        vec_norm_override = args.vec_normalize,
    )

