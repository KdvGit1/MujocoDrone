"""
Two-Stage Local Training Pipeline
==================================
Stage 1 – Base hover policy         (WhoopDroneEnv,   clean sim, 5 M steps)
Stage 2 – Domain-randomized fine-tune (WhoopDroneEnvDR, 2 M steps, resume S1)

Usage:
    python training/train_full.py                             # both stages
    python training/train_full.py --s1-steps 3000000         # shorter S1
    python training/train_full.py --only-stage1              # S1 only
    python training/train_full.py --skip-stage1 models/trained/stage1_base_final.zip
    python training/train_full.py --n-envs 4                 # fewer parallel envs

Logs    → logs/stage1_base/  and  logs/stage2_dr/   (TensorBoard)
Models  → models/trained/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.drone_env    import WhoopDroneEnv
from envs.drone_env_dr import WhoopDroneEnvDR


# ─────────────────────────────────────────────────────────────────────────────
# Default stage configurations
# ─────────────────────────────────────────────────────────────────────────────

STAGE1 = dict(
    run_name         = "stage1_base",
    EnvClass         = WhoopDroneEnv,
    total_timesteps  = 5_000_000,
    n_envs           = 8,
    seed             = 42,
    learning_rate    = 3e-4,
    n_steps          = 2048,
    batch_size       = 64,
    n_epochs         = 10,
    clip_range       = 0.2,
    ent_coef         = 0.001,
    log_dir          = "logs",
    save_dir         = "models/trained",
    checkpoint_freq  = 100_000,
    eval_freq        = 50_000,
)

STAGE2 = dict(
    run_name         = "stage2_dr",
    EnvClass         = WhoopDroneEnvDR,
    total_timesteps  = 2_000_000,
    n_envs           = 8,
    seed             = 123,
    learning_rate    = 1e-4,    # lower LR for fine-tuning
    n_steps          = 2048,
    batch_size       = 64,
    n_epochs         = 5,       # fewer epochs → less forgetting
    clip_range       = 0.15,    # tighter clip → conservative updates
    ent_coef         = 0.0005,
    log_dir          = "logs",
    save_dir         = "models/trained",
    checkpoint_freq  = 50_000,
    eval_freq        = 25_000,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _env_fn(EnvClass, rank: int, seed: int):
    def _init():
        env = EnvClass()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def _build_vec_env(EnvClass, n_envs: int, seed: int, vec_norm_path: str = None):
    """Create (optionally normalised) vectorised environment."""
    fns = [_env_fn(EnvClass, i, seed) for i in range(n_envs)]
    raw = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)

    if vec_norm_path and os.path.exists(vec_norm_path):
        vn = VecNormalize.load(vec_norm_path, raw)
        vn.training    = True
        vn.norm_reward = True
    else:
        vn = VecNormalize(
            raw,
            norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0,
            gamma=0.99,
        )
    return vn


def _build_eval_env(EnvClass, seed: int, vec_norm_path: str = None):
    """Create eval environment (reward NOT normalised)."""
    raw = DummyVecEnv([_env_fn(EnvClass, 0, seed)])

    if vec_norm_path and os.path.exists(vec_norm_path):
        vn = VecNormalize.load(vec_norm_path, raw)
        vn.training    = False
        vn.norm_reward = False
    else:
        vn = VecNormalize(
            raw,
            norm_obs=True, norm_reward=False,
            clip_obs=10.0, training=False,
        )
    return vn


# ─────────────────────────────────────────────────────────────────────────────
# Core stage runner
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(cfg: dict, resume_path: str = None, vec_norm_path: str = None):
    """
    Train one stage of the pipeline.

    Parameters
    ----------
    cfg           : stage configuration dict (STAGE1 or STAGE2)
    resume_path   : path to a .zip checkpoint to resume from (Stage 2 uses S1 output)
    vec_norm_path : path to a VecNormalize .pkl to warm-start normalisation stats

    Returns
    -------
    (model_zip_path, vec_norm_pkl_path)
    """
    run_name = cfg["run_name"]
    n_envs   = cfg["n_envs"]
    seed     = cfg["seed"]
    log_dir  = cfg["log_dir"]
    save_dir = cfg["save_dir"]

    os.makedirs(os.path.join(save_dir, "best"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_env = _build_vec_env(cfg["EnvClass"], n_envs, seed, vec_norm_path)
    eval_env  = _build_eval_env(cfg["EnvClass"], seed + 9999, vec_norm_path)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, cfg["checkpoint_freq"] // n_envs),
        save_path=save_dir,
        name_prefix=run_name,
        save_vecnormalize=True,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=os.path.join(log_dir, run_name),
        eval_freq=max(1, cfg["eval_freq"] // n_envs),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    # ── Build or resume model ─────────────────────────────────────────────────
    if resume_path and os.path.exists(resume_path):
        print(f"\n[{run_name}] Resuming from  {resume_path}")
        model = PPO.load(
            resume_path,
            env=train_env,
            learning_rate=cfg["learning_rate"],
            clip_range=cfg["clip_range"],
            ent_coef=cfg["ent_coef"],
            n_epochs=cfg["n_epochs"],
            tensorboard_log=log_dir,
            verbose=1,
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate  = cfg["learning_rate"],
            n_steps        = cfg["n_steps"],
            batch_size     = cfg["batch_size"],
            n_epochs       = cfg["n_epochs"],
            gamma          = 0.99,
            gae_lambda     = 0.95,
            clip_range     = cfg["clip_range"],
            ent_coef       = cfg["ent_coef"],
            vf_coef        = 0.5,
            max_grad_norm  = 0.5,
            policy_kwargs  = policy_kwargs,
            tensorboard_log= log_dir,
            verbose        = 1,
            seed           = seed,
        )

    print(f"\n{'═'*58}")
    print(f"  {run_name.upper()}")
    print(f"  Env       : {cfg['EnvClass'].__name__}")
    print(f"  n_envs    : {n_envs}")
    print(f"  steps     : {cfg['total_timesteps']:,}")
    print(f"  lr        : {cfg['learning_rate']}")
    print(f"{'═'*58}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps  = cfg["total_timesteps"],
        callback         = CallbackList([ckpt_cb, eval_cb]),
        tb_log_name      = run_name,
        progress_bar     = True,
        reset_num_timesteps = (resume_path is None),
    )

    # ── Save artefacts ─────────────────────────────────────────────────────────
    out_model = os.path.join(save_dir, f"{run_name}_final")
    out_vn    = os.path.join(save_dir, f"vec_normalize_{run_name}.pkl")
    model.save(out_model)
    train_env.save(out_vn)

    print(f"\n[{run_name}] Model  → {out_model}.zip")
    print(f"[{run_name}] VecNorm→ {out_vn}")

    train_env.close()
    eval_env.close()

    return f"{out_model}.zip", out_vn


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage PPO training for Whoop drone"
    )
    parser.add_argument("--s1-steps",    type=int, default=5_000_000,
                        help="Stage 1 total timesteps (default 5M)")
    parser.add_argument("--s2-steps",    type=int, default=2_000_000,
                        help="Stage 2 total timesteps (default 2M)")
    parser.add_argument("--n-envs",      type=int, default=8,
                        help="Parallel environments (default 8)")
    parser.add_argument("--skip-stage1", default=None, metavar="ZIP",
                        help="Skip Stage 1, load this .zip as the base model")
    parser.add_argument("--only-stage1", action="store_true",
                        help="Run Stage 1 only (skip DR fine-tuning)")
    args = parser.parse_args()

    STAGE1["total_timesteps"] = args.s1_steps
    STAGE1["n_envs"]          = args.n_envs
    STAGE2["total_timesteps"] = args.s2_steps
    STAGE2["n_envs"]          = args.n_envs

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.skip_stage1:
        s1_zip = args.skip_stage1
        # Try to find the matching VecNormalize pkl alongside the checkpoint
        d      = os.path.dirname(os.path.abspath(s1_zip))
        s1_vn  = None
        for candidate in [
            os.path.join(d, "vec_normalize_stage1_base.pkl"),
            os.path.join(d, "vec_normalize.pkl"),
        ]:
            if os.path.exists(candidate):
                s1_vn = candidate
                print(f"[main] Found VecNorm → {s1_vn}")
                break
        if s1_vn is None:
            print("[main] WARNING: No VecNormalize pkl found for Stage 1 – "
                  "Stage 2 will start with fresh normalisation stats.")
        print(f"[main] Skipping Stage 1, using {s1_zip}")
    else:
        s1_zip, s1_vn = run_stage(STAGE1)

    if args.only_stage1:
        print("\n[main] --only-stage1 set. Done.")
        return

    # ── Stage 2 (DR fine-tune) ────────────────────────────────────────────────
    run_stage(STAGE2, resume_path=s1_zip, vec_norm_path=s1_vn)

    print("\n[main] Both stages complete.")
    print("[main] Deploy model: models/trained/stage2_dr_final.zip")
    print("[main] VecNorm:      models/trained/vec_normalize_stage2_dr.pkl")


if __name__ == "__main__":
    main()
