"""
PPO Training Script – Whoop Drone Hover Task
=============================================
Usage:
    python training/train.py                        # default config
    python training/train.py --config my_cfg.yaml   # custom config
    python training/train.py --resume models/trained/whoop_ppo_500000_steps.zip

TensorBoard logs are written to  logs/<run_name>/
Checkpoints are saved under      models/trained/
Best model is saved under        models/trained/best/
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from envs.drone_env import WhoopDroneEnv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_env(rank: int, seed: int = 0):
    """Factory function for a single training environment."""
    def _init():
        env = WhoopDroneEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def _make_eval_env(seed: int = 999):
    """Factory function for the evaluation environment."""
    def _init():
        env = WhoopDroneEnv()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def _default_config() -> dict:
    return {
        "training": {
            "n_envs":            8,
            "total_timesteps":   5_000_000,
            "log_dir":           "logs/whoop_hover",
            "save_dir":          "models/trained",
            "checkpoint_freq":   100_000,
            "eval_freq":         50_000,
            "seed":              42,
            "run_name":          "whoop_hover_ppo",
        },
        "ppo": {
            "policy":         "MlpPolicy",
            "learning_rate":  3e-4,
            "n_steps":        2048,
            "batch_size":     64,
            "n_epochs":       10,
            "gamma":          0.99,
            "gae_lambda":     0.95,
            "clip_range":     0.2,
            "ent_coef":       0.001,
            "vf_coef":        0.5,
            "max_grad_norm":  0.5,
            "net_arch_pi":    [256, 256],
            "net_arch_vf":    [256, 256],
        },
    }


def _load_config(path: str) -> dict:
    cfg = _default_config()
    if path and os.path.exists(path):
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        # Merge (user values override defaults)
        for section in ("training", "ppo"):
            if section in user_cfg:
                cfg[section].update(user_cfg[section])
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(config_path: str = None, resume_path: str = None):
    cfg       = _load_config(config_path)
    train_cfg = cfg["training"]
    ppo_cfg   = cfg["ppo"]

    n_envs    = train_cfg["n_envs"]
    seed      = train_cfg["seed"]
    log_dir   = train_cfg["log_dir"]
    save_dir  = train_cfg["save_dir"]
    run_name  = train_cfg["run_name"]

    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # ── Vectorised training environments ─────────────────────────────────────
    # Use SubprocVecEnv for true parallelism; fall back to DummyVecEnv if n_envs=1
    vec_cls = SubprocVecEnv if n_envs > 1 else None
    env = make_vec_env(
        _make_env(0, seed),
        n_envs=n_envs,
        vec_env_cls=vec_cls,
    )
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=ppo_cfg["gamma"],
    )

    # ── Evaluation environment (no reward normalisation) ─────────────────────
    eval_env = make_vec_env(_make_eval_env(seed + 9999), n_envs=1)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, train_cfg["checkpoint_freq"] // n_envs),
        save_path=save_dir,
        name_prefix=run_name,
        save_vecnormalize=True,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=log_dir,
        eval_freq=max(1, train_cfg["eval_freq"] // n_envs),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # ── PPO model ─────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(
            pi=ppo_cfg["net_arch_pi"],
            vf=ppo_cfg["net_arch_vf"],
        ),
        log_std_init=-1.5,  # std≈0.22 – keeps early throttle variation small
    )

    if resume_path:
        print(f"[train] Resuming from {resume_path}")
        model = PPO.load(
            resume_path,
            env=env,
            tensorboard_log=log_dir,
            verbose=1,
        )
    else:
        model = PPO(
            policy=ppo_cfg["policy"],
            env=env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
            seed=seed,
        )

    print("\n" + "=" * 60)
    print(f"  Whoop Drone – PPO Training")
    print(f"  n_envs          : {n_envs}")
    print(f"  total_timesteps : {train_cfg['total_timesteps']:,}")
    print(f"  hover_throttle  : {WhoopDroneEnv.HOVER_THROTTLE:.3f}")
    print(f"  control_freq    : {1.0 / (model.env.envs[0].model.opt.timestep * WhoopDroneEnv.N_SUBSTEPS):.0f} Hz"
          if hasattr(model.env, 'envs') else "")
    print("=" * 60 + "\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=True,
        reset_num_timesteps=(resume_path is None),
    )

    # ── Save final artefacts ──────────────────────────────────────────────────
    final_model_path = os.path.join(save_dir, f"{run_name}_final")
    vec_norm_path    = os.path.join(save_dir, "vec_normalize.pkl")
    model.save(final_model_path)
    env.save(vec_norm_path)

    print(f"\n[train] Done!  Model → {final_model_path}.zip")
    print(f"[train]        VecNorm → {vec_norm_path}")

    env.close()
    eval_env.close()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on WhoopDroneEnv")
    parser.add_argument("--config",  default=None, help="Path to YAML config")
    parser.add_argument("--resume",  default=None, help="Path to checkpoint .zip to resume from")
    args = parser.parse_args()
    train(config_path=args.config, resume_path=args.resume)
