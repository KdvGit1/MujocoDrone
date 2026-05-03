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
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.drone_env        import WhoopDroneEnv
from envs.drone_env_dr     import WhoopDroneEnvDR
from envs.drone_env_dr_nav import WhoopDroneEnvDRNav


class SyncVecNormCallback(BaseCallback):
    """Sync eval-env VecNormalize stats from the training env before each eval."""

    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env  = eval_env

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self) -> None:
        """Copy running mean/var from train_env to eval_env."""
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms


class SaveBestVecNormCallback(BaseCallback):
    """
    Whenever EvalCallback saves a new best_model.zip, immediately save the
    training VecNormalize stats alongside it as vec_normalize.pkl.
    This keeps the model and its normalisation stats in sync so the next
    stage always starts from the peak checkpoint, not the final (possibly
    collapsed) one.
    """

    def __init__(
        self,
        eval_cb:    EvalCallback,
        train_env:  VecNormalize,
        save_path:  str,          # full path, e.g. models/trained/best/vec_normalize.pkl
        verbose:    int = 0,
    ):
        super().__init__(verbose)
        self.eval_cb   = eval_cb
        self.train_env = train_env
        self.save_path = save_path
        self._last_best = -float("inf")

    def _on_step(self) -> bool:
        current_best = self.eval_cb.best_mean_reward
        if current_best > self._last_best:
            self._last_best = current_best
            self.train_env.save(self.save_path)
            if self.verbose:
                print(f"[SaveBestVecNorm] New best ({current_best:.2f}) "
                      f"→ saved VecNorm to {self.save_path}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Default stage configurations
# ─────────────────────────────────────────────────────────────────────────────

STAGE1 = dict(
    run_name         = "stage1_base",
    EnvClass         = WhoopDroneEnv,
    total_timesteps  = 5_000_000,
    n_envs           = 12,
    seed             = 42,
    learning_rate    = 3e-4,
    n_steps          = 2048,
    batch_size       = 64,
    n_epochs         = 10,
    clip_range       = 0.2,
    ent_coef         = 0.001,
    log_dir          = "logs",
    save_dir         = "models/trained",
    checkpoint_freq  = 250_000,
    eval_freq        = 200_000,
)

STAGE2 = dict(
    run_name         = "stage2_dr",
    EnvClass         = WhoopDroneEnvDR,
    total_timesteps  = 2_000_000,
    n_envs           = 12,
    seed             = 123,
    learning_rate    = 1e-4,    # lower LR for fine-tuning
    n_steps          = 2048,
    batch_size       = 64,
    n_epochs         = 5,       # fewer epochs → less forgetting
    clip_range       = 0.15,    # tighter clip → conservative updates
    ent_coef         = 0.0005,
    log_dir          = "logs",
    save_dir         = "models/trained",
    checkpoint_freq  = 200_000,
    eval_freq        = 100_000,
)

STAGE3 = dict(
    run_name         = "stage3_nav",
    EnvClass         = WhoopDroneEnvDRNav,
    total_timesteps  = 10_000_000,
    n_envs           = 12,
    seed             = 456,
    learning_rate    = 5e-5,    # very conservative – preserve DR skills
    n_steps          = 2048,
    batch_size       = 64,
    n_epochs         = 5,
    clip_range       = 0.10,    # tight clip – do not overwrite hover/DR knowledge
    ent_coef         = 0.0005,
    log_dir          = "logs",
    save_dir         = "models/trained",
    checkpoint_freq  = 200_000,
    eval_freq        = 100_000,
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
    """Create eval environment (reward NOT normalised).

    The eval env shares the same normalisation statistics as the training env
    by loading the same pkl file.  For Stage 2 this means it uses the Stage 1
    stats as a warm-start (same as the training env), so both see the same
    normalisation and EvalCallback scores are meaningful.
    """
    raw = DummyVecEnv([_env_fn(EnvClass, 0, seed)])

    if vec_norm_path and os.path.exists(vec_norm_path):
        vn = VecNormalize.load(vec_norm_path, raw)
        vn.training    = False   # eval env never updates stats
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

def run_stage(cfg: dict, resume_path: str = None, vec_norm_path: str = None,
              extra_callbacks: list = None):
    """
    Train one stage of the pipeline.

    Parameters
    ----------
    cfg              : stage configuration dict (STAGE1 or STAGE2)
    resume_path      : path to a .zip checkpoint to resume from
    vec_norm_path    : path to a VecNormalize .pkl to warm-start normalisation stats
    extra_callbacks  : optional list of additional SB3 callbacks (e.g. LivePlotCallback)

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

    best_dir    = os.path.join(save_dir, "best")
    best_vn_path = os.path.join(best_dir, "vec_normalize.pkl")
    os.makedirs(best_dir, exist_ok=True)

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
        best_model_save_path=best_dir,
        log_path=os.path.join(log_dir, run_name),
        eval_freq=max(1, cfg["eval_freq"] // n_envs),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    sync_cb      = SyncVecNormCallback(train_env, eval_env)
    best_vn_cb   = SaveBestVecNormCallback(eval_cb, train_env, best_vn_path, verbose=1)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                         log_std_init=-1.5)   # std≈0.22 – keeps early throttle variation small

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
            device="cpu",
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
            device         = "cpu",
        )

    print(f"\n{'═'*58}")
    print(f"  {run_name.upper()}")
    print(f"  Env       : {cfg['EnvClass'].__name__}")
    print(f"  n_envs    : {n_envs}")
    print(f"  steps     : {cfg['total_timesteps']:,}")
    print(f"  lr        : {cfg['learning_rate']}")
    print(f"{'═'*58}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    _cb_list = [ckpt_cb, sync_cb, eval_cb, best_vn_cb] + (extra_callbacks or [])
    model.learn(
        total_timesteps     = cfg["total_timesteps"],
        callback            = CallbackList(_cb_list),
        tb_log_name         = run_name,
        progress_bar        = True,
        reset_num_timesteps = (resume_path is None),
    )

    # ── Save final artefacts (archive / fallback) ──────────────────────────────
    final_model = os.path.join(save_dir, f"{run_name}_final")
    final_vn    = os.path.join(save_dir, f"vec_normalize_{run_name}.pkl")
    model.save(final_model)
    train_env.save(final_vn)

    # ── Determine which model to hand to the next stage ────────────────────────
    # Prefer best_model.zip (peak performance) over _final.zip (may have collapsed)
    best_model_zip = os.path.join(best_dir, "best_model.zip")
    if os.path.exists(best_model_zip) and os.path.exists(best_vn_path):
        out_model = best_model_zip
        out_vn    = best_vn_path
        print(f"\n[{run_name}] ✓ Best model → {out_model}")
        print(f"[{run_name}] ✓ Best VecNorm→ {out_vn}")
    else:
        # Fallback: best_model.zip exists but VecNorm was never saved (e.g. no
        # improvement was recorded).  Use final artefacts.
        out_model = f"{final_model}.zip"
        out_vn    = final_vn
        print(f"\n[{run_name}] (no best checkpoint found – using final)")
        print(f"[{run_name}] Model  → {out_model}")
        print(f"[{run_name}] VecNorm→ {out_vn}")

    train_env.close()
    eval_env.close()

    return out_model, out_vn


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _find_vn(save_dir: str, run_name: str) -> str:
    """Return vec_normalize pkl path for a given run, or None."""
    for candidate in [
        os.path.join(save_dir, f"vec_normalize_{run_name}.pkl"),
        os.path.join(save_dir, "vec_normalize.pkl"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Three-stage PPO training for Whoop drone"
    )
    parser.add_argument("--s1-steps",    type=int, default=5_000_000,
                        help="Stage 1 total timesteps (default 5M)")
    parser.add_argument("--s2-steps",    type=int, default=2_000_000,
                        help="Stage 2 total timesteps (default 2M)")
    parser.add_argument("--s3-steps",    type=int, default=10_000_000,
                        help="Stage 3 total timesteps (default 2M)")
    parser.add_argument("--n-envs",      type=int, default=8,
                        help="Parallel environments (default 8)")
    parser.add_argument("--skip-stage1", default=None, metavar="ZIP",
                        help="Skip Stage 1, provide .zip path")
    parser.add_argument("--skip-stage2", default=None, metavar="ZIP",
                        help="Skip Stage 1+2, provide Stage-2 .zip path for Stage 3")
    parser.add_argument("--only-stage1", action="store_true",
                        help="Run Stage 1 only")
    parser.add_argument("--only-stage2", action="store_true",
                        help="Run Stage 1 + Stage 2 only (skip Stage 3)")
    args = parser.parse_args()

    STAGE1["total_timesteps"] = args.s1_steps
    STAGE1["n_envs"]          = args.n_envs
    STAGE2["total_timesteps"] = args.s2_steps
    STAGE2["n_envs"]          = args.n_envs
    STAGE3["total_timesteps"] = args.s3_steps
    STAGE3["n_envs"]          = args.n_envs

    save_dir = STAGE1["save_dir"]

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.skip_stage2:
        # Skipping both S1 and S2 – go straight to Stage 3
        s2_zip = args.skip_stage2
        s2_vn  = _find_vn(os.path.dirname(os.path.abspath(s2_zip)), "stage2_dr")
        if s2_vn is None:
            s2_vn = _find_vn(os.path.dirname(os.path.abspath(s2_zip)), "stage2_dr")
        print(f"[main] Skipping Stage 1+2, using {s2_zip}")
        if s2_vn:
            print(f"[main] Found VecNorm → {s2_vn}")
        else:
            print("[main] WARNING: No VecNormalize pkl found for Stage 2.")
        s1_zip = s1_vn = None  # not needed
    elif args.skip_stage1:
        s1_zip = args.skip_stage1
        s1_vn  = _find_vn(os.path.dirname(os.path.abspath(s1_zip)), "stage1_base")
        if s1_vn is None:
            s1_vn = _find_vn(os.path.dirname(os.path.abspath(s1_zip)), "")
        print(f"[main] Skipping Stage 1, using {s1_zip}")
        if s1_vn:
            print(f"[main] Found VecNorm → {s1_vn}")
        else:
            print("[main] WARNING: No VecNormalize pkl found for Stage 1.")
        s2_zip = s2_vn = None  # will be set after S2
    else:
        s1_zip, s1_vn = run_stage(STAGE1)
        s2_zip = s2_vn = None

    if args.only_stage1:
        print("\n[main] --only-stage1 set. Done.")
        return

    # ── Stage 2 (DR fine-tune) ────────────────────────────────────────────────
    if not args.skip_stage2:
        s2_zip, s2_vn = run_stage(STAGE2, resume_path=s1_zip, vec_norm_path=s1_vn)

    if args.only_stage2:
        print("\n[main] --only-stage2 set. Done.")
        print(f"[main] Deploy model : {s2_zip}")
        print(f"[main] VecNorm      : {s2_vn}")
        return

    # ── Stage 3 (DR + Navigation challenge) ──────────────────────────────────
    s3_zip, s3_vn = run_stage(STAGE3, resume_path=s2_zip, vec_norm_path=s2_vn)

    print("\n[main] All three stages complete.")
    print(f"[main] Deploy model : {s3_zip}")
    print(f"[main] VecNorm      : {s3_vn}")


if __name__ == "__main__":
    main()
