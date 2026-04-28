"""
Notebook Generator – Whoop Drone Cloud Training
================================================
Run ONCE on your local machine to generate the Colab / Kaggle notebook:

    python notebooks/make_notebook.py

Output: notebooks/whoop_drone_cloud.ipynb

Upload that .ipynb to Google Colab or Kaggle, then run all cells.
The notebook is completely self-contained (all source files are embedded
as base64 so no repository clone is needed).

What the generated notebook does
----------------------------------
1.  Install Python packages (mujoco, gymnasium, stable-baselines3, …)
2.  Detect Colab vs Kaggle, create project directories
3.  Write whoop_drone.xml from embedded base64
4.  Write envs/drone_env.py    from embedded base64
5.  Write envs/drone_env_dr.py from embedded base64
6.  Stage 1 – PPO base hover policy     (WhoopDroneEnv,   5 M steps)
7.  Stage 2 – Domain-random fine-tune   (WhoopDroneEnvDR, 2 M steps)
8.  Headless evaluation (mean reward / distance-to-target)
9.  Package artefacts as a .zip and download / save
"""

import base64
import json
import os
import textwrap

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR  = os.path.dirname(os.path.abspath(__file__))


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


# ─────────────────────────────────────────────────────────────────────────────
# Embed project files as base64
# ─────────────────────────────────────────────────────────────────────────────
XML_B64    = _b64(_read("models/whoop_drone.xml"))
ENV_B64    = _b64(_read("envs/drone_env.py"))
DR_ENV_B64 = _b64(_read("envs/drone_env_dr.py"))


# ─────────────────────────────────────────────────────────────────────────────
# Notebook cell builders
# ─────────────────────────────────────────────────────────────────────────────

def _md(text: str) -> dict:
    return {
        "cell_type":  "markdown",
        "metadata":   {},
        "source":     text,
    }


def _code(text: str) -> dict:
    return {
        "cell_type":       "code",
        "execution_count": None,
        "metadata":        {},
        "outputs":         [],
        "source":          textwrap.dedent(text).lstrip("\n"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Individual cells
# ─────────────────────────────────────────────────────────────────────────────

CELL_TITLE = _md(
    "# Whoop Drone – PPO Training Pipeline (Colab / Kaggle)\n"
    "\n"
    "**Stage 1** – Base hover policy (clean simulation, 5 M steps)\n\n"
    "**Stage 2** – Domain-randomization fine-tuning (sim-to-real hardening, 2 M steps)\n\n"
    "> Run cells top-to-bottom. All source files are embedded; no git clone needed.\n"
    "> Artefacts are packaged and downloaded in the final cell.\n"
)


CELL_INSTALL = _code("""
    # ── 1. Install Python packages ─────────────────────────────────────────────
    import subprocess, sys, os, warnings

    # Silence the verbose CUDA / computation-placer warnings that Colab
    # prints at import time (harmless duplicate-registration messages from
    # pre-loaded TF/JAX).
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # PPO/MLP runs on CPU
    warnings.filterwarnings("ignore")

    # Remove the unmaintained legacy 'gym' package that ships with Colab/Kaggle
    # base images. It intercepts 'import gym' and triggers deprecation noise.
    subprocess.call(
        [sys.executable, "-m", "pip", "uninstall", "-q", "-y", "gym"],
        stderr=subprocess.DEVNULL,
    )

    _pkgs = [
        "mujoco>=3.1.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.3.0",
        "shimmy>=0.2.0",       # gym-gymnasium compatibility shim (SB3 needs it)
        "pyyaml",
        "rich",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + _pkgs)
    print("All packages installed.")
""")


CELL_SETUP = _code("""
    # ── 2. Detect Colab / Kaggle, create directories ──────────────────────────
    import os, sys

    try:
        import google.colab   # noqa
        _ON_COLAB  = True
    except ImportError:
        _ON_COLAB  = False

    _ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

    if _ON_COLAB:
        BASE_DIR = "/content/whoop_drone"
    elif _ON_KAGGLE:
        BASE_DIR = "/kaggle/working/whoop_drone"
    else:
        BASE_DIR = os.path.join(os.getcwd(), "whoop_drone")

    MODEL_DIR = os.path.join(BASE_DIR, "models")
    ENV_DIR   = os.path.join(BASE_DIR, "envs")
    LOG_DIR   = os.path.join(BASE_DIR, "logs")
    SAVE_DIR  = os.path.join(BASE_DIR, "models", "trained", "best")

    for _d in [MODEL_DIR, ENV_DIR, LOG_DIR, SAVE_DIR]:
        os.makedirs(_d, exist_ok=True)

    # Add project root to Python path so 'import envs' works
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    print(f"Platform : {'Colab' if _ON_COLAB else ('Kaggle' if _ON_KAGGLE else 'Other')}")
    print(f"BASE_DIR : {BASE_DIR}")
""")


# --------------------------------------------------------------------------- #
# XML cell – use the pre-computed base64 constant from this script             #
# --------------------------------------------------------------------------- #
CELL_WRITE_XML = _code(f"""
    # ── 3. Write whoop_drone.xml ────────────────────────────────────────────────
    import base64, os

    _XML_B64 = "{XML_B64}"

    _xml_path = os.path.join(MODEL_DIR, "whoop_drone.xml")
    with open(_xml_path, "wb") as _f:
        _f.write(base64.b64decode(_XML_B64))
    print(f"XML written → {{_xml_path}}")
""")


CELL_WRITE_ENV = _code(f"""
    # ── 4. Write envs/drone_env.py and envs/drone_env_dr.py ────────────────────
    import base64, os

    # drone_env.py  ─────────────────────────────────────────────────────────
    _ENV_B64 = "{ENV_B64}"
    _env_src = base64.b64decode(_ENV_B64).decode("utf-8")

    # Patch the XML path: the env locates the XML relative to its own __file__.
    # In the cloud the env lives at BASE_DIR/envs/drone_env.py, and the XML
    # is at BASE_DIR/models/whoop_drone.xml – the relative path "../models/…"
    # is already correct, so NO patch is needed.

    with open(os.path.join(ENV_DIR, "__init__.py"), "w") as _f:
        _f.write("from .drone_env import WhoopDroneEnv\\n"
                 "from .drone_env_dr import WhoopDroneEnvDR\\n")

    with open(os.path.join(ENV_DIR, "drone_env.py"), "w") as _f:
        _f.write(_env_src)

    # drone_env_dr.py  ──────────────────────────────────────────────────────
    _DR_B64 = "{DR_ENV_B64}"
    with open(os.path.join(ENV_DIR, "drone_env_dr.py"), "wb") as _f:
        _f.write(base64.b64decode(_DR_B64))

    print("Environment files written.")
""")


CELL_IMPORTS = _code("""
    # ── 5. Imports ──────────────────────────────────────────────────────────────
    import os, sys
    import numpy as np

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CallbackList, CheckpointCallback, EvalCallback,
    )
    from stable_baselines3.common.monitor  import Monitor
    from stable_baselines3.common.vec_env  import DummyVecEnv, VecNormalize

    from envs.drone_env    import WhoopDroneEnv
    from envs.drone_env_dr import WhoopDroneEnvDR

    print("Imports OK.")
    print(f"  WhoopDroneEnv    hover throttle : {WhoopDroneEnv.HOVER_THROTTLE:.3f}")
""")


# --------------------------------------------------------------------------- #
# Shared training helpers (inline in notebook)                                 #
# --------------------------------------------------------------------------- #
CELL_HELPERS = _code("""
    # ── 6. Training helper functions ────────────────────────────────────────────

    def _env_fn(EnvClass, rank, seed):
        def _init():
            env = EnvClass()
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init


    def build_train_env(EnvClass, n_envs, seed, vn_path=None):
        fns = [_env_fn(EnvClass, i, seed) for i in range(n_envs)]
        raw = DummyVecEnv(fns)   # DummyVecEnv avoids fork issues on Colab/Kaggle
        if vn_path and os.path.exists(vn_path):
            vn = VecNormalize.load(vn_path, raw)
            vn.training    = True
            vn.norm_reward = True
        else:
            vn = VecNormalize(raw, norm_obs=True, norm_reward=True,
                              clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        return vn


    def build_eval_env(EnvClass, seed, vn_path=None):
        raw = DummyVecEnv([_env_fn(EnvClass, 0, seed)])
        if vn_path and os.path.exists(vn_path):
            vn = VecNormalize.load(vn_path, raw)
            vn.training    = False
            vn.norm_reward = False
        else:
            vn = VecNormalize(raw, norm_obs=True, norm_reward=False,
                              clip_obs=10.0, training=False)
        return vn


    def run_stage(cfg, resume_path=None, vn_path=None):
        \"\"\"Run one training stage; returns (model_zip_path, vn_pkl_path).\"\"\"
        run_name = cfg["run_name"]
        n_envs   = cfg["n_envs"]
        seed     = cfg["seed"]
        save_dir = cfg["save_dir"]
        log_dir  = cfg["log_dir"]

        os.makedirs(os.path.join(save_dir, "best"), exist_ok=True)
        os.makedirs(os.path.join(log_dir,  run_name), exist_ok=True)

        train_env = build_train_env(cfg["EnvClass"], n_envs, seed, vn_path)
        eval_env  = build_eval_env (cfg["EnvClass"], seed + 9999, vn_path)

        ckpt_cb = CheckpointCallback(
            save_freq  = max(1, cfg["checkpoint_freq"] // n_envs),
            save_path  = save_dir,
            name_prefix= run_name,
            save_vecnormalize=True,
            verbose=1,
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path = os.path.join(save_dir, "best"),
            log_path             = os.path.join(log_dir, run_name),
            eval_freq            = max(1, cfg["eval_freq"] // n_envs),
            n_eval_episodes      = 10,
            deterministic        = True,
            render               = False,
            verbose              = 1,
        )

        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

        if resume_path and os.path.exists(resume_path):
            print(f"[{run_name}] Resuming from {resume_path}")
            model = PPO.load(
                resume_path, env=train_env,
                learning_rate = cfg["learning_rate"],
                clip_range    = cfg["clip_range"],
                ent_coef      = cfg["ent_coef"],
                n_epochs      = cfg["n_epochs"],
                tensorboard_log = log_dir, verbose=1,
            )
        else:
            model = PPO(
                "MlpPolicy", train_env,
                learning_rate  = cfg["learning_rate"],
                n_steps        = cfg["n_steps"],
                batch_size     = cfg["batch_size"],
                n_epochs       = cfg["n_epochs"],
                gamma=0.99, gae_lambda=0.95,
                clip_range     = cfg["clip_range"],
                ent_coef       = cfg["ent_coef"],
                vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs  = policy_kwargs,
                tensorboard_log= log_dir,
                verbose=1, seed=seed,
            )

        print(f"\\n{'='*55}")
        print(f"  {run_name.upper()}")
        print(f"  Env    : {cfg['EnvClass'].__name__}")
        print(f"  n_envs : {n_envs}    steps : {cfg['total_timesteps']:,}")
        print(f"{'='*55}\\n")

        model.learn(
            total_timesteps     = cfg["total_timesteps"],
            callback            = CallbackList([ckpt_cb, eval_cb]),
            tb_log_name         = run_name,
            progress_bar        = True,
            reset_num_timesteps = (resume_path is None),
        )

        out_zip = os.path.join(save_dir, f"{run_name}_final")
        out_vn  = os.path.join(save_dir, f"vec_normalize_{run_name}.pkl")
        model.save(out_zip)
        train_env.save(out_vn)

        print(f"[{run_name}] Saved → {out_zip}.zip")
        train_env.close()
        eval_env.close()
        return f"{out_zip}.zip", out_vn


    print("Helper functions defined.")
""")


CELL_STAGE1_CFG = _code("""
    # ── 7a. Stage 1 configuration ───────────────────────────────────────────────
    # Adjust n_envs to the number of CPU cores available on the cloud instance.
    # Colab free tier : 2 vCPU  → n_envs = 2
    # Colab Pro/Pro+  : 8 vCPU  → n_envs = 8
    # Kaggle          : 4 vCPU  → n_envs = 4

    STAGE1_CFG = dict(
        run_name        = "stage1_base",
        EnvClass        = WhoopDroneEnv,
        total_timesteps = 5_000_000,
        n_envs          = 2,           # ← change to match your vCPU count
        seed            = 42,
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 10,
        clip_range      = 0.2,
        ent_coef        = 0.001,
        log_dir         = os.path.join(BASE_DIR, "logs"),
        save_dir        = os.path.join(BASE_DIR, "models", "trained"),
        checkpoint_freq = 100_000,
        eval_freq       = 50_000,
    )
    print("Stage 1 config ready.")
""")


CELL_STAGE1_RUN = _code("""
    # ── 7b. Stage 1 – Run base training ────────────────────────────────────────
    s1_zip, s1_vn = run_stage(STAGE1_CFG)
    print(f"\\nStage 1 complete.\\n  model : {s1_zip}\\n  vn    : {s1_vn}")
""")


CELL_STAGE2_CFG = _code("""
    # ── 8a. Stage 2 configuration (Domain Randomization) ───────────────────────
    STAGE2_CFG = dict(
        run_name        = "stage2_dr",
        EnvClass        = WhoopDroneEnvDR,
        total_timesteps = 2_000_000,
        n_envs          = 2,           # ← same as Stage 1
        seed            = 123,
        learning_rate   = 1e-4,        # lower LR for fine-tuning
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 5,           # fewer epochs → less catastrophic forgetting
        clip_range      = 0.15,        # tighter clip → conservative updates
        ent_coef        = 0.0005,
        log_dir         = os.path.join(BASE_DIR, "logs"),
        save_dir        = os.path.join(BASE_DIR, "models", "trained"),
        checkpoint_freq = 50_000,
        eval_freq       = 25_000,
    )
    print("Stage 2 config ready.")
""")


CELL_STAGE2_RUN = _code("""
    # ── 8b. Stage 2 – Domain-randomization fine-tune ───────────────────────────
    s2_zip, s2_vn = run_stage(STAGE2_CFG, resume_path=s1_zip, vn_path=s1_vn)
    print(f"\\nStage 2 complete.\\n  model : {s2_zip}\\n  vn    : {s2_vn}")
""")


CELL_EVAL = _code("""
    # ── 9. Headless evaluation ──────────────────────────────────────────────────
    N_EVAL_EPS = 10

    _eval_env = build_eval_env(WhoopDroneEnvDR, seed=777, vn_path=s2_vn)
    _model    = PPO.load(s2_zip, device="cpu")

    ep_rewards, ep_lengths, ep_dists = [], [], []

    for _ep in range(N_EVAL_EPS):
        _obs  = _eval_env.reset()
        _done = False
        _r, _s, _d = 0.0, 0, []

        while not _done:
            _act, _ = _model.predict(_obs, deterministic=True)
            _obs, _rew, _done_arr, _info_arr = _eval_env.step(_act)
            _r   += float(_rew[0])
            _s   += 1
            _done = bool(_done_arr[0])
            if "distance_to_target" in _info_arr[0]:
                _d.append(_info_arr[0]["distance_to_target"])

        ep_rewards.append(_r)
        ep_lengths.append(_s)
        ep_dists.append(float(np.mean(_d)) if _d else float("nan"))
        print(f"  Ep {_ep+1:2d}:  reward={_r:8.1f}  steps={_s:5d}  "
              f"dist={ep_dists[-1]:.3f} m")

    print("─" * 50)
    print(f"  Mean reward : {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print(f"  Mean steps  : {np.mean(ep_lengths):.0f}")
    print(f"  Mean dist   : {np.nanmean(ep_dists):.3f} m")

    _eval_env.close()
""")


CELL_DOWNLOAD = _code("""
    # ── 10. Package artefacts and download ──────────────────────────────────────
    import zipfile, shutil

    _archive_name = os.path.join(BASE_DIR, "whoop_drone_trained.zip")

    with zipfile.ZipFile(_archive_name, "w", zipfile.ZIP_DEFLATED) as _zf:
        for _fpath in [s1_zip, s1_vn, s2_zip, s2_vn]:
            if _fpath and os.path.exists(_fpath):
                _zf.write(_fpath, os.path.basename(_fpath))
        # Also include best model if present
        _best = os.path.join(BASE_DIR, "models", "trained", "best", "best_model.zip")
        if os.path.exists(_best):
            _zf.write(_best, "best_model.zip")
        # Include best VecNorm if present
        _best_vn = os.path.join(BASE_DIR, "models", "trained", "best", "vec_normalize.pkl")
        if os.path.exists(_best_vn):
            _zf.write(_best_vn, "best_vec_normalize.pkl")

    print(f"Archive created: {_archive_name}")
    print(f"Size: {os.path.getsize(_archive_name) / 1024 / 1024:.1f} MB")

    # ── Download ─────────────────────────────────────────────────────────────
    try:
        from google.colab import files
        files.download(_archive_name)
        print("Download started (Colab).")
    except ImportError:
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            print(f"Kaggle: find the archive in the Output tab: {_archive_name}")
        else:
            print(f"File saved locally at: {_archive_name}")
""")


# ─────────────────────────────────────────────────────────────────────────────
# Assemble notebook
# ─────────────────────────────────────────────────────────────────────────────

NOTEBOOK = {
    "nbformat":       4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language":     "python",
            "name":         "python3",
        },
        "language_info": {
            "name":    "python",
            "version": "3.10.0",
        },
    },
    "cells": [
        CELL_TITLE,
        CELL_INSTALL,
        CELL_SETUP,
        CELL_WRITE_XML,
        CELL_WRITE_ENV,
        CELL_IMPORTS,
        CELL_HELPERS,
        CELL_STAGE1_CFG,
        CELL_STAGE1_RUN,
        CELL_STAGE2_CFG,
        CELL_STAGE2_RUN,
        CELL_EVAL,
        CELL_DOWNLOAD,
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Write notebook
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_path = os.path.join(NB_DIR, "whoop_drone_cloud.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(NOTEBOOK, f, indent=1, ensure_ascii=True)
    print(f"Notebook written → {out_path}")
    print()
    print("Next steps:")
    print("  1. Upload notebooks/whoop_drone_cloud.ipynb to Colab or Kaggle")
    print("  2. Set n_envs in cells 7a/8a to match the platform's vCPU count")
    print("  3. Run all cells  (Runtime → Run all)")
    print("  4. The final cell downloads  whoop_drone_trained.zip")
    print()
    print("  For Colab GPU: PPO with MlpPolicy runs faster on CPU;")
    print("  the GPU is unused here. Set device='cpu' (already the default).")
