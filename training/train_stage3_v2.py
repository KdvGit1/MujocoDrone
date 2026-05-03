"""
Stage 3 v2 – Infinite nav training with live reward/length plot
===============================================================
Changes vs original Stage 3:
  • Reward: flip penalty -100 (was 0), linear pos reward (was exp), ±45° init yaw
  • Runs INDEFINITELY – Ctrl+C veya pencereyi kapatarak durdur
  • Canlı grafik: ep_rew_mean ve ep_len_mean penceresi

Usage:
    python training/train_stage3_v2.py

Outputs:
    models/trained/stage3_v2_final.zip        (Ctrl+C anında kaydedilir)
    models/trained/vec_normalize_stage3_v2.pkl
    models/trained/best/best_model.zip        (en iyi eval skoru her zaman güncel)
    logs/stage3_v2/
"""

import os
import sys

# Fix: multiple OpenMP runtimes (libomp + libiomp5) in conda envs with PyTorch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Matplotlib live plot ──────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    _PLOT_OK = True
except Exception:
    _PLOT_OK = False

from stable_baselines3.common.callbacks import BaseCallback

from training.train_full import run_stage
from envs.drone_env_dr_nav import WhoopDroneEnvDRNav

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(ROOT, "models", "trained")

# ── Resume checkpoint ─────────────────────────────────────────────────────────
# Latest stage3_v2 checkpoint (132.4M steps) – reverted from bad 147M run
_LATEST      = os.path.join(SAVE_DIR, "stage3_v2_132395208_steps.zip")
_LATEST_VN   = os.path.join(SAVE_DIR, "stage3_v2_vecnormalize_132395208_steps.pkl")
_BEST        = os.path.join(SAVE_DIR, "best", "best_model.zip")
_FINAL       = os.path.join(SAVE_DIR, "stage3_nav_final.zip")
RESUME_MODEL = _LATEST if os.path.exists(_LATEST) else (_BEST if os.path.exists(_BEST) else _FINAL)

_BEST_VN  = os.path.join(SAVE_DIR, "best", "vec_normalize.pkl")
_FINAL_VN = os.path.join(SAVE_DIR, "vec_normalize_stage3_nav.pkl")
RESUME_VN = _LATEST_VN if os.path.exists(_LATEST_VN) else (_BEST_VN if os.path.exists(_BEST_VN) else _FINAL_VN)


# ── Live plot callback ────────────────────────────────────────────────────────

class LivePlotCallback(BaseCallback):
    """
    Her `update_calls` adımda canlı grafik günceller.
    update_calls=500 → 12 env × 500 = 6000 env adımı ≈ her ~0.3 rollout.
    """

    def __init__(self, update_calls: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self._upd   = update_calls
        self._steps: list = []
        self._rew:   list = []
        self._len:   list = []
        self._fig   = None

    # ── rolling smoothing helper ──────────────────────────────────────────────
    @staticmethod
    def _smooth(arr, w=20):
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    def _on_training_start(self) -> None:
        if not _PLOT_OK:
            print("[LivePlot] matplotlib/TkAgg bulunamadı – grafik devre dışı.")
            return
        plt.ion()
        self._fig, (self._ax_r, self._ax_l) = plt.subplots(
            2, 1, figsize=(13, 7), sharex=True
        )
        self._fig.suptitle("Stage 3 v2 – Canlı Eğitim Grafiği", fontsize=13)
        for ax in (self._ax_r, self._ax_l):
            ax.grid(True, alpha=0.25)
        self._ax_r.set_ylabel("ep_rew_mean")
        self._ax_l.set_ylabel("ep_len_mean")
        self._ax_l.set_xlabel("Environment steps")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def _on_step(self) -> bool:
        if not _PLOT_OK or self._fig is None:
            return True
        if self.n_calls % self._upd != 0:
            return True
        if len(self.model.ep_info_buffer) == 0:
            return True

        buf = self.model.ep_info_buffer
        ep_rew = float(np.mean([ep["r"] for ep in buf]))
        ep_len = float(np.mean([ep["l"] for ep in buf]))

        self._steps.append(self.num_timesteps)
        self._rew.append(ep_rew)
        self._len.append(ep_len)

        xs = np.array(self._steps)
        rs = np.array(self._rew)
        ls = np.array(self._len)
        xs_s = xs[19:] if len(xs) >= 20 else xs   # aligned with smooth output

        self._ax_r.clear()
        self._ax_l.clear()

        # Raw (faint) + smoothed (bold)
        self._ax_r.plot(xs, rs, color="royalblue",  alpha=0.25, linewidth=0.8)
        self._ax_r.plot(xs_s, self._smooth(rs), color="royalblue", linewidth=1.8,
                        label=f"smooth  latest={ep_rew:.1f}")
        self._ax_r.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        self._ax_r.set_ylabel("ep_rew_mean")
        self._ax_r.legend(fontsize=8, loc="upper left")
        self._ax_r.grid(True, alpha=0.25)

        self._ax_l.plot(xs, ls, color="darkorange", alpha=0.25, linewidth=0.8)
        self._ax_l.plot(xs_s, self._smooth(ls), color="darkorange", linewidth=1.8,
                        label=f"smooth  latest={ep_len:.0f}")
        self._ax_l.axhline(2000, color="green", linewidth=0.8, linestyle="--",
                           label="max (2000 steps)")
        self._ax_l.set_ylabel("ep_len_mean")
        self._ax_l.set_xlabel("Environment steps")
        self._ax_l.legend(fontsize=8, loc="upper left")
        self._ax_l.grid(True, alpha=0.25)

        self._fig.suptitle(
            f"Stage 3 v2 – {self.num_timesteps:,} steps  |  "
            f"rew={ep_rew:.1f}  len={ep_len:.0f}",
            fontsize=12,
        )
        plt.tight_layout()
        try:
            plt.pause(0.001)
        except Exception:
            pass
        return True


# ── Stage config  (total_timesteps = sonsuz) ─────────────────────────────────

STAGE3_V2 = dict(
    run_name         = "stage3_v2",
    EnvClass         = WhoopDroneEnvDRNav,
    total_timesteps  = int(1e12),   # sonsuz – Ctrl+C ile durdur
    n_envs           = 12,
    seed             = 789,
    learning_rate    = 3e-5,
    n_steps          = 2048,
    batch_size       = 64,
    n_epochs         = 5,
    clip_range       = 0.12,
    ent_coef         = 0.001,
    log_dir          = os.path.join(ROOT, "logs"),
    save_dir         = SAVE_DIR,
    checkpoint_freq  = 200_000,
    eval_freq        = 100_000,
)

if __name__ == "__main__":
    print(f"\n{'═'*60}")
    print(f"  STAGE 3 v2  –  SONSUZ fine-tune (Ctrl+C ile durdur)")
    print(f"  Resume model  : {RESUME_MODEL}")
    print(f"  Resume VecNorm: {RESUME_VN}")
    print(f"  Reward/Nav düzeltmeleri (v2.1):")
    print(f"    + YAW_REWARD_SCALE 1.0 → 0.5  (pos reward artık dominant)")
    print(f"    + NAV_CMD_INTERVAL 150 → 300  (hedefe ulaşmaya zaman var)")
    print(f"    + NAV_CMD_RADIUS   1.5 → 0.8  (küçük hedeflerden başla)")
    print(f"    + NAV_SETTLE_DIST  0.20 → 0.30 (biraz toleranslı)")
    print(f"    + lr 5e-5 → 3e-5  (clip_fraction baskısı)")
    print(f"    + ent_coef 3e-3 → 1e-3  (132M'den revert, dengeli keşif)")
    print(f"  Checkpointlar : her 200k adımda otomatik kaydedilir")
    print(f"  Best model    : her eval'da güncellenir")
    print(f"{'═'*60}\n")

    live_plot = LivePlotCallback(update_calls=500)

    try:
        out_model, out_vn = run_stage(
            cfg              = STAGE3_V2,
            resume_path      = RESUME_MODEL,
            vec_norm_path    = RESUME_VN,
            extra_callbacks  = [live_plot],
        )
    except KeyboardInterrupt:
        print("\n[train] Ctrl+C alındı – son checkpoint otomatik kaydedildi.")
        print("[train] Benchmark için en son stage3_v2_*.zip dosyasını kullan.")
        sys.exit(0)

    print(f"\n{'═'*60}")
    print(f"  DONE")
    print(f"  Final model  : {out_model}")
    print(f"  Final VecNorm: {out_vn}")
    print(f"  Benchmark:  python benchmark.py --model {out_model} --vec-norm {out_vn}")
    print(f"{'═'*60}\n")
