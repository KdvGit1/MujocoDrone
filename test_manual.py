"""
test_manual.py  -  Model otopilot, klavye hedef belirler
=========================================================
Egitilmis PPO modeli drone'u otomatik ucurur.
Sen klavye ile HEDEF NOKTAYI degistirirsin, model oraya gider.

Klavye:
    ↑ / ↓        : Hedefi ileri / geri  (X ekseni)
    ← / →        : Hedefi sola / saga   (Y ekseni)
    Q / E        : Yukari / asagi       (Z ekseni)
    Z / C        : Yaw hedefini CCW / CW
    R            : Hedefi sifirla [0, 0, 1]
    H            : Yardim
    ESC / X      : Cik

Kullanim:
    python test_manual.py
    python test_manual.py --model models/trained/stage3_v2_132395208_steps.zip
                          --vec-norm models/trained/stage3_v2_vecnormalize_132395208_steps.pkl
    python test_manual.py --dr        # domain randomization ile
"""

import os
import sys
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pynput import keyboard as kb
    from pynput.keyboard import Key
    _PYNPUT_OK = True
except ImportError:
    _PYNPUT_OK = False

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.drone_env_dr_nav import WhoopDroneEnvDRNav
from envs.drone_env import WhoopDroneEnv

ROOT             = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL    = os.path.join(ROOT, "models", "trained", "best", "best_model.zip")
DEFAULT_VEC_NORM = os.path.join(ROOT, "models", "trained", "best", "vec_normalize.pkl")

os.system("")
_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"; _C = "\033[96m"; _B = "\033[1m"; _X = "\033[0m"

# Paylasilan durum
_final_target = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # klavye tarafindan set edilir
_target       = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # carrot: _final_target'e dogru ilerler
_target_yaw   = 0.0
_held         = set()
_quit         = False

POS_STEP    = 0.15
YAW_STEP    = 0.10
CARROT_STEP = 0.04   # m / adim — 50 Hz'de max ~2 m/s

_MOVE_KEYS = {Key.up, Key.down, Key.left, Key.right, 'q', 'e'}


def _on_press(key):
    global _quit, _target_yaw
    try:
        ch = key.char.lower()
        _held.add(ch)
        if ch == 'r':
            _final_target[:] = [0.0, 0.0, 1.0]
            _target[:] = [0.0, 0.0, 1.0]
            _target_yaw = 0.0
            print(f"  {_Y}[R] Hedef sifirlandi -> [0, 0, 1]{_X}")
        if ch == 'x': _quit = True
        if ch == 'h': _print_help()
    except AttributeError:
        _held.add(key)
        if key == Key.esc: _quit = True


def _on_release(key):
    try:   _held.discard(key.char.lower())
    except AttributeError: _held.discard(key)


def _print_help():
    print(f"\n{_B}{'─'*54}{_X}")
    print(f"{_B}  KLAVYE  ->  HEDEF DEGISTIR  (model kendisi ucar){_X}")
    print(f"  {_C}↑/↓{_X}     : Hedef ileri / geri   (X)")
    print(f"  {_C}←/→{_X}     : Hedef sol  / sag     (Y)")
    print(f"  {_C}Q{_X} / {_C}E{_X}   : Hedef yukari / asagi (Z)")
    print(f"  {_C}Z{_X} / {_C}C{_X}   : Yaw CCW / CW")
    print(f"  {_C}R{_X}       : Hedefi [0,0,1] sifirla")
    print(f"  {_C}ESC{_X}/{_C}X{_X}   : Cik")
    print(f"  Adim: {POS_STEP} m  |  {np.degrees(YAW_STEP):.1f} deg")
    print(f"{_B}{'─'*54}{_X}\n")


def _update_target():
    """Klavye tuslarini _final_target'e uygular."""
    global _target_yaw
    if Key.up    in _held: _final_target[0] += POS_STEP
    if Key.down  in _held: _final_target[0] -= POS_STEP
    if Key.left  in _held: _final_target[1] += POS_STEP
    if Key.right in _held: _final_target[1] -= POS_STEP
    if 'q' in _held: _final_target[2] += POS_STEP
    if 'e' in _held: _final_target[2] -= POS_STEP
    if 'z' in _held: _target_yaw -= YAW_STEP
    if 'c' in _held: _target_yaw += YAW_STEP
    _final_target[0]  = float(np.clip(_final_target[0], -4.0,  4.0))
    _final_target[1]  = float(np.clip(_final_target[1], -4.0,  4.0))
    _final_target[2]  = float(np.clip(_final_target[2],  0.3,  4.0))
    _target_yaw = float(np.arctan2(np.sin(_target_yaw), np.cos(_target_yaw)))


def _advance_carrot():
    """Hareket tusu basılıyken _target'i _final_target'e dogru ilerlet.
    Tus bırakıldığında _final_target = _target konumuna snap'lenir (donma)."""
    if not (_held & _MOVE_KEYS):
        # Hic hareket tusu yok: final_target'i mevcut carrot konumuna cek
        _final_target[:] = _target
        return
    delta = _final_target - _target
    dist  = float(np.linalg.norm(delta))
    if dist <= CARROT_STEP:
        _target[:] = _final_target
    else:
        _target[:] += (delta / dist) * CARROT_STEP


def _dist_color(d):
    if d < 0.40: return _G
    if d < 1.00: return _Y
    return _R


def _setup_tracking_camera(env, distance: float = 2.5):
    """MuJoCo viewer kamerasini drone'u takip edecek sekilde ayarla."""
    import mujoco
    v = env._viewer
    if v is None:
        return
    v.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = env._drone_body_id
    v.cam.distance    = distance
    v.cam.elevation   = -20.0   # hafif yukari bak
    v.cam.azimuth     = 135.0   # arka-sol kose


def run_manual(model_path, vec_norm_path, use_dr, max_steps, seed, fps):
    global _quit, _target_yaw

    if not _PYNPUT_OK:
        print(f"{_R}HATA: pynput kurulu degil.  pip install pynput{_X}")
        sys.exit(1)

    EnvClass = WhoopDroneEnvDRNav if use_dr else WhoopDroneEnv
    step_dt  = 1.0 / fps

    _dummy   = DummyVecEnv([lambda: EnvClass()])
    vec_norm = VecNormalize.load(vec_norm_path, _dummy)
    vec_norm.training    = False
    vec_norm.norm_reward = False
    model = PPO.load(model_path, device="cpu")

    print(f"\n{'='*58}")
    print(f"{_B}  MODEL OTOPILOT  +  KLAVYE HEDEF  --  MuJoCo Viewer{_X}")
    print(f"  Model   : {os.path.basename(model_path)}")
    print(f"  VecNorm : {os.path.basename(vec_norm_path)}")
    print(f"  Env     : {EnvClass.__name__}")
    print(f"  FPS     : {fps}  |  max steps: {max_steps}  |  sonsuz mod")
    _print_help()

    listener = kb.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    env      = EnvClass(render_mode="human", max_episode_steps=max_steps)
    cam_set  = False   # kamera ilk render'dan sonra bir kere ayarlanir
    ep       = 0

    while not _quit:
        ep += 1
        _final_target[:] = [0.0, 0.0, 1.0]
        _target[:] = [0.0, 0.0, 1.0]
        _target_yaw = 0.0

        obs, _ = env.reset(seed=seed + ep)
        env.target_pos = _target.copy()
        env.target_yaw = _target_yaw

        done    = False
        step    = 0
        total_r = 0.0
        t0      = time.time()

        print(f"\n{_B}  -- Episode {ep} --{_X}")
        print(f"  {'Adim':>5}  {'Dist':>7}  {'Z':>6}  {'Hiz':>6}  {'Rew':>7}  Hedef")

        while not done and not _quit:
            _update_target()
            _advance_carrot()
            env.target_pos = _target.copy()
            env.target_yaw = _target_yaw

            obs            = env._get_obs()
            norm_obs       = vec_norm.normalize_obs(obs[np.newaxis, :])
            action, _state = model.predict(norm_obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action[0])

            total_r += float(rew)
            step    += 1
            done     = terminated or truncated

            if step % 5 == 0 or done:
                pos  = env._get_pos()
                vel  = float(np.linalg.norm(env._get_vel()))
                dist = float(info["distance_to_target"])
                dc   = _dist_color(dist)
                tgt_str = (f"C[{_target[0]:+.2f},{_target[1]:+.2f},{_target[2]:.2f}]"
                           f" G[{_final_target[0]:+.2f},{_final_target[1]:+.2f},{_final_target[2]:.2f}]")

                end_tag = ""
                if done:
                    quat = env._get_quat()
                    up_z = 1.0 - 2.0 * (float(quat[1])**2 + float(quat[2])**2)
                    if pos[2] < env.CRASH_Z:
                        end_tag = f"  {_R}CAKTI (zemin){_X}"
                    elif up_z < env.FLIP_UP_Z:
                        end_tag = f"  {_R}CAKTI (flip){_X}"
                    elif step >= max_steps:
                        end_tag = f"  {_G}HAYATTA{_X}"
                    else:
                        end_tag = f"  {_R}CAKTI (OOB){_X}"

                print(
                    f"  {step:5d}  "
                    f"{dc}{dist:7.3f}m{_X}  "
                    f"{pos[2]:6.3f}  "
                    f"{vel:6.3f}  "
                    f"{rew:7.2f}  "
                    f"{tgt_str}{end_tag}"
                )

            env.render()

            # Viewer hazir olduktan sonra kamerayi bir kere ayarla
            if not cam_set and env._viewer is not None:
                _setup_tracking_camera(env, distance=2.5)
                cam_set = True

            elapsed  = time.time() - t0
            expected = step * step_dt
            if expected > elapsed:
                time.sleep(expected - elapsed)

        print(f"  Toplam odul: {total_r:.1f}  |  Adim: {step}  |  Yeniden basliyor...")

    listener.stop()
    env.close()
    _dummy.close()
    print(f"\n{'='*58}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    parser.add_argument("--vec-norm",  default=DEFAULT_VEC_NORM, dest="vec_norm")
    parser.add_argument("--dr",        action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--fps",       type=int, default=50)
    args = parser.parse_args()

    run_manual(
        model_path    = args.model,
        vec_norm_path = args.vec_norm,
        use_dr        = args.dr,
        max_steps     = args.max_steps,
        seed          = args.seed,
        fps           = args.fps,
    )
