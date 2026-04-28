"""
RC Waypoint Controller
=======================
Klavye ile target_pos'u değiştirerek drone'u yönlendirir.
İki mod desteklenir:

  --mode esp32   Standalone ESP32 modu:  PC sadece waypoint gönderir (12 byte UDP)
                 Model ESP32 üzerinde çalışır (model_weights.h)

  --mode pc      PC model modu: PC PPO inference yapar, motor komutu gönderir.
                 Orijinal pc_controller.py mantığı, dinamik target ile.

Kurulum:
    pip install pynput

Kullanım:
    # Standalone ESP32 modu (önerilen)
    python deploy/rc_control.py --mode esp32 --esp32 192.168.1.100

    # PC model modu
    python deploy/rc_control.py --mode pc \\
        --model models/trained/stage2_dr_final.zip \\
        --norm  models/trained/vec_normalize_stage2_dr.pkl \\
        --esp32 192.168.1.100

Klavye:
    Z / C    → Yaw sol / Yaw sağ  (target_yaw)
    W / S    → İleri / Geri    (target_x)
    A / D    → Sol  / Sağ     (target_y)
    Q / E    → Yukarı / Aşağı  (target_z)
    SPACE    → target_pos dondur (hover in place)
    R        → Home – target'ı [0,0,target_z]'ye sıfırla
    X / ESC  → Acil durdur (motorları kes)

Her tuş basışında target 0.1 m kayar.
Tuş basılı tutulursa 50 Hz'de tekrar tetiklenir (~5 m/s max).
"""

import argparse
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pynput import keyboard as kb
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    print("WARNING: pynput not installed. Run:  pip install pynput")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
STEP     = 0.10    # m per control loop tick (at 50 Hz, held key → 5 m/s max)
YAW_STEP = 0.052   # rad per tick  ≈ 3°/tick  (held key → ~150°/s at 50 Hz)
FREQ     = 50.0    # Hz


# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────
target     = np.array([0.0, 0.0, 1.0], dtype=np.float32)
target_yaw = 0.0   # radians, wrapped to [-π, π]
pressed    = set()
running    = True


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard listener callbacks
# ─────────────────────────────────────────────────────────────────────────────

def on_press(key):
    try:
        pressed.add(key.char.lower())
    except AttributeError:
        pressed.add(key)


def on_release(key):
    try:
        pressed.discard(key.char.lower())
    except AttributeError:
        pressed.discard(key)


def process_keys() -> bool:
    """
    Update target based on currently pressed keys.
    Returns True if a kill command was issued.
    """
    global running, target_yaw

    # Kill
    kill_keys = {'x'}
    if HAS_PYNPUT:
        kill_keys.add(kb.Key.esc)
    if kill_keys & pressed:
        running = False
        return True

    # Freeze (space bar)
    if ' ' in pressed:
        return False

    # Home
    if 'r' in pressed:
        target[0]  = 0.0
        target[1]  = 0.0
        target_yaw = 0.0
        return False

    # Movement
    if 'w' in pressed:  target[0] += STEP
    if 's' in pressed:  target[0] -= STEP
    if 'a' in pressed:  target[1] += STEP
    if 'd' in pressed:  target[1] -= STEP
    if 'q' in pressed:  target[2] += STEP
    if 'e' in pressed:  target[2] -= STEP

    # Yaw  (Z = CCW / left,  C = CW / right)
    if 'z' in pressed:  target_yaw -= YAW_STEP
    if 'c' in pressed:  target_yaw += YAW_STEP
    target_yaw = float(np.arctan2(np.sin(target_yaw), np.cos(target_yaw)))

    # Altitude floor
    target[2] = max(0.15, target[2])
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: ESP32 standalone – PC sends 12-byte waypoints
# ─────────────────────────────────────────────────────────────────────────────

def run_esp32_mode(esp32_ip: str, esp32_port: int, local_port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", local_port))
    sock.settimeout(0.02)

    print(f"\n[RC] ESP32-standalone mode  →  {esp32_ip}:{esp32_port}")
    print("[RC] W/S=fwd/bck  A/D=left/right  Q/E=up/dn  Z/C=yaw  SPC=freeze  R=home  X/ESC=kill\n")

    dt = 1.0 / FREQ

    while running:
        t0   = time.monotonic()
        kill = process_keys()

        if kill:
            # Send kill signal: z = -1.0 is the kill sentinel
            pkt = struct.pack("4f", 0.0, 0.0, -1.0, 0.0)
            for _ in range(20):
                sock.sendto(pkt, (esp32_ip, esp32_port))
                time.sleep(0.01)
            break

        pkt = struct.pack("4f", float(target[0]), float(target[1]), float(target[2]), float(target_yaw))
        sock.sendto(pkt, (esp32_ip, esp32_port))

        print(
            f"  target=[{target[0]:+.2f}, {target[1]:+.2f}, z={target[2]:.2f}]  yaw={np.degrees(target_yaw):+.0f}\u00b0  ",
            end="\r",
        )

        elapsed = time.monotonic() - t0
        rem = dt - elapsed
        if rem > 0:
            time.sleep(rem)

    sock.close()
    print("\n[RC] ESP32 mode stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: PC model – PC runs PPO inference, sends motor commands
# ─────────────────────────────────────────────────────────────────────────────

def run_pc_mode(
    model_path:    str,
    vec_norm_path: str,
    esp32_ip:      str,
    esp32_port:    int,
    local_port:    int,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs.drone_env import WhoopDroneEnv
    from deploy.pc_controller import ComplementaryFilter

    # ── Load model ─────────────────────────────────────────────────────────────
    model = PPO.load(model_path, device="cpu")
    vec_norm = None
    if vec_norm_path and os.path.exists(vec_norm_path):
        dummy = DummyVecEnv([lambda: WhoopDroneEnv()])
        vn    = VecNormalize.load(vec_norm_path, dummy)
        vn.training    = False
        vn.norm_reward = False
        vec_norm = vn

    def normalize(obs):
        if vec_norm is None:
            return obs
        return vec_norm.normalize_obs(obs.reshape(1, -1)).flatten()

    # ── Socket ─────────────────────────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", local_port))
    sock.settimeout(0.05)

    cf           = ComplementaryFilter()
    position     = np.array([0.0, 0.0, 1.0])
    ang_vel      = np.zeros(3)
    velocity     = np.zeros(3)
    prev_action  = np.ones(4, dtype=np.float32) * WhoopDroneEnv.HOVER_THROTTLE
    last_sensor  = 0.0

    def recv_sensor():
        nonlocal last_sensor, position, ang_vel
        try:
            data, _ = sock.recvfrom(64)
        except (socket.timeout, OSError):
            return False
        if len(data) < 28:
            return False
        ax, ay, az, gx, gy, gz, altitude = struct.unpack_from("7f", data)
        now = time.monotonic()
        dt  = now - last_sensor if last_sensor > 0 else 0.005
        last_sensor = now
        cf.update(ax, ay, az, gx, gy, gz, dt)
        ang_vel[:]  = [gx, gy, gz]
        position[2] = float(altitude)
        return True

    def send_motors(cmds):
        cmds = np.clip(cmds, 0.0, 0.80).astype(np.float32)
        sock.sendto(struct.pack("4f", *cmds), (esp32_ip, esp32_port))

    def emergency_stop():
        zeros = np.zeros(4, dtype=np.float32)
        for _ in range(20):
            send_motors(zeros)
            time.sleep(0.01)

    def is_safe():
        if position[2] < 0.12:
            return False
        w = cf.to_quat()[0]
        return np.degrees(2.0 * np.arccos(np.clip(abs(w), 0.0, 1.0))) < 50.0

    print(f"\n[RC] PC-model mode  →  {esp32_ip}:{esp32_port}")
    print("[RC] W/S=fwd/bck  A/D=left/right  Q/E=up/dn  Z/C=yaw  SPC=freeze  R=home  X/ESC=kill\n")

    dt = 1.0 / FREQ

    try:
        while running:
            t0   = time.monotonic()
            kill = process_keys()
            if kill:
                break

            if recv_sensor():
                if not is_safe():
                    print("\n[SAFETY] Unsafe – cutting motors.")
                    break

                pos_err = position - target
                quat    = cf.to_quat()
                yaw_err = np.float32(np.arctan2(
                    np.sin(cf.yaw - target_yaw),
                    np.cos(cf.yaw - target_yaw),
                ))
                obs     = np.concatenate([
                    pos_err.astype(np.float32),
                    velocity.astype(np.float32),
                    quat,
                    ang_vel.astype(np.float32),
                    [yaw_err],
                    prev_action,
                ])
                norm_obs = normalize(obs)
                action, _ = model.predict(norm_obs, deterministic=True)
                action    = np.clip(action.astype(np.float32), 0.0, 1.0)
                send_motors(action)
                prev_action[:] = action

                print(
                    f"  z={position[2]:.2f}m  tgt={target[2]:.2f}m  "
                    f"yaw={np.degrees(cf.yaw):+.0f}\u00b0  "
                    f"M=[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.2f}]  ",
                    end="\r",
                )

            elapsed = time.monotonic() - t0
            rem = dt - elapsed
            if rem > 0:
                time.sleep(rem)

    except KeyboardInterrupt:
        pass
    finally:
        emergency_stop()
        sock.close()
        if vec_norm is not None:
            vec_norm.close()
    print("\n[RC] PC-model mode stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RC Waypoint Controller for Whoop Drone")
    parser.add_argument("--mode",       choices=["esp32", "pc"], default="esp32",
                        help="esp32=on-device inference  pc=PC inference (default: esp32)")
    parser.add_argument("--esp32",      default="192.168.1.100",  help="ESP32 IP address")
    parser.add_argument("--esp32-port", type=int, default=8888,   help="ESP32 UDP port")
    parser.add_argument("--local-port", type=int, default=8889,   help="PC UDP listen port")
    parser.add_argument("--model",      default=None,             help="[pc mode] .zip model path")
    parser.add_argument("--norm",       default=None,             help="[pc mode] vec_normalize.pkl")
    parser.add_argument("--target-z",   type=float, default=1.0,  help="Initial hover altitude (m)")
    args = parser.parse_args()

    target[2] = max(0.15, args.target_z)

    if not HAS_PYNPUT:
        print("ERROR: pynput is required.  pip install pynput")
        sys.exit(1)

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        if args.mode == "esp32":
            run_esp32_mode(args.esp32, args.esp32_port, args.local_port)
        else:
            if not args.model:
                print("ERROR: --model is required in pc mode.")
                sys.exit(1)
            run_pc_mode(
                model_path    = args.model,
                vec_norm_path = args.norm,
                esp32_ip      = args.esp32,
                esp32_port    = args.esp32_port,
                local_port    = args.local_port,
            )
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
