"""
PC-side Real-Time Drone Controller
====================================
Loads a trained PPO model and runs inference at 50 Hz.
Communicates with the ESP32 over local WiFi via UDP.

UDP Protocol
────────────
  ESP32 → PC  (28 bytes, every 5 ms)
    float32[7]: ax, ay, az (m/s²)  gx, gy, gz (rad/s)  altitude (m)

  PC → ESP32  (16 bytes)
    float32[4]: motor_FL, motor_FR, motor_BL, motor_BR  ∈ [0, 1]

Usage:
    python deploy/pc_controller.py models/trained/best/best_model.zip \\
        --norm   models/trained/vec_normalize.pkl \\
        --esp32  192.168.1.100 \\
        --target-z 1.0 \\
        --freq   50

Press  Ctrl+C  to land and stop.
"""

import argparse
import os
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.drone_env import WhoopDroneEnv

# ─────────────────────────────────────────────────────────────────────────────
# Safety constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_THROTTLE_SAFE = 0.80   # Never exceed 80 % in real flight
MIN_ALTITUDE_M    = 0.12   # Emergency stop if below 12 cm
MAX_TILT_DEG      = 50.0   # Emergency stop if tilted more than 50°
CMD_TIMEOUT_S     = 0.5    # Stop if no sensor packet for 0.5 s

# ─────────────────────────────────────────────────────────────────────────────

class ComplementaryFilter:
    """
    Minimal attitude estimator blending gyro integration with
    accelerometer tilt.  Good enough for slow indoor hover.
    For aggressive flight, replace with a full EKF or Madgwick filter.
    """

    def __init__(self, alpha: float = 0.98):
        self.alpha  = alpha
        self.roll   = 0.0
        self.pitch  = 0.0
        self.yaw    = 0.0    # gyro-integrated yaw (drifts, no magnetometer)
        self._last  = None

    def update(self, ax, ay, az, gx, gy, gz, dt: float):
        # Accelerometer angles (noise-free in static case)
        norm = np.sqrt(ax**2 + ay**2 + az**2)
        if norm < 1e-6:
            return self.roll, self.pitch
        ax_n, ay_n, az_n = ax / norm, ay / norm, az / norm

        roll_acc  = np.arctan2(ay_n, az_n)
        pitch_acc = np.arctan2(-ax_n, np.sqrt(ay_n**2 + az_n**2))

        # Complementary blend
        self.roll  = self.alpha * (self.roll  + gx * dt) + (1 - self.alpha) * roll_acc
        self.pitch = self.alpha * (self.pitch + gy * dt) + (1 - self.alpha) * pitch_acc
        # Integrate gyro-z for yaw (no magnetometer correction → slow drift)
        self.yaw  += gz * dt
        self.yaw   = float(np.arctan2(np.sin(self.yaw), np.cos(self.yaw)))  # wrap [-π,π]
        return self.roll, self.pitch

    def to_quat(self) -> np.ndarray:
        """Convert roll/pitch/yaw to quaternion (ZYX Euler convention)."""
        hr, hp, hy = self.roll / 2.0, self.pitch / 2.0, self.yaw / 2.0
        cr, sr = np.cos(hr), np.sin(hr)
        cp, sp = np.cos(hp), np.sin(hp)
        cy, sy = np.cos(hy), np.sin(hy)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        q  = np.array([qw, qx, qy, qz], dtype=np.float32)
        return q / (np.linalg.norm(q) + 1e-8)


class DroneController:
    def __init__(
        self,
        model_path:    str,
        vec_norm_path: str  = None,
        esp32_ip:      str  = "192.168.1.100",
        local_port:    int  = 8889,
        esp32_port:    int  = 8888,
    ):
        # ── RL model ─────────────────────────────────────────────────────────
        self.model = PPO.load(model_path, device="cpu")
        print(f"[ctrl] Model loaded: {model_path}")

        self.vec_norm = None
        if vec_norm_path and os.path.exists(vec_norm_path):
            dummy = DummyVecEnv([lambda: WhoopDroneEnv()])
            vn    = VecNormalize.load(vec_norm_path, dummy)
            vn.training    = False
            vn.norm_reward = False
            self.vec_norm  = vn
            print(f"[ctrl] VecNormalize loaded: {vec_norm_path}")

        # ── UDP socket ───────────────────────────────────────────────────────
        self.esp32_addr = (esp32_ip, esp32_port)
        self.sock       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", local_port))
        self.sock.settimeout(0.05)   # 50 ms read timeout
        print(f"[ctrl] UDP  listen :{local_port}  →  {esp32_ip}:{esp32_port}")

        # ── State ─────────────────────────────────────────────────────────────
        self.target_pos   = np.array([0.0, 0.0, 1.0])
        self.target_yaw   = 0.0
        self.position     = np.array([0.0, 0.0, 1.0])
        self.velocity     = np.zeros(3)
        self.ang_vel      = np.zeros(3)
        self.prev_action  = np.ones(4, dtype=np.float32) * WhoopDroneEnv.HOVER_THROTTLE
        self.cf           = ComplementaryFilter()
        self._last_sensor = 0.0

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.vec_norm is None:
            return obs
        return self.vec_norm.normalize_obs(obs.reshape(1, -1)).flatten()

    # ── UDP I/O ───────────────────────────────────────────────────────────────

    def _recv_sensor(self) -> bool:
        """Return True if a fresh sensor packet was received."""
        try:
            data, _ = self.sock.recvfrom(64)
        except socket.timeout:
            return False
        except OSError:
            return False

        if len(data) < 28:
            return False

        ax, ay, az, gx, gy, gz, altitude = struct.unpack_from("7f", data)

        now = time.monotonic()
        dt  = now - self._last_sensor if self._last_sensor > 0 else 0.005
        self._last_sensor = now

        # Attitude estimation
        roll, pitch = self.cf.update(ax, ay, az, gx, gy, gz, dt)
        self.ang_vel    = np.array([gx, gy, gz], dtype=np.float32)

        # Altitude from barometer (relative)
        self.position[2] = float(altitude)

        return True

    def _send_motors(self, cmds: np.ndarray):
        cmds = np.clip(cmds, 0.0, MAX_THROTTLE_SAFE).astype(np.float32)
        self.sock.sendto(struct.pack("4f", *cmds), self.esp32_addr)

    # ── Safety ────────────────────────────────────────────────────────────────

    def _is_safe(self) -> bool:
        if self.position[2] < MIN_ALTITUDE_M:
            print(f"[SAFETY] Low altitude: {self.position[2]:.2f} m")
            return False
        w         = self.cf.to_quat()[0]
        tilt_deg  = np.degrees(2.0 * np.arccos(np.clip(abs(w), 0.0, 1.0)))
        if tilt_deg > MAX_TILT_DEG:
            print(f"[SAFETY] Excessive tilt: {tilt_deg:.1f}°")
            return False
        return True

    def _emergency_stop(self):
        print("[SAFETY] EMERGENCY STOP – cutting motors")
        zeros = np.zeros(4, dtype=np.float32)
        for _ in range(20):
            self._send_motors(zeros)
            time.sleep(0.01)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        pos_err = self.position - self.target_pos
        quat    = self.cf.to_quat()
        yaw_err = np.float32(np.arctan2(
            np.sin(self.cf.yaw - self.target_yaw),
            np.cos(self.cf.yaw - self.target_yaw),
        ))
        obs     = np.concatenate([
            pos_err.astype(np.float32),
            self.velocity.astype(np.float32),
            quat,
            self.ang_vel.astype(np.float32),
            [yaw_err],
            self.prev_action,
        ])
        return obs

    def run(self, control_freq: float = 50.0):
        dt      = 1.0 / control_freq
        running = True

        print(f"\n[ctrl] Control loop starting at {control_freq:.0f} Hz")
        print(f"[ctrl] Target position: {self.target_pos}")
        print("[ctrl] Press Ctrl+C to land.\n")

        try:
            while running:
                t0 = time.monotonic()

                received = self._recv_sensor()

                if received:
                    # Check sensor timeout
                    if time.monotonic() - self._last_sensor > CMD_TIMEOUT_S:
                        print("[ctrl] Sensor timeout!")
                        self._emergency_stop()
                        break

                    if not self._is_safe():
                        self._emergency_stop()
                        break

                    obs      = self._build_obs()
                    norm_obs = self._normalize(obs)
                    action, _ = self.model.predict(norm_obs, deterministic=True)
                    action    = np.clip(action.astype(np.float32), 0.0, 1.0)

                    self._send_motors(action)
                    self.prev_action = action.copy()

                    # Diagnostic print at ~2 Hz
                    if int(t0 * 2) % 2 == 0:
                        print(
                            f"  z={self.position[2]:.2f}m  "
                        f"yaw={np.degrees(self.cf.yaw):+.0f}\u00b0  "
                            end="\r",
                        )

                # Maintain loop frequency
                elapsed   = time.monotonic() - t0
                remainder = dt - elapsed
                if remainder > 0:
                    time.sleep(remainder)

        except KeyboardInterrupt:
            print("\n[ctrl] KeyboardInterrupt – landing …")
        finally:
            self._emergency_stop()
            self.sock.close()
            if self.vec_norm is not None:
                self.vec_norm.close()
            print("[ctrl] Stopped.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PC-side RL drone controller")
    parser.add_argument("model_path",        help="Trained PPO model (.zip)")
    parser.add_argument("--norm",  default=None,          help="vec_normalize.pkl path")
    parser.add_argument("--esp32", default="192.168.1.100", help="ESP32 IP address")
    parser.add_argument("--freq",  type=float, default=50.0, help="Control frequency Hz")
    parser.add_argument("--target-z", type=float, default=1.0, help="Hover altitude (m)")
    args = parser.parse_args()

    ctrl = DroneController(
        model_path    = args.model_path,
        vec_norm_path = args.norm,
        esp32_ip      = args.esp32,
    )
    ctrl.target_pos[2] = args.target_z
    ctrl.run(control_freq=args.freq)


if __name__ == "__main__":
    main()
