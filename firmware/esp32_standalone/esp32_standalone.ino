/*
 * Whoop Drone – ESP32 Standalone Firmware (On-Device RL Inference)
 * ================================================================
 * Model inference runs LOCALLY on ESP32. PC is optional (waypoints only).
 *
 * Hardware:
 *   ESP32 WROOM-32 / S3
 *   GY-91  (MPU9250 + BMP280) via I2C  SDA=21, SCL=22
 *   4x ESC  FL=13  FR=12  BL=14  BR=27
 *
 * BEFORE FLASHING:
 *   1. Run:  python deploy/export_to_esp32.py <model.zip> --norm <vn.pkl> --out firmware/esp32_standalone/model_weights.h
 *   2. Set WIFI_SSID / WIFI_PASS / PC_IP below.
 *   3. Flash with:  Board=ESP32 Dev Module, Flash size=4MB, Partition=No OTA (2MB APP)
 *   4. First test with PROPS REMOVED.
 *
 * UDP Protocol:
 *   PC -> ESP32  (16 bytes)  float[4] = {target_x, target_y, target_z, target_yaw}
 *                             Special: z=-1.0 = kill,  z=-2.0 = IMU recalibrate
 *   ESP32 -> PC  (28 bytes)  float[7] = {ax, ay, az, gx, gy, gz, altitude}
 *
 * Observation layout (must match envs/drone_env.py):
 *   [0:3]   pos_err  = position - target  (m)
 *   [3:6]   velocity  (m/s)
 *   [6:10]  quaternion [w, x, y, z]
 *   [10:13] angular velocity (rad/s)
 *   [13]    yaw_err  (rad)  ← DO NOT SKIP
 *   [14:18] previous motor actions
 *   Total: 18 dimensions
 *
 * Motor order: [FL(CCW), FR(CW), BL(CW), BR(CCW)]
 *
 * Libraries (Arduino Library Manager):
 *   MPU9250  by hideakitai
 *   Adafruit BMP280 Library
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <MPU9250.h>
#include <Adafruit_BMP280.h>

// ── GENERATED MODEL WEIGHTS – run export_to_esp32.py first ──────────────────
#include "model_weights.h"

// ── Network configuration ────────────────────────────────────────────────────
static const char*    WIFI_SSID     = "YOUR_WIFI_SSID";
static const char*    WIFI_PASS     = "YOUR_WIFI_PASSWORD";
static const char*    PC_IP         = "192.168.1.101";   // PC address
static const uint16_t LOCAL_PORT    = 8888;              // receive waypoints
static const uint16_t PC_DATA_PORT  = 8889;              // send telemetry

// ── Motor pins ───────────────────────────────────────────────────────────────
static const int PIN_FL = 13, PIN_FR = 12, PIN_BL = 14, PIN_BR = 27;
static const int CH_FL  = 0,  CH_FR  = 1,  CH_BL  = 2,  CH_BR  = 3;

// Standard ESC 50 Hz PWM (16-bit LEDC)
//   1000 µs (armed idle) = 3276 ticks
//   2000 µs (full)       = 6553 ticks
static const int PWM_ARMED_MIN = 3276;
static const int PWM_MAX       = 6553;
static const int PWM_DISARMED  = 3000;

// ── Safety thresholds ────────────────────────────────────────────────────────
static const float        MIN_ALT_M      = 0.12f;   // emergency stop below 12 cm
static const float        MAX_TILT_W     = 0.40f;   // |quat.w| < 0.40 = flip (~66°)
static const unsigned long WP_TIMEOUT_MS = 2000;    // ms without waypoint → keep last

// ── Globals ──────────────────────────────────────────────────────────────────
WiFiUDP         udp;
MPU9250         imu;
Adafruit_BMP280 bmp;

// Sensor state
float imu_ax = 0, imu_ay = 0, imu_az = 0;
float imu_gx = 0, imu_gy = 0, imu_gz = 0;
float baro_alt = 0.0f, base_alt = 0.0f;

// Complementary filter attitude
float cf_roll = 0.0f, cf_pitch = 0.0f, cf_yaw = 0.0f;
static const float CF_ALPHA = 0.98f;

// RL state
float target_pos[3]  = {0.0f, 0.0f, 1.0f};   // default: hover 1 m
float target_yaw     = 0.0f;                   // default: face forward
float position[3]    = {0.0f, 0.0f, 1.0f};
float velocity[3]    = {0.0f, 0.0f, 0.0f};
float prev_action[4] = {0.50f, 0.50f, 0.50f, 0.50f};

unsigned long last_wp_ms = 0;
bool armed = false;

// ─────────────────────────────────────────────────────────────────────────────
// Complementary filter
// ─────────────────────────────────────────────────────────────────────────────
void cf_update(float ax, float ay, float az, float gx, float gy, float gz, float dt) {
    float norm = sqrtf(ax*ax + ay*ay + az*az);
    if (norm < 1e-6f) return;
    float axn = ax/norm, ayn = ay/norm, azn = az/norm;
    float roll_acc  = atan2f(ayn, azn);
    float pitch_acc = atan2f(-axn, sqrtf(ayn*ayn + azn*azn));
    cf_roll  = CF_ALPHA * (cf_roll  + gx * dt) + (1.0f - CF_ALPHA) * roll_acc;
    cf_pitch = CF_ALPHA * (cf_pitch + gy * dt) + (1.0f - CF_ALPHA) * pitch_acc;
    cf_yaw  += gz * dt;
    // Wrap to [-π, π]
    while (cf_yaw >  M_PI) cf_yaw -= 2.0f * M_PI;
    while (cf_yaw < -M_PI) cf_yaw += 2.0f * M_PI;
}

void cf_to_quat(float* qw, float* qx, float* qy, float* qz) {
    float hr = cf_roll  / 2.0f;
    float hp = cf_pitch / 2.0f;
    float hy = cf_yaw   / 2.0f;
    float cr = cosf(hr), sr = sinf(hr);
    float cp = cosf(hp), sp = sinf(hp);
    float cy = cosf(hy), sy = sinf(hy);
    *qw = cr*cp*cy + sr*sp*sy;
    *qx = sr*cp*cy - cr*sp*sy;
    *qy = cr*sp*cy + sr*cp*sy;
    *qz = cr*cp*sy - sr*sp*cy;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build observation vector (MODEL_OBS_DIM = 18)
// ─────────────────────────────────────────────────────────────────────────────
void build_obs(float* obs) {
    float qw, qx, qy, qz;
    cf_to_quat(&qw, &qx, &qy, &qz);

    obs[0]  = position[0] - target_pos[0];   // pos error x
    obs[1]  = position[1] - target_pos[1];   // pos error y
    obs[2]  = position[2] - target_pos[2];   // pos error z
    obs[3]  = velocity[0];
    obs[4]  = velocity[1];
    obs[5]  = velocity[2];
    obs[6]  = qw;
    obs[7]  = qx;
    obs[8]  = qy;
    obs[9]  = qz;
    obs[10] = imu_gx;
    obs[11] = imu_gy;
    obs[12] = imu_gz;
    // yaw_err: arctan2(sin(yaw - target_yaw), cos(yaw - target_yaw))
    float yaw_diff = cf_yaw - target_yaw;
    obs[13] = atan2f(sinf(yaw_diff), cosf(yaw_diff));
    obs[14] = prev_action[0];
    obs[15] = prev_action[1];
    obs[16] = prev_action[2];
    obs[17] = prev_action[3];
}

// ─────────────────────────────────────────────────────────────────────────────
// Motor helpers
// ─────────────────────────────────────────────────────────────────────────────
void motor_setup() {
    ledcSetup(CH_FL, 50, 16); ledcSetup(CH_FR, 50, 16);
    ledcSetup(CH_BL, 50, 16); ledcSetup(CH_BR, 50, 16);
    ledcAttachPin(PIN_FL, CH_FL); ledcAttachPin(PIN_FR, CH_FR);
    ledcAttachPin(PIN_BL, CH_BL); ledcAttachPin(PIN_BR, CH_BR);
    ledcWrite(CH_FL, PWM_DISARMED); ledcWrite(CH_FR, PWM_DISARMED);
    ledcWrite(CH_BL, PWM_DISARMED); ledcWrite(CH_BR, PWM_DISARMED);
}

void set_motor(int ch, float t) {
    // Hard cap at 80% for safety
    if (t < 0.0f) t = 0.0f;
    if (t > 0.80f) t = 0.80f;
    ledcWrite(ch, (int)(PWM_ARMED_MIN + t * (float)(PWM_MAX - PWM_ARMED_MIN)));
}

void apply_motors(float* act) {
    set_motor(CH_FL, act[0]);
    set_motor(CH_FR, act[1]);
    set_motor(CH_BL, act[2]);
    set_motor(CH_BR, act[3]);
}

void stop_motors() {
    ledcWrite(CH_FL, PWM_DISARMED); ledcWrite(CH_FR, PWM_DISARMED);
    ledcWrite(CH_BL, PWM_DISARMED); ledcWrite(CH_BR, PWM_DISARMED);
    for (int i = 0; i < 4; i++) prev_action[i] = 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Safety check
// ─────────────────────────────────────────────────────────────────────────────
bool is_safe() {
    if (position[2] < MIN_ALT_M) {
        Serial.printf("[SAFETY] Low alt: %.2f m\n", position[2]);
        return false;
    }
    float qw, qx, qy, qz;
    cf_to_quat(&qw, &qx, &qy, &qz);
    if (fabsf(qw) < MAX_TILT_W) {
        Serial.printf("[SAFETY] Flip detected qw=%.2f\n", qw);
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println("\n=== Whoop Drone Standalone ===");

    // Motors first – keep silent
    motor_setup();

    // WiFi
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("WiFi connecting");
    for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
        delay(500); Serial.print('.');
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\nWiFi FAILED.");
    }
    udp.begin(LOCAL_PORT);
    Serial.printf("UDP listening on port %d\n", LOCAL_PORT);

    // IMU
    Wire.begin(21, 22);
    Wire.setClock(400000);

    MPU9250Setting s;
    s.accel_fs_sel      = ACCEL_FS_SEL::A4G;
    s.gyro_fs_sel       = GYRO_FS_SEL::G500DPS;
    s.fifo_sample_rate  = FIFO_SAMPLE_RATE::SMPL_200HZ;
    s.gyro_dlpf_cfg     = GYRO_DLPF_CFG::DLPF_41HZ;
    s.accel_dlpf_cfg    = ACCEL_DLPF_CFG::DLPF_45HZ;
    if (!imu.setup(0x68, s, Wire)) {
        Serial.println("MPU9250 not found on 0x68 – check wiring!");
        while (true) delay(1000);
    }
    Serial.println("Gyro calibration (keep still 5 s)...");
    imu.calibrateAccelGyro();
    Serial.println("IMU OK");

    bool bmp_ok = bmp.begin(0x76);
    if (!bmp_ok) bmp_ok = bmp.begin(0x77);
    if (!bmp_ok) {
        Serial.println("BMP280 not found!");
    } else {
        bmp.setSampling(
            Adafruit_BMP280::MODE_NORMAL,
            Adafruit_BMP280::SAMPLING_X2,
            Adafruit_BMP280::SAMPLING_X16,
            Adafruit_BMP280::FILTER_X16,
            Adafruit_BMP280::STANDBY_MS_1
        );
        delay(300);
        base_alt = bmp.readAltitude(1013.25f);
        Serial.printf("BMP280 OK – base alt: %.2f m\n", base_alt);
    }

    // Arm ESCs (min throttle for 3 s)
    Serial.println("Arming ESCs (3 s) – keep props clear...");
    ledcWrite(CH_FL, PWM_ARMED_MIN); ledcWrite(CH_FR, PWM_ARMED_MIN);
    ledcWrite(CH_BL, PWM_ARMED_MIN); ledcWrite(CH_BR, PWM_ARMED_MIN);
    delay(3000);
    armed = true;
    last_wp_ms = millis();
    Serial.println("Armed. Ready.\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Main loop
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
    static unsigned long last_sens_ms = 0;
    static unsigned long last_ctrl_ms = 0;
    static float         last_alt     = 0.0f;
    static unsigned long last_alt_t   = 0;
    static int           dbg_ctr      = 0;

    unsigned long now = millis();

    // ── 200 Hz: read sensors ──────────────────────────────────────────────────
    if (now - last_sens_ms >= 5) {
        float dt = (now - last_sens_ms) * 1e-3f;
        last_sens_ms = now;

        if (imu.update()) {
            imu_ax = imu.getLinearAccX();
            imu_ay = imu.getLinearAccY();
            imu_az = imu.getLinearAccZ();
            imu_gx = imu.getGyroX() * DEG_TO_RAD;
            imu_gy = imu.getGyroY() * DEG_TO_RAD;
            imu_gz = imu.getGyroZ() * DEG_TO_RAD;
            cf_update(imu_ax, imu_ay, imu_az, imu_gx, imu_gy, imu_gz, dt);
        }

        float new_alt = bmp.readAltitude(1013.25f) - base_alt;
        position[2]   = new_alt;

        // Estimate vertical velocity from baro
        float alt_dt = (now - last_alt_t) * 1e-3f;
        if (alt_dt > 0.02f) {
            velocity[2]  = (new_alt - last_alt) / alt_dt;
            last_alt     = new_alt;
            last_alt_t   = now;
        }

        // Send telemetry to PC
        float pkt[7] = {imu_ax, imu_ay, imu_az, imu_gx, imu_gy, imu_gz, position[2]};
        udp.beginPacket(PC_IP, PC_DATA_PORT);
        udp.write((const uint8_t*)pkt, sizeof(pkt));
        udp.endPacket();
    }

    // ── Receive waypoints from PC ──────────────────────────────────────────────
    int sz = udp.parsePacket();
    if (sz >= 16) {
        uint8_t buf[16];
        udp.read(buf, 16);
        float wp[4];
        memcpy(wp, buf, 16);

        bool valid = !isnan(wp[0]) && !isnan(wp[1]) && !isnan(wp[2]) && !isnan(wp[3]);

        if (valid && wp[2] < -1.5f) {
            // z = -2.0: IMU recalibration request
            Serial.println("[UDP] IMU recalibration requested – keep drone still!");
            stop_motors();
            armed = false;
            imu.calibrateAccelGyro();
            Serial.println("[CAL] Done. Power-cycle or reflash to re-arm.");
            return;
        }

        if (valid && wp[2] < 0.0f) {
            // z = -1.0: kill signal
            Serial.println("[UDP] Kill signal received.");
            stop_motors();
            armed = false;
            return;
        }

        if (valid) {
            target_pos[0] = wp[0];
            target_pos[1] = wp[1];
            target_pos[2] = wp[2] < MIN_ALT_M ? MIN_ALT_M : wp[2];
            // Wrap received yaw to [-π, π]
            float ty = wp[3];
            while (ty >  M_PI) ty -= 2.0f * M_PI;
            while (ty < -M_PI) ty += 2.0f * M_PI;
            target_yaw = ty;
            last_wp_ms = now;
        }
    }

    // Waypoint timeout warning (non-fatal – hold last known target)
    if (armed && last_wp_ms > 0 && (now - last_wp_ms) > WP_TIMEOUT_MS) {
        static unsigned long last_warn_ms = 0;
        if (now - last_warn_ms > 1000) {  // warn once per second
            last_warn_ms = now;
            Serial.printf("[WARN] No waypoint for %lu ms – holding last target z=%.2f\n",
                          now - last_wp_ms, target_pos[2]);
        }
    }

    // ── 50 Hz: RL inference ───────────────────────────────────────────────────
    if (now - last_ctrl_ms >= 20) {
        last_ctrl_ms = now;

        if (!armed) return;

        if (!is_safe()) {
            stop_motors();
            armed = false;
            Serial.println("[SAFETY] Motors cut.");
            return;
        }

        float obs[MODEL_OBS_DIM];
        float act[MODEL_ACT_DIM];

        build_obs(obs);
        model_infer(obs, act);
        apply_motors(act);
        memcpy(prev_action, act, sizeof(float) * MODEL_ACT_DIM);

        // Serial debug ~2 Hz
        if (++dbg_ctr >= 25) {
            dbg_ctr = 0;
            Serial.printf(
                "z=%.2f tgt=%.2f  M=[%.2f %.2f %.2f %.2f]\n",
                position[2], target_pos[2],
                act[0], act[1], act[2], act[3]
            );
        }
    }
}
