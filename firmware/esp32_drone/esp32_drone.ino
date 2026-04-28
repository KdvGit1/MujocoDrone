/*
 * Whoop Drone – ESP32 Firmware
 * ============================================================
 * Hardware:
 *   ESP32 (WROOM-32 or S3)
 *   GY-91 module (MPU9250 + BMP280) via I2C
 *   4× Brushless ESCs (standard 50 Hz PWM, 1000–2000 µs)
 *
 * Pin assignments (change as needed):
 *   I2C  SDA → GPIO 21
 *   I2C  SCL → GPIO 22
 *   ESC FL   → GPIO 13
 *   ESC FR   → GPIO 12
 *   ESC BL   → GPIO 14
 *   ESC BR   → GPIO 27
 *
 * Motor order matches PC controller and MuJoCo model:
 *   ctrl[0] = FL  (front-left,  CCW)
 *   ctrl[1] = FR  (front-right, CW)
 *   ctrl[2] = BL  (back-left,   CW)
 *   ctrl[3] = BR  (back-right,  CCW)
 *
 * UDP Protocol:
 *   ESP32 → PC  every 5 ms  (200 Hz)
 *     28 bytes: float[7] = {ax, ay, az, gx, gy, gz, altitude}
 *     ax/ay/az in m/s²,  gx/gy/gz in rad/s,  altitude in m
 *
 *   PC → ESP32  (any time)
 *     16 bytes: float[4] = {m_FL, m_FR, m_BL, m_BR}  ∈ [0.0, 1.0]
 *
 * Libraries required (install via Arduino Library Manager):
 *   - "MPU9250" by hideakitai   (https://github.com/hideakitai/MPU9250)
 *   - "Adafruit BMP280 Library" by Adafruit
 *
 * IMPORTANT – Read before first flight:
 *   1. Flash and test with props REMOVED.
 *   2. Verify correct motor rotation direction; swap any two motor wires to reverse.
 *   3. Perform ESC calibration (max throttle on power-up, then min).
 *   4. Keep WIFI_SSID / WIFI_PASS in a separate header, NOT in version control.
 * ============================================================
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <MPU9250.h>
#include <Adafruit_BMP280.h>

// ─────────────────────────────────────────────────────────────
//  Network  (replace with your values)
// ─────────────────────────────────────────────────────────────
static const char* WIFI_SSID = "YOUR_WIFI_SSID";
static const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

// Port this ESP32 LISTENS on for motor commands
static const uint16_t LOCAL_CMD_PORT = 8888;

// PC address and port to SEND sensor data to
static const char*   PC_IP       = "192.168.1.101";   // ← your PC IP
static const uint16_t PC_DATA_PORT = 8889;

// ─────────────────────────────────────────────────────────────
//  Motor pin assignment
// ─────────────────────────────────────────────────────────────
static const int PIN_FL = 13;
static const int PIN_FR = 12;
static const int PIN_BL = 14;
static const int PIN_BR = 27;

// LEDC channels (0-15 available on ESP32)
static const int CH_FL = 0;
static const int CH_FR = 1;
static const int CH_BL = 2;
static const int CH_BR = 3;

// Standard ESC PWM: 50 Hz, 1000–2000 µs
// With 16-bit LEDC at 50 Hz: period = 20 000 µs = 65 535 ticks
//   1000 µs → 3 276 ticks
//   2000 µs → 6 553 ticks
static const int PWM_FREQ       = 50;
static const int PWM_RESOLUTION = 16;
static const int PWM_ARMED_MIN  = 3276;   // 1000 µs  (armed idle)
static const int PWM_MAX        = 6553;   // 2000 µs  (full throttle)
static const int PWM_DISARMED   = 3000;   // < 1000 µs – some ESCs want this to stay silent

// ─────────────────────────────────────────────────────────────
//  Safety
// ─────────────────────────────────────────────────────────────
static const unsigned long CMD_TIMEOUT_MS = 500;   // emergency stop after 500 ms silence

// ─────────────────────────────────────────────────────────────
//  Globals
// ─────────────────────────────────────────────────────────────
WiFiUDP       udp;
MPU9250       imu;
Adafruit_BMP280 bmp;

// Sensor readings
volatile float imu_ax = 0, imu_ay = 0, imu_az = 0;
volatile float imu_gx = 0, imu_gy = 0, imu_gz = 0;
volatile float baro_altitude = 0.0f;

float  sea_level_hPa  = 1013.25f;
float  base_altitude  = 0.0f;        // altitude at boot, used as zero

// Motor commands received from PC
float  motor_cmd[4]   = {0, 0, 0, 0};

bool   armed          = false;
unsigned long last_cmd_ms = 0;

// ─────────────────────────────────────────────────────────────
//  Forward declarations
// ─────────────────────────────────────────────────────────────
void setupWiFi();
void setupIMU();
void setupMotors();
void armESCs();
void stopMotors();
void setMotorThrottle(int ch, float t);
void applyMotorCommands();
void readSensors();
void sendSensorPacket();
void receiveMotorCommands();
void safetyCheck();

// ─────────────────────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println("\n=== Whoop Drone ESP32 Firmware ===");

    setupMotors();   // motors first – keep them silent
    setupWiFi();
    setupIMU();

    // Capture ground-level altitude reference
    delay(300);
    base_altitude = bmp.readAltitude(sea_level_hPa);
    Serial.print("Base altitude: ");
    Serial.print(base_altitude, 2);
    Serial.println(" m");

    armESCs();
    Serial.println("Ready. Waiting for PC commands …");
}

// ─────────────────────────────────────────────────────────────
//  Main loop  (~200 Hz)
// ─────────────────────────────────────────────────────────────
void loop() {
    static unsigned long lastSensorMs = 0;
    unsigned long now = millis();

    // Read & transmit sensors at 200 Hz
    if (now - lastSensorMs >= 5) {
        readSensors();
        sendSensorPacket();
        lastSensorMs = now;
    }

    // Check for incoming motor commands (non-blocking)
    receiveMotorCommands();

    // Safety: cut motors on PC communication loss
    safetyCheck();

    // Apply latest commands
    if (armed) {
        applyMotorCommands();
    }
}

// ─────────────────────────────────────────────────────────────
//  WiFi
// ─────────────────────────────────────────────────────────────
void setupWiFi() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("Connecting to WiFi");
    for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
        delay(500);
        Serial.print('.');
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\nWiFi FAILED – check credentials.");
    }
    udp.begin(LOCAL_CMD_PORT);
    Serial.printf("UDP listening on port %d\n", LOCAL_CMD_PORT);
}

// ─────────────────────────────────────────────────────────────
//  IMU  (GY-91: MPU9250 + BMP280 on I2C)
// ─────────────────────────────────────────────────────────────
void setupIMU() {
    Wire.begin(21, 22);          // SDA=21, SCL=22
    Wire.setClock(400000);       // 400 kHz fast mode

    // ── MPU9250 ────────────────────────────────────────────────
    MPU9250Setting s;
    s.accel_fs_sel    = ACCEL_FS_SEL::A4G;        // ±4 g
    s.gyro_fs_sel     = GYRO_FS_SEL::G500DPS;     // ±500 °/s
    s.fifo_sample_rate= FIFO_SAMPLE_RATE::SMPL_200HZ;
    s.gyro_dlpf_cfg   = GYRO_DLPF_CFG::DLPF_41HZ;
    s.accel_dlpf_cfg  = ACCEL_DLPF_CFG::DLPF_45HZ;

    if (!imu.setup(0x68, s, Wire)) {
        Serial.println("MPU9250 not found on 0x68! Check wiring.");
        while (true) delay(1000);
    }
    Serial.println("MPU9250 OK");

    Serial.println("Gyro calibration – keep drone still for 5 s …");
    imu.calibrateAccelGyro();
    Serial.println("Calibration done.");

    // ── BMP280 ─────────────────────────────────────────────────
    // GY-91 typically has BMP280 at 0x76; try 0x77 as fallback
    bool bmpOk = bmp.begin(0x76);
    if (!bmpOk) bmpOk = bmp.begin(0x77);
    if (!bmpOk) {
        Serial.println("BMP280 not found! Altitude will read 0.");
    } else {
        bmp.setSampling(
            Adafruit_BMP280::MODE_NORMAL,
            Adafruit_BMP280::SAMPLING_X2,    // temperature
            Adafruit_BMP280::SAMPLING_X16,   // pressure
            Adafruit_BMP280::FILTER_X16,
            Adafruit_BMP280::STANDBY_MS_1
        );
        Serial.println("BMP280 OK");
    }
}

// ─────────────────────────────────────────────────────────────
//  Motors / ESCs
// ─────────────────────────────────────────────────────────────
void setupMotors() {
    // Configure LEDC PWM for each motor channel
    ledcSetup(CH_FL, PWM_FREQ, PWM_RESOLUTION);
    ledcSetup(CH_FR, PWM_FREQ, PWM_RESOLUTION);
    ledcSetup(CH_BL, PWM_FREQ, PWM_RESOLUTION);
    ledcSetup(CH_BR, PWM_FREQ, PWM_RESOLUTION);

    ledcAttachPin(PIN_FL, CH_FL);
    ledcAttachPin(PIN_FR, CH_FR);
    ledcAttachPin(PIN_BL, CH_BL);
    ledcAttachPin(PIN_BR, CH_BR);

    stopMotors();
    Serial.println("Motors initialised (disarmed).");
}

/*
 * Standard ESC arming sequence:
 *   1. Power on ESC with throttle at minimum → ESC plays arming tones.
 *   For most BLHeli/SimonK ESCs this happens automatically when min signal
 *   is detected.  If your ESCs need explicit calibration (max first),
 *   uncomment the calibration block below.
 */
void armESCs() {
    Serial.println("Arming ESCs (min throttle for 3 s) …");

    // Optional calibration: uncomment if ESCs need max-then-min sequence
    // for (int ch = 0; ch < 4; ch++) ledcWrite(ch, PWM_MAX);
    // delay(3000);

    for (int ch = 0; ch < 4; ch++) ledcWrite(ch, PWM_ARMED_MIN);
    delay(3000);

    armed = true;
    Serial.println("ESCs armed.");
}

void stopMotors() {
    for (int ch = 0; ch < 4; ch++) ledcWrite(ch, PWM_DISARMED);
    motor_cmd[0] = motor_cmd[1] = motor_cmd[2] = motor_cmd[3] = 0.0f;
}

/*
 * Map throttle [0.0, 1.0] → PWM tick in [PWM_ARMED_MIN, PWM_MAX].
 */
void setMotorThrottle(int ch, float throttle) {
    throttle      = constrain(throttle, 0.0f, 1.0f);
    int pwm_ticks = (int)(PWM_ARMED_MIN + throttle * (float)(PWM_MAX - PWM_ARMED_MIN));
    ledcWrite(ch, pwm_ticks);
}

void applyMotorCommands() {
    setMotorThrottle(CH_FL, motor_cmd[0]);
    setMotorThrottle(CH_FR, motor_cmd[1]);
    setMotorThrottle(CH_BL, motor_cmd[2]);
    setMotorThrottle(CH_BR, motor_cmd[3]);
}

// ─────────────────────────────────────────────────────────────
//  Sensor reading
// ─────────────────────────────────────────────────────────────
void readSensors() {
    if (imu.update()) {
        // getLinearAcc* returns acceleration with gravity removed (m/s²)
        imu_ax = imu.getLinearAccX();
        imu_ay = imu.getLinearAccY();
        imu_az = imu.getLinearAccZ();
        // getGyro* returns °/s – convert to rad/s
        imu_gx = imu.getGyroX() * DEG_TO_RAD;
        imu_gy = imu.getGyroY() * DEG_TO_RAD;
        imu_gz = imu.getGyroZ() * DEG_TO_RAD;
    }
    baro_altitude = bmp.readAltitude(sea_level_hPa) - base_altitude;
}

// ─────────────────────────────────────────────────────────────
//  UDP send: sensor packet to PC
// ─────────────────────────────────────────────────────────────
void sendSensorPacket() {
    float pkt[7] = {
        imu_ax, imu_ay, imu_az,
        imu_gx, imu_gy, imu_gz,
        baro_altitude
    };
    udp.beginPacket(PC_IP, PC_DATA_PORT);
    udp.write((const uint8_t*)pkt, sizeof(pkt));   // 28 bytes
    udp.endPacket();
}

// ─────────────────────────────────────────────────────────────
//  UDP receive: motor commands from PC
// ─────────────────────────────────────────────────────────────
void receiveMotorCommands() {
    int sz = udp.parsePacket();
    if (sz < 16) return;

    uint8_t buf[16];
    udp.read(buf, 16);

    float cmds[4];
    memcpy(cmds, buf, 16);

    // Validate and clamp
    for (int i = 0; i < 4; i++) {
        if (isnan(cmds[i]) || isinf(cmds[i])) {
            return;   // discard malformed packet
        }
        motor_cmd[i] = constrain(cmds[i], 0.0f, 1.0f);
    }
    last_cmd_ms = millis();
}

// ─────────────────────────────────────────────────────────────
//  Safety: cut motors on communication loss
// ─────────────────────────────────────────────────────────────
void safetyCheck() {
    if (!armed || last_cmd_ms == 0) return;

    if ((millis() - last_cmd_ms) > CMD_TIMEOUT_MS) {
        Serial.println("[SAFETY] Command timeout – stopping motors!");
        stopMotors();
        last_cmd_ms = 0;   // prevent repeated prints
    }
}
