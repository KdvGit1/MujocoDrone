# Whoop Drone – Uçuş Kılavuzu

---

## İÇİNDEKİLER

1. [Sistem Mimarisi](#1-sistem-mimarisi)
2. [Donanım Bağlantıları](#2-donanım-bağlantıları)
3. [Yazılım Kurulumu (PC)](#3-yazılım-kurulumu-pc)
4. [Model → ESP32 Aktarımı (esp32 modu)](#4-model--esp32-aktarımı)
5. [ESP32 Firmware Yapılandırması ve Flash](#5-esp32-firmware-yapılandırması-ve-flash)
6. [Uçuş Modları](#6-uçuş-modları)
   - 6a. ESP32 Standalone Modu (önerilen)
   - 6b. PC Model Modu
7. [Klavye Kontrol Referansı](#7-klavye-kontrol-referansı)
8. [IMU Kalibrasyon Prosedürü](#8-imu-kalibrasyon-prosedürü)
9. [Güvenlik ve Acil Durdurma](#9-güvenlik-ve-acil-durdurma)
10. [Simülasyonda Test (test_manual.py)](#10-simülasyonda-test)
11. [Sorun Giderme](#11-sorun-giderme)

---

## 1. Sistem Mimarisi

```
┌──────────────────────────────────────────────┐
│  PC  (Windows/Linux)                         │
│  python deploy/rc_control.py                 │
│  • Klavye → carrot hedef hesapla             │
│  • UDP → ESP32'ye target_x/y/z/yaw gönder   │
└───────────────────┬──────────────────────────┘
                    │ WiFi UDP  (4×float32, 16 byte)
                    ▼
┌──────────────────────────────────────────────┐
│  ESP32 (Standalone modu)                     │
│  firmware/esp32_standalone/esp32_standalone  │
│  • MPU9250 IMU → Complementary Filter        │
│  • BMP280 Barometrik irtifa                  │
│  • model_weights.h → model_infer()  @ 50 Hz │
│  • PWM → 4×ESC → Motor                      │
└──────────────────────────────────────────────┘
```

> **PC model modu** alternatifinde inference PC'de yapılır, ESP32 sadece sensör gönderir ve motor komutu alır. Gecikmeye karşı daha hassastır.

---

## 2. Donanım Bağlantıları

### ESP32 Pin Haritası

| Bileşen | Pin |
|---|---|
| I2C SDA (MPU9250 + BMP280) | GPIO 21 |
| I2C SCL (MPU9250 + BMP280) | GPIO 22 |
| ESC FL (Front-Left, CCW) | GPIO 13 |
| ESC FR (Front-Right, CW) | GPIO 12 |
| ESC BL (Back-Left, CW) | GPIO 14 |
| ESC BR (Back-Right, CCW) | GPIO 27 |

### Sensör I2C Adresleri

| Sensör | Adres |
|---|---|
| MPU9250 (IMU) | 0x68 |
| BMP280 (Baro) | 0x76 veya 0x77 (otomatik dener) |

### Kanat Düzeni (yukarıdan)

```
    FRONT
FL(CCW)  FR(CW)
   \      /
    [ESP32]
   /      \
BL(CW)  BR(CCW)
```

---

## 3. Yazılım Kurulumu (PC)

```bash
pip install -r requirements.txt
# requirements.txt: stable-baselines3, mujoco, gymnasium, pynput, numpy
```

---

## 4. Model → ESP32 Aktarımı

Bu adım yalnızca **ESP32 Standalone modu** için gereklidir.

```bash
# En iyi modeli C header'a çevir
python deploy/export_to_esp32.py \
    models/trained/best/best_model.zip \
    --norm  models/trained/best/vec_normalize.pkl \
    --out   firmware/esp32_standalone/model_weights.h
```

Üretilen dosya: `firmware/esp32_standalone/model_weights.h`  
İçerik: `OBS_MEAN`, `OBS_VAR`, `W1/B1`, `W2/B2`, `W3/B3` C dizileri + `model_infer()` fonksiyonu

> **Ağ mimarisi H=64 ise ~21 KB, H=128 ise ~72 KB** Flash kullanır. SRAM'i etkilemez.

---

## 5. ESP32 Firmware Yapılandırması ve Flash

### 5.1 Arduino IDE Kütüphaneleri

Arduino IDE → Library Manager'dan yükle:
- **MPU9250** by hideakitai
- **Adafruit BMP280 Library**

### 5.2 Dosyayı Aç

```
firmware/esp32_standalone/esp32_standalone.ino
```

### 5.3 WiFi ve IP Ayarları

Dosyanın üstündeki sabitleri düzenle:

```cpp
static const char*    WIFI_SSID    = "EV_WIFI_ADI";
static const char*    WIFI_PASS    = "WIFI_SIFRESI";
static const char*    PC_IP        = "192.168.1.101";  // PC'nin IP'si
static const uint16_t LOCAL_PORT   = 8888;   // ESP32 dinleme portu
static const uint16_t PC_DATA_PORT = 8889;   // PC'ye telemetri portu
```

> PC'nin IP'sini öğrenmek için: `ipconfig` (Windows) → IPv4 Address

### 5.4 Arduino IDE Board Ayarları

| Ayar | Değer |
|---|---|
| Board | ESP32 Dev Module |
| Flash Size | 4MB |
| Partition Scheme | **No OTA (2MB APP / 2MB SPIFFS)** |
| Upload Speed | 921600 |

### 5.5 Flash ve İlk Test

1. ESP32'yi USB'ye bağla
2. **Upload** düğmesine bas (→)
3. Serial Monitor aç (115200 baud)
4. Beklenen çıktı:
   ```
   === Whoop Drone Standalone ===
   WiFi connecting.......
   Connected! IP: 192.168.1.100
   UDP listening on port 8888
   Gyro calibration (keep still 5 s)...
   IMU OK
   BMP280 OK – base alt: 0.00 m
   Arming ESCs (3 s) – keep props clear...
   Armed. Ready.
   ```

> ⚠️ **PERVANE TAKMADAN ÖNCE** bu adımı tamamla. ESC arming sırasında motorlar çalışır.

---

## 6. Uçuş Modları

### 6a. ESP32 Standalone Modu (ÖNERİLEN)

Model ESP32'de çalışır. PC sadece hedef pozisyon gönderir.
Gecikmeye en dayanıklı mod — WiFi kesilse bile drone son hedefte askıda kalır.

```bash
python deploy/rc_control.py \
    --mode esp32 \
    --esp32 192.168.1.100 \
    --target-z 1.0
```

PC → ESP32 paket formatı (16 byte):
```
float32[4] = { target_x, target_y, target_z, target_yaw }
```

Özel sentineller:
```
z = -1.0  →  Kill: motorları kes, arm'ı kaldır
z = -2.0  →  IMU recalibrate: motorları kes + calibrateAccelGyro() çağır
```

> **WiFi kesilirse:** Son bilinen hedefte hover devam eder. Serial Monitor'da `[WARN] No waypoint` uyarısı çıkar. Drone güvenli inişe iner, motorları KESMEZ — bağlantıyı geri getir veya fiziksel olarak müdahale et.

---

### 6b. PC Model Modu

Inference PC'de, motor komutu UDP ile gönderilir. WiFi gecikmesine duyarlı.

```bash
python deploy/rc_control.py \
    --mode pc \
    --model  models/trained/best/best_model.zip \
    --norm   models/trained/best/vec_normalize.pkl \
    --esp32  192.168.1.100 \
    --target-z 1.0
```

Firmware: `firmware/esp32_drone/esp32_drone.ino`

PC ← ESP32 paket (28 byte, 200 Hz):
```
float32[7] = { ax, ay, az, gx, gy, gz, altitude }
```

PC → ESP32 paket (16 byte, 50 Hz):
```
float32[4] = { motor_FL, motor_FR, motor_BL, motor_BR }  ∈ [0, 1]
```

> **500 ms motor komutu gelmezse** `esp32_drone.ino` **motorları otomatik keser** — bu moddaki tek güvenlik katmanı.

---

## 7. Klavye Kontrol Referansı

| Tuş | Eylem |
|---|---|
| `W` / `S` | İleri / Geri (X ekseni) |
| `A` / `D` | Sol / Sağ (Y ekseni) |
| `Q` / `E` | Yukarı / Aşağı (Z ekseni) |
| `Z` / `C` | Yaw sola (CCW) / Yaw sağa (CW) |
| `K` | IMU Kalibrasyonu başlat |
| `SPACE` | Pozisyon dondur (carrot durdur) |
| `R` | Home — X/Y sıfırla, hedef `[0,0,target_z]` |
| `X` / `ESC` | Acil durdur — motorları kes |

### Carrot Sistemi

- Tuşa basınca `final_target` (nihai hedef) kayar
- `target` (carrot) ise `final_target`'e doğru her döngüde **0.04 m** yaklaşır (~2 m/s)
- **Tuşu bırakınca** carrot o anda bulunduğu noktada **donar** — drone yumuşakça durur
- Bu sayede ani konum sıçraması olmaz, drone akıcı hareket eder

---

## 8. IMU Kalibrasyon Prosedürü

### Firmware İçi Kalibrasyon (Önyükleme)

ESP32 her açılışta `calibrateAccelGyro()` çağırır (~5 saniye).  
Drone'u **düz ve hareketsiz** tut, kablolardan sarsmadan bekle.

### Uçuş Sırasında Yeniden Kalibrasyon (K Tuşu)

1. Drone'u **düz bir yüzeye** koy (motor kapalı veya hareketsiz zemin)
2. `K` tuşuna bas
3. **PC modu (`--mode pc`):** 2 saniye (~100 örnek) gyro ve ivmeölçer bias toplanır, otomatik uygulanır. Drone uçmaya devam eder.
4. **ESP32 modu (`--mode esp32`):** `z=-2.0` sentinel gönderilir. Firmware `stop_motors()` + `calibrateAccelGyro()` (~5 sn) çağırır. **Motor kesilir, drone yerde olmalı.** Kalibrasyon bittikten sonra güç döngüsü veya yeniden flash gerekir.

Terminal çıktısı (PC modu):
```
[CAL] Tamamlandı  gyro=[+0.0012,−0.0034,+0.0008]  accel_xy=[+0.0021,−0.0015]
```

> ESP32 modunda K tuşuna havada **kesinlikle basma** — motorlar kesilir.

---

## 9. Güvenlik ve Acil Durdurma

### Otomatik Güvenlik Kontrolleri

**ESP32 Standalone (`esp32_standalone.ino`) — Her 50 Hz döngüsünde:**

| Kontrol | Eşik | Tepki |
|---|---|---|
| Minimum irtifa | 12 cm | Motorları kes |
| Maksimum yatış | ~66° (`qw < 0.40`) | Motorları kes |
| WiFi timeout | 2 sn | Uyarı yaz, hover devam |

**PC Modu (`esp32_drone.ino`) — Her loop() iterasyonunda:**

| Kontrol | Eşik | Tepki |
|---|---|---|
| Motor komutu timeout | 500 ms | **Motorları kes** |

**PC (`rc_control.py --mode pc`) — Her 50 Hz döngüsünde:**

| Kontrol | Eşik | Tepki |
|---|---|---|
| Minimum irtifa | 12 cm | emergency_stop() |
| Maksimum yatış | 50° | emergency_stop() |

### Manuel Acil Durdurma

- `X` veya `ESC` tuşu → 200 ms boyunca 20 kez sıfır komut gönderilir
- ESP32 standalone: `z=-1.0` kill paketi
- PC modu: `emergency_stop()` doğrudan çağrılır

### Maksimum Gaz Sınırı

Kod içinde `MAX_THROTTLE_SAFE = 0.80` (ESC ve motorlar % 80 ile kısıtlı).

---

## 10. Simülasyonda Test

Gerçek uçuştan önce simülasyonda test et:

```bash
python test_manual.py \
    --model  models/trained/best/best_model.zip \
    --vec-norm models/trained/best/vec_normalize.pkl \
    --fps 50
```

| Tuş | Simülasyon |
|---|---|
| ↑↓←→ | Hedef XY |
| Q/E | Hedef Z |
| Z/C | Yaw |
| R | Sıfırla |
| ESC/X | Çık |

Carrot sistemi simülasyonda da aktif — aynı davranışı gözlemle.

---

## 11. Sorun Giderme

| Belirti | Olası Neden | Çözüm |
|---|---|---|
| Drone kalkınca hemen devrilir | `model_weights.h` yanlış model | `export_to_esp32.py` tekrar çalıştır |
| Drone havada yavaşça sürüklenir | IMU bias var | `K` tuşuyla yeniden kalibre et (yerde!) |
| ESP32 WiFi'a bağlanmıyor | SSID/PASS yanlış veya 5GHz ağ | 2.4 GHz ağ kullan, firmware düzelt |
| Serial'de "MPU9250 not found" | I2C kablolama sorunu | SDA=21 SCL=22 kontrol et |
| Serial'de "BMP280 not found" | Adres 0x76/0x77 yanlış | Jumper'ı kontrol et veya `bmp.begin(0x77)` dene |
| UDP paket gelmiyor | PC_IP yanlış | `ipconfig` ile PC IP'sini teyit et |
| W/S/A/D basınca drone tepki vermiyor | XY konumlama yok | Normal — drone relatıf hareket eder, GPS yok |
| Drone yatay drift yapıyor | Gyro bias veya rüzgar | K ile kalibre et, kapalı ortamda uç |
| K tuşu esp32 modunda **havada** basıldı | Motor kesildi! | Güç döngüsü gerekli, yeniden flash |
| Motor sesleri farklı / vibrasyon | Pervane dengesiz veya ters | CW/CCW düzenini kontrol et (bkz. §2) |
| Drone yüksekliği tutmuyor | BMP280 çözünürlüğü düşük | Kapalı ortamda, hava akımından uzakta uç |
| Serial: `[WARN] No waypoint` | PC bağlantısı kesildi | rc_control.py'yi yeniden başlat |
| PC modda drone aniden düşüyor | WiFi gecikmesi 500ms aştı | esp32 standalone moda geç |

---

## Hızlı Başlangıç Özeti

```
1. python deploy/export_to_esp32.py  <model.zip> --norm <vn.pkl>
2. Arduino IDE → firmware/esp32_standalone.ino → WiFi/IP düzenle → Flash
3. Drone düz zemine koy → Güç ver → Serial'de "Armed. Ready." bekle
4. python deploy/rc_control.py --mode esp32 --esp32 <ESP32_IP>
5. PERVANE OLMADAN KISA TEST  →  K ile kalibre  →  PERVANE TAK  →  Uç
```
