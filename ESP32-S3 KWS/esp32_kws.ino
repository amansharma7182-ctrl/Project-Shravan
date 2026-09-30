
#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <driver/i2s_std.h>
#include <driver/gpio.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <MicroTFLite.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>

#include "model_data.h"

// Favor execution speed for the DSP/KWS hot path on ESP32-S3.
#pragma GCC optimize ("O3")

#if !defined(CONFIG_IDF_TARGET_ESP32S3)
#error "This firmware is intended for ESP32-S3. Select an ESP32-S3 board in PlatformIO/Arduino."
#endif

enum class RunMode : uint8_t {
    KWS_ONLY,
    WEBSOCKET_ONLY,
    COMBINED
};

namespace cfg {

constexpr RunMode MODE = RunMode::COMBINED;
constexpr bool USE_KWS = (MODE != RunMode::WEBSOCKET_ONLY);
constexpr bool USE_NET = (MODE != RunMode::KWS_ONLY);

// ---- network ---------------------------------------------------------------
constexpr const char* WIFI_SSID = "YOUR_WIFI";
constexpr const char* WIFI_PASS = "YOUR_PASS";
constexpr const char* SERVER_IP = "SERVER_IP";
constexpr uint16_t SERVER_PORT = 8000;
constexpr const char* SERVER_PATH = "/audio";

// ---- INMP441 ---------------------------------------------------------------
constexpr i2s_port_t I2S_PORT = I2S_NUM_0;
constexpr int PIN_SCK = 4;
constexpr int PIN_WS = 5;
constexpr int PIN_SD = 16;

// LEDpins
constexpr int WIFI_PIN = 8;
constexpr int WEBSOCKET_PIN = 9;
constexpr int ASR_PIN = 12;

// INMP441 delivers 24-bit audio in a 32-bit I2S slot.
constexpr int MIC_GAIN_SHIFT = 16;
constexpr int MIC_BLOCK = 512;   // 64 ms of 32-bit mono samples at 8 kHz
constexpr int DMA_BUFS = 4;      // 256 ms DMA slack
constexpr int MIC_TASK_CORE = 0; // keep I2S capture away from Arduino loop/KWS

// ---- audio / KWS -----------------------------------------------------------
constexpr int SAMPLE_RATE = 8000;
constexpr int RING_SAMPLES = 8000;
constexpr int KWS_SAMPLES = 8000;

constexpr int FRAME_LEN = 240;
constexpr int FRAME_STEP = 160;
constexpr int FFT_LEN = 256;
constexpr int NUM_FRAMES = 49;
constexpr int NUM_MELS = 40;
constexpr float MEL_LOW_HZ = 80.0f;
constexpr float MEL_HIGH_HZ = 3800.0f;
constexpr int MEL_WEIGHTS_MAX = 2 * (FFT_LEN / 2 + 1);

// ---- detection -------------------------------------------------------------
constexpr float KWS_THRESHOLD = 0.78f;
constexpr int KWS_HITS = 1;
constexpr uint32_t KWS_STRIDE_MS = 900;
constexpr int KWS_GATE_PEAK = 750;
constexpr int KWS_ARENA_BYTES = 61 * 1024;

// ---- ASR -------------------------------------------------------------------
constexpr int STREAM_SECONDS = 15;
constexpr int STREAM_CHUNK = 256;
constexpr int STREAM_GATE_PEAK = 700;
constexpr uint32_t STREAM_DRAIN_MS = 300;
constexpr uint32_t STREAM_LOG_MS = 250;

// ---- resource monitor ------------------------------------------------------
// S3 monitor uses the actual internal 8-bit-capable heap size at runtime.
// This avoids assuming that all physical SRAM is available to the application.
constexpr uint32_t RAM_BUDGET_KB = 256;
constexpr float CPU_BUDGET_PCT = 10.0f;
constexpr uint32_t MONITOR_MS = 1000;

// KWS runs every 500 ms. Short CPU bursts during Mel/FFT/TFLite are expected.
// The S3 should be judged primarily by sustained/average CPU, not the instant peak.

// ---- main loop -------------------------------------------------------------
constexpr uint32_t LOOP_DELAY_MS = 2;
constexpr uint32_t NET_LOOP_MS = 5;

// ---- logging ---------------------------------------------------------------
constexpr unsigned long LOG_BAUD = 115200;
constexpr int LOG_LINE_MAX = 150;

static_assert(KWS_SAMPLES <= RING_SAMPLES, "ring must hold one KWS window");
static_assert(
    (NUM_FRAMES - 1) * FRAME_STEP + FRAME_LEN <= KWS_SAMPLES,
    "KWS framing out of bounds"
);
static_assert(MIC_GAIN_SHIFT >= 8 && MIC_GAIN_SHIFT <= 16, "bad mic gain shift");

// Default pins target the common ESP32-S3-DevKitC-1 / N8R8 class boards.
// GPIO 26-37 are intentionally avoided because module variants can reserve
// them for flash/PSRAM.

} // namespace cfg


void LED_setup () {
    pinMode (cfg::WIFI_PIN, OUTPUT);
    pinMode (cfg::WEBSOCKET_PIN, OUTPUT);
    pinMode (cfg::ASR_PIN, OUTPUT);
}


class Logger {
public:
    void begin() {
        Serial.begin(cfg::LOG_BAUD);
    }

    void line(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        emit(true, fmt, ap);
        va_end(ap);
    }

    void live(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        emit(false, fmt, ap);
        va_end(ap);
    }

private:
    void emit(bool newline, const char* fmt, va_list ap) {
        const int lim = cfg::LOG_LINE_MAX;
        const uint32_t ms = millis();

        buf_[0] = '\r';

        int n = snprintf(
            buf_ + 1,
            lim,
            "[%4lu.%03lu] ",
            (unsigned long)(ms / 1000UL),
            (unsigned long)(ms % 1000UL)
        );
        if (n < 0) return;
        if (n >= lim) n = lim - 1;

        int body = n;
        int add = vsnprintf(
            buf_ + 1 + body,
            lim - body,
            fmt,
            ap
        );
        if (add < 0) add = 0;
        if (body + add > lim - 1) {
            body = lim - 1;
        } else {
            body += add;
        }

        int len = 1 + body;

        if (!newline) {
            for (int pad = prevLive_ - body; pad > 0 && len < lim + 1; --pad) {
                buf_[len++] = ' ';
            }
            prevLive_ = body;
        } else {
            prevLive_ = 0;
            buf_[len++] = '\n';
        }

        Serial.write((const uint8_t*)buf_, len);
    }

    char buf_[cfg::LOG_LINE_MAX + 2];
    int prevLive_ = 0;
};

// ============================================================================
// [resource_monitor]
// ============================================================================

class ResourceMonitor {
public:
    void begin() {
        totalInternalBytes_ = heap_caps_get_total_size(caps());
        update(true);
    }

    void update(bool force = false) {
        const uint32_t now = millis();
        if (!force && (now - lastMs_) < cfg::MONITOR_MS) return;

        const UBaseType_t count = uxTaskGetSystemState(
            tasks_,
            TASK_CAPACITY,
            nullptr
        );

        uint32_t total = 0;
        uint32_t idle = 0;

        for (UBaseType_t i = 0; i < count; ++i) {
            total += tasks_[i].ulRunTimeCounter;

            // Covers IDLE0, IDLE1, and any future IDLEx task names.
            if (strncmp(tasks_[i].pcTaskName, "IDLE", 4) == 0) {
                idle += tasks_[i].ulRunTimeCounter;
            }
        }

        if (valid_) {
            const uint32_t dt = total - lastTotal_;
            const uint32_t di = idle - lastIdle_;

            if (dt > 0) {
                float value = 100.0f * (1.0f - (float)di / (float)dt);
                if (value < 0.0f) value = 0.0f;
                if (value > 100.0f) value = 100.0f;
                cpu_ = value;
            }
        } else {
            valid_ = true;
        }

        lastTotal_ = total;
        lastIdle_ = idle;
        lastMs_ = now;
    }

    float cpu() const {
        return cpu_;
    }

    uint32_t ramUsedKB() const {
        return usedKB(heap_caps_get_free_size(caps()));
    }

    uint32_t ramPeakKB() const {
        return usedKB(heap_caps_get_minimum_free_size(caps()));
    }

    uint32_t ramTotalKB() const {
        return totalInternalBytes_ / 1024U;
    }

    bool cpuOver() const {
        return cpu_ > cfg::CPU_BUDGET_PCT;
    }

    bool ramOver() const {
        return ramUsedKB() > cfg::RAM_BUDGET_KB;
    }

private:
    static uint32_t caps() {
        return MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    }

    enum { TASK_CAPACITY = 48 };

    uint32_t usedKB(size_t freeBytes) const {
        const size_t freeBytesClamped =
            freeBytes > totalInternalBytes_ ? totalInternalBytes_ : freeBytes;
        return (uint32_t)((totalInternalBytes_ - freeBytesClamped) / 1024U);
    }

    TaskStatus_t tasks_[TASK_CAPACITY];
    size_t totalInternalBytes_ = 0;
    uint32_t lastTotal_ = 0;
    uint32_t lastIdle_ = 0;
    uint32_t lastMs_ = 0;
    float cpu_ = 0.0f;
    bool valid_ = false;
};

struct AudioStats {
    int peak = 0;
    int avgAbs = 0;

    static AudioStats compute(const int16_t* pcm, int n) {
        int peak = 0;
        uint32_t sum = 0;

        for (int i = 0; i < n; ++i) {
            int v = pcm[i];
            int a = v < 0 ? -v : v;
            if (a > peak) peak = a;
            sum += (uint32_t)a;
        }

        AudioStats out;
        out.peak = peak;
        out.avgAbs = n > 0 ? (int)(sum / (uint32_t)n) : 0;
        return out;
    }
};

// ============================================================================
// [i2s_mic]
// ============================================================================

class I2SMic {
public:
    // Bytes this class pulls from the heap: the 1 s ring buffer plus the
    // per-DMA-block raw capture scratch. Neither is a hidden static buffer -
    // both are heap_caps_malloc'd in begin() and counted by ResourceMonitor.
    static constexpr size_t footprintBytes() {
        return (size_t)cfg::RING_SAMPLES * sizeof(int16_t)
             + (size_t)cfg::MIC_BLOCK * sizeof(int32_t);
    }

    bool begin() {
        ring_ = (int16_t*)heap_caps_malloc(
            (size_t)cfg::RING_SAMPLES * sizeof(int16_t),
            MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL
        );

        if (!ring_) return false;

        memset(ring_, 0, (size_t)cfg::RING_SAMPLES * sizeof(int16_t));

        raw_ = (int32_t*)heap_caps_malloc(
            (size_t)cfg::MIC_BLOCK * sizeof(int32_t),
            MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL
        );

        if (!raw_) {
            cleanupBuffers();
            return false;
        }

        i2s_chan_config_t chanCfg = I2S_CHANNEL_DEFAULT_CONFIG(
            cfg::I2S_PORT,
            I2S_ROLE_MASTER
        );
        chanCfg.dma_desc_num = cfg::DMA_BUFS;
        chanCfg.dma_frame_num = cfg::MIC_BLOCK;
        chanCfg.auto_clear = false;

        if (i2s_new_channel(&chanCfg, nullptr, &rx_) != ESP_OK) {
            cleanupBuffers();
            return false;
        }

        i2s_std_config_t stdCfg = {
            .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(cfg::SAMPLE_RATE),
            .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                I2S_DATA_BIT_WIDTH_32BIT,
                I2S_SLOT_MODE_MONO
            ),
            .gpio_cfg = {
                .mclk = I2S_GPIO_UNUSED,
                .bclk = (gpio_num_t)cfg::PIN_SCK,
                .ws = (gpio_num_t)cfg::PIN_WS,
                .dout = I2S_GPIO_UNUSED,
                .din = (gpio_num_t)cfg::PIN_SD,
                .invert_flags = {
                    .mclk_inv = false,
                    .bclk_inv = false,
                    .ws_inv = false,
                },
            },
        };

        // INMP441 L/R = GND -> left slot.
        stdCfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

        if (i2s_channel_init_std_mode(rx_, &stdCfg) != ESP_OK) {
            cleanupDriver();
            cleanupBuffers();
            return false;
        }

        if (i2s_channel_enable(rx_) != ESP_OK) {
            cleanupDriver();
            cleanupBuffers();
            return false;
        }

        if (xTaskCreatePinnedToCore(
                taskEntry,
                "mic",
                3072,
                this,
                3,
                &task_,
                cfg::MIC_TASK_CORE) != pdPASS) {
            cleanupDriver();
            cleanupBuffers();
            return false;
        }

        return true;
    }

    bool primed() const {
        return primed_;
    }

    uint32_t totalSamples() const {
        return totalSamples_;
    }

    uint32_t overruns() const {
        return overruns_;
    }

    // Copy the latest complete 1-second window into a stable destination.
    // Only the memcpy section is protected, so KWS DSP does NOT stall I2S.
    bool copyLatestWindow(int16_t* dst, size_t dstSamples) {
        if (!dst || dstSamples < (size_t)cfg::KWS_SAMPLES || !primed_) {
            return false;
        }

        portENTER_CRITICAL(&mux_);

        const int start = head_;
        const size_t n = cfg::KWS_SAMPLES;
        const size_t first = cfg::RING_SAMPLES - start;

        if (first >= n) {
            memcpy(dst, ring_ + start, n * sizeof(int16_t));
        } else {
            memcpy(dst, ring_ + start, first * sizeof(int16_t));
            memcpy(
                dst + first,
                ring_,
                (n - first) * sizeof(int16_t)
            );
        }

        portEXIT_CRITICAL(&mux_);
        return true;
    }

    size_t readSince(uint32_t& cursor, int16_t* dst, size_t maxN) {
        portENTER_CRITICAL(&mux_);

        uint32_t behind = totalSamples_ - cursor;

        if (behind > (uint32_t)cfg::RING_SAMPLES) {
            overruns_ += behind - (uint32_t)cfg::RING_SAMPLES;
            cursor = totalSamples_ - (uint32_t)cfg::RING_SAMPLES;
            behind = cfg::RING_SAMPLES;
        }

        const size_t n = behind < (uint32_t)maxN
                       ? (size_t)behind
                       : maxN;

        if (n > 0) {
            int start = head_ - (int)behind;
            if (start < 0) start += cfg::RING_SAMPLES;

            size_t first = cfg::RING_SAMPLES - start;
            if (first > n) first = n;

            memcpy(dst, ring_ + start, first * sizeof(int16_t));

            if (n > first) {
                memcpy(
                    dst + first,
                    ring_,
                    (n - first) * sizeof(int16_t)
                );
            }

            cursor += (uint32_t)n;
        }

        portEXIT_CRITICAL(&mux_);
        return n;
    }

    ~I2SMic() {
        if (task_) {
            vTaskDelete(task_);
            task_ = nullptr;
        }

        cleanupDriver();
        cleanupBuffers();
    }

private:
    static void taskEntry(void* arg) {
        static_cast<I2SMic*>(arg)->captureLoop();
    }

    static int16_t convertSample(int32_t raw) {
        const int32_t x = raw >> cfg::MIC_GAIN_SHIFT;

        // Integer one-pole DC removal. No float work per 8 kHz sample.
        dc_ += (x - dc_) >> 8;
        int32_t y = x - dc_;

        if (y > 32767) y = 32767;
        if (y < -32768) y = -32768;
        return (int16_t)y;
    }

    void captureLoop() {
        const size_t rawBytes = (size_t)cfg::MIC_BLOCK * sizeof(int32_t);

        for (;;) {
            size_t bytes = 0;
            const esp_err_t err = i2s_channel_read(
                rx_,
                raw_,
                rawBytes,
                &bytes,
                pdMS_TO_TICKS(50)
            );

            if (err != ESP_OK || bytes == 0) continue;

            const int n = (int)(bytes / sizeof(int32_t));

            portENTER_CRITICAL(&mux_);

            for (int i = 0; i < n; ++i) {
                ring_[head_] = convertSample(raw_[i]);
                ++head_;
                if (head_ >= cfg::RING_SAMPLES) head_ = 0;
                ++totalSamples_;
            }

            if (totalSamples_ >= (uint32_t)cfg::RING_SAMPLES) {
                primed_ = true;
            }

            portEXIT_CRITICAL(&mux_);
        }
    }

    void cleanupDriver() {
        if (!rx_) return;
        i2s_channel_disable(rx_);
        i2s_del_channel(rx_);
        rx_ = nullptr;
    }

    // Frees every buffer this class pulled from the heap. Kept as one
    // function (rather than one per buffer) so every begin()-failure path
    // releases everything it had allocated so far.
    void cleanupBuffers() {
        if (ring_) {
            heap_caps_free(ring_);
            ring_ = nullptr;
        }
        if (raw_) {
            heap_caps_free(raw_);
            raw_ = nullptr;
        }
    }

    i2s_chan_handle_t rx_ = nullptr;

    int16_t* ring_ = nullptr;
    TaskHandle_t task_ = nullptr;

    int32_t* raw_ = nullptr;

    volatile uint32_t totalSamples_ = 0;
    volatile uint32_t overruns_ = 0;
    volatile int head_ = 0;
    volatile bool primed_ = false;

    static volatile int32_t dc_;
    static portMUX_TYPE mux_;
};

volatile int32_t I2SMic::dc_ = 0;
portMUX_TYPE I2SMic::mux_ = portMUX_INITIALIZER_UNLOCKED;


class LogMelExtractor {
public:
    // Every table this class needs (window, twiddle factors, bit-reversal,
    // mel filterbank, FFT scratch, output features) is heap_caps_malloc'd in
    // begin() instead of living as fixed-size class members. Fixed-size
    // members of a globally-instantiated object land in .bss - invisible to
    // ResourceMonitor - so converting them to heap pointers is what actually
    // makes them show up in the RAM totals.
    static constexpr size_t footprintBytes() {
        return (size_t)cfg::FRAME_LEN * sizeof(float)                    // window_
             + (size_t)(cfg::FFT_LEN / 2) * sizeof(float)                // cos_
             + (size_t)(cfg::FFT_LEN / 2) * sizeof(float)                // sin_
             + (size_t)cfg::FFT_LEN * sizeof(uint16_t)                   // bitRev_
             + (size_t)cfg::NUM_MELS * sizeof(MelBand)                   // bands_
             + (size_t)cfg::MEL_WEIGHTS_MAX * sizeof(float)              // weights_
             + (size_t)cfg::FFT_LEN * sizeof(float)                      // re_
             + (size_t)cfg::FFT_LEN * sizeof(float)                      // im_
             + (size_t)cfg::NUM_FRAMES * cfg::NUM_MELS * sizeof(float);  // features_
    }

    bool begin() {
        if (!alloc()) return false;

        for (int n = 0; n < cfg::FRAME_LEN; ++n) {
            window_[n] =
                (0.5f -
                 0.5f * cosf(2.0f * (float)PI * (float)n / (float)cfg::FRAME_LEN)) /
                32768.0f;
        }

        for (int k = 0; k < cfg::FFT_LEN / 2; ++k) {
            const float a =
                -2.0f * (float)PI * (float)k / (float)cfg::FFT_LEN;
            cos_[k] = cosf(a);
            sin_[k] = sinf(a);
        }

        for (int i = 0; i < cfg::FFT_LEN; ++i) {
            int x = i;
            int r = 0;
            for (int b = 0; b < 8; ++b) {
                r = (r << 1) | (x & 1);
                x >>= 1;
            }
            bitRev_[i] = (uint16_t)r;
        }

        buildMelBands();
        return true;
    }

    // pcmStart is the start index of a stable 1-second PCM window.
    // The ESP32-S3 path passes a linear snapshot with pcmStart = 0.
    const float* compute(const int16_t* ring, int pcmStart) {
        for (int frame = 0; frame < cfg::NUM_FRAMES; ++frame) {
            const int frameStart = pcmStart + frame * cfg::FRAME_STEP;

            for (int i = 0; i < cfg::FRAME_LEN; ++i) {
                int index = frameStart + i;
                if (index >= cfg::RING_SAMPLES) index -= cfg::RING_SAMPLES;
                re_[i] = (float)ring[index] * window_[i];
            }

            memset(
                re_ + cfg::FRAME_LEN,
                0,
                (cfg::FFT_LEN - cfg::FRAME_LEN) * sizeof(float)
            );
            memset(im_, 0, cfg::FFT_LEN * sizeof(float));

            fft();

            float* out = features_ + frame * cfg::NUM_MELS;

            for (int m = 0; m < cfg::NUM_MELS; ++m) {
                const MelBand& b = bands_[m];
                const float* w = weights_ + b.offset;
                float energy = 0.0f;

                for (int j = 0; j < b.count; ++j) {
                    const int k = b.first + j;
                    energy +=
                        (re_[k] * re_[k] + im_[k] * im_[k]) * w[j];
                }

                out[m] = logf(energy + 1e-6f);
            }
        }

        // FIX A: global z-score over all 49x40 values.
        const int N = cfg::NUM_FRAMES * cfg::NUM_MELS;

        float sum = 0.0f;
        float sumSq = 0.0f;

        for (int i = 0; i < N; ++i) {
            const float v = features_[i];
            sum += v;
            sumSq += v * v;
        }

        const float mean = sum / (float)N;
        const float var = (sumSq / (float)N) - (mean * mean);
        const float safeVar = var > 0.0f ? var : 0.0f;
        const float inv = 1.0f / (sqrtf(safeVar) + 1e-6f);

        for (int i = 0; i < N; ++i) {
            features_[i] = (features_[i] - mean) * inv;
        }

        return features_;
    }

    ~LogMelExtractor() {
        heap_caps_free(window_);
        heap_caps_free(cos_);
        heap_caps_free(sin_);
        heap_caps_free(bitRev_);
        heap_caps_free(bands_);
        heap_caps_free(weights_);
        heap_caps_free(re_);
        heap_caps_free(im_);
        heap_caps_free(features_);
    }

private:
    struct MelBand {
        uint16_t first;
        uint16_t count;
        uint16_t offset;
    };

    bool alloc() {
        constexpr uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

        window_ = (float*)heap_caps_aligned_alloc(
            16, (size_t)cfg::FRAME_LEN * sizeof(float), caps);
        cos_ = (float*)heap_caps_aligned_alloc(
            16, (size_t)(cfg::FFT_LEN / 2) * sizeof(float), caps);
        sin_ = (float*)heap_caps_aligned_alloc(
            16, (size_t)(cfg::FFT_LEN / 2) * sizeof(float), caps);
        bitRev_ = (uint16_t*)heap_caps_malloc(
            (size_t)cfg::FFT_LEN * sizeof(uint16_t), caps);
        bands_ = (MelBand*)heap_caps_malloc(
            (size_t)cfg::NUM_MELS * sizeof(MelBand), caps);
        weights_ = (float*)heap_caps_aligned_alloc(
            16, (size_t)cfg::MEL_WEIGHTS_MAX * sizeof(float), caps);
        re_ = (float*)heap_caps_aligned_alloc(
            16, (size_t)cfg::FFT_LEN * sizeof(float), caps);
        im_ = (float*)heap_caps_aligned_alloc(
            16, (size_t)cfg::FFT_LEN * sizeof(float), caps);
        features_ = (float*)heap_caps_aligned_alloc(
            16, (size_t)cfg::NUM_FRAMES * cfg::NUM_MELS * sizeof(float), caps);

        return window_ && cos_ && sin_ && bitRev_ && bands_
            && weights_ && re_ && im_ && features_;
    }

    static float hzToMel(float hz) {
        return 2595.0f * log10f(1.0f + hz / 700.0f);
    }

    static float melToHz(float mel) {
        return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }

    void buildMelBands() {
        float hz[cfg::NUM_MELS + 2];

        const float lo = hzToMel(cfg::MEL_LOW_HZ);
        const float hi = hzToMel(cfg::MEL_HIGH_HZ);

        for (int i = 0; i < cfg::NUM_MELS + 2; ++i) {
            const float mel =
                lo + (hi - lo) *
                ((float)i / (float)(cfg::NUM_MELS + 1));
            hz[i] = melToHz(mel);
        }

        const float binPerHz =
            (float)cfg::FFT_LEN / (float)cfg::SAMPLE_RATE;

        int offset = 0;

        for (int m = 0; m < cfg::NUM_MELS; ++m) {
            const float left = hz[m] * binPerHz;
            const float center = hz[m + 1] * binPerHz;
            const float right = hz[m + 2] * binPerHz;

            MelBand& b = bands_[m];
            b.first = 0;
            b.count = 0;
            b.offset = (uint16_t)offset;

            for (int k = 0; k <= cfg::FFT_LEN / 2; ++k) {
                const float fk = (float)k;
                float w = 0.0f;

                if (fk >= left && fk <= center && center > left) {
                    w = (fk - left) / (center - left);
                } else if (fk > center && fk <= right && right > center) {
                    w = (right - fk) / (right - center);
                }

                if (w <= 0.0f) continue;

                if (b.count == 0) b.first = (uint16_t)k;

                if (offset >= cfg::MEL_WEIGHTS_MAX) break;
                weights_[offset++] = w;
                ++b.count;
            }
        }
    }

    void fft() {
        const int N = cfg::FFT_LEN;

        for (int i = 0; i < N; ++i) {
            const int j = bitRev_[i];
            if (i < j) {
                float t = re_[i];
                re_[i] = re_[j];
                re_[j] = t;

                t = im_[i];
                im_[i] = im_[j];
                im_[j] = t;
            }
        }

        for (int len = 2; len <= N; len <<= 1) {
            const int half = len >> 1;
            const int step = N / len;

            for (int i = 0; i < N; i += len) {
                for (int j = 0; j < half; ++j) {
                    const float wr = cos_[j * step];
                    const float wi = sin_[j * step];

                    const int u = i + j;
                    const int v = u + half;

                    const float tr = re_[v] * wr - im_[v] * wi;
                    const float ti = re_[v] * wi + im_[v] * wr;

                    re_[v] = re_[u] - tr;
                    im_[v] = im_[u] - ti;
                    re_[u] += tr;
                    im_[u] += ti;
                }
            }
        }
    }

    float* window_ = nullptr;
    float* cos_ = nullptr;
    float* sin_ = nullptr;
    uint16_t* bitRev_ = nullptr;
    MelBand* bands_ = nullptr;
    float* weights_ = nullptr;
    float* re_ = nullptr;
    float* im_ = nullptr;
    float* features_ = nullptr;
};

struct KwsProbs {
    enum Label : uint8_t { SIL = 0, UNK = 1, KEY = 2 };

    float silence = 0.0f;
    float unknown = 0.0f;
    float keyword = 0.0f;

    Label label() const {
        Label bestLabel = SIL;
        float best = silence;

        if (unknown > best) {
            bestLabel = UNK;
            best = unknown;
        }
        if (keyword > best) {
            bestLabel = KEY;
        }

        return bestLabel;
    }

    const char* name() const {
        switch (label()) {
            case SIL: return "SIL";
            case UNK: return "UNK";
            default: return "KEY";
        }
    }
};

class KwsModel {
public:
    // Named so App can log it alongside every other buffer's size.
    static constexpr size_t footprintBytes() {
        return (size_t)cfg::KWS_ARENA_BYTES;
    }

    bool begin() {
        // The tensor arena is deliberately internal-DRAM only (never PSRAM),
        // and now heap-allocated so it shows up in ResourceMonitor's totals
        // instead of sitting in .bss as a hidden static buffer. This is the
        // hot TFLite workspace and should stay in internal memory for
        // predictable latency.
        arena_ = (uint8_t*)heap_caps_aligned_alloc(
            16,
            (size_t)cfg::KWS_ARENA_BYTES,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

        if (!arena_) return false;

        return ModelInit(g_model, arena_, cfg::KWS_ARENA_BYTES);
    }

    bool run(const float* features, KwsProbs& out) {
        const int count = cfg::NUM_FRAMES * cfg::NUM_MELS;

        for (int i = 0; i < count; ++i) {
            ModelSetInput(features[i], i);
        }

        if (!ModelRunInference()) return false;

        // Training order:
        // [0] hello_shravan, [1] silence, [2] unknown
        out.keyword = ModelGetOutput(0);
        out.silence = ModelGetOutput(1);
        out.unknown = ModelGetOutput(2);
        return true;
    }

    ~KwsModel() {
        heap_caps_free(arena_);
        arena_ = nullptr;
    }

private:
    uint8_t* arena_ = nullptr;
};

struct WindowReport {
    AudioStats audio;
    KwsProbs probs;
    bool gated = false;
    bool error = false;
    bool triggered = false;
};

class WakeWordDetector {
public:
    // Aggregate of everything this stage allocates: mel/FFT tables, the
    // tensor arena, and the 1 s KWS snapshot. Used by App to report the
    // full named footprint for the KWS path.
    static constexpr size_t footprintBytes() {
        return LogMelExtractor::footprintBytes()
             + KwsModel::footprintBytes()
             + (size_t)cfg::KWS_SAMPLES * sizeof(int16_t);
    }

    bool begin() {
        if (!mel_.begin()) return false;

        // The KWS snapshot is heap-allocated from internal DRAM (never
        // PSRAM), so it shows up in ResourceMonitor's heap totals instead of
        // sitting in .bss as a hidden static buffer.
        snapshot_ = (int16_t*)heap_caps_malloc(
            (size_t)cfg::KWS_SAMPLES * sizeof(int16_t),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
        );

        if (!snapshot_) return false;

        memset(snapshot_, 0, (size_t)cfg::KWS_SAMPLES * sizeof(int16_t));

        return model_.begin();
    }

    void reset() {
        lastMs_ = millis();
        hits_ = 0;
    }

    bool poll(I2SMic& mic, WindowReport& r) {
        const uint32_t now = millis();

        if (!mic.primed()) return false;
        if (now - lastMs_ < cfg::KWS_STRIDE_MS) return false;

        lastMs_ = now;
        r = WindowReport();

        // Take a short atomic snapshot of the live ring. The microphone keeps
        // running while Mel/DSP/TFLite process this stable 1-second window.
        if (!mic.copyLatestWindow(snapshot_, cfg::KWS_SAMPLES)) {
            r.error = true;
            hits_ = 0;
            return true;
        }

        r.audio = computeStats(snapshot_);

        if (r.audio.peak < cfg::KWS_GATE_PEAK) {
            r.gated = true;
            hits_ = 0;
            return true;
        }

        const float* features = mel_.compute(snapshot_, 0);

        if (!model_.run(features, r.probs)) {
            r.error = true;
            hits_ = 0;
            return true;
        }

        const bool hit =
            (r.probs.label() == KwsProbs::KEY) &&
            (r.probs.keyword >= cfg::KWS_THRESHOLD);

        hits_ = hit ? (hits_ + 1) : 0;

        if (hits_ >= cfg::KWS_HITS) {
            hits_ = 0;
            r.triggered = true;
        }

        return true;
    }

    ~WakeWordDetector() {
        heap_caps_free(snapshot_);
        snapshot_ = nullptr;
    }

private:
    static AudioStats computeStats(const int16_t* pcm) {
        return AudioStats::compute(pcm, cfg::KWS_SAMPLES);
    }

    LogMelExtractor mel_;
    KwsModel model_;

    int16_t* snapshot_ = nullptr;
    uint32_t lastMs_ = 0;
    int hits_ = 0;
};

class AsrStreamer {
public:
    explicit AsrStreamer(Logger& log) : log_(log) {}

    void begin() {

        digitalWrite (cfg::WIFI_PIN, HIGH);
        while (!connectWifi(15000)) {
            log_.live("[NET] Wi-Fi failed, retrying...");
        }

        digitalWrite (cfg::WIFI_PIN, LOW);
        digitalWrite (cfg::WEBSOCKET_PIN, HIGH);

        log_.line(
            "[NET] Wi-Fi OK | IP %s | RSSI %d dBm",
            WiFi.localIP().toString().c_str(),
            (int)WiFi.RSSI()
        );

        ws_.begin(cfg::SERVER_IP, cfg::SERVER_PORT, cfg::SERVER_PATH);
        ws_.onEvent([this](WStype_t type, uint8_t*, size_t) {
            if (type == WStype_CONNECTED) {
                log_.line("[NET] WebSocket connected");
                digitalWrite (cfg::WIFI_PIN, LOW);
            } else if (type == WStype_DISCONNECTED) {
                log_.line("[NET] WebSocket disconnected");
                digitalWrite (cfg::WIFI_PIN, HIGH);
            }
        });
        ws_.setReconnectInterval(5000);
    }

    void loop() {
        const uint32_t now = millis();
        if (now - lastLoopMs_ < cfg::NET_LOOP_MS) return;
        lastLoopMs_ = now;
        ws_.loop();
    }

    bool ready() {
        return ws_.isConnected();
    }

    void stream(I2SMic& mic, ResourceMonitor& mon) {
        const uint32_t total =
            (uint32_t)cfg::STREAM_SECONDS * (uint32_t)cfg::SAMPLE_RATE;

        const uint32_t chunkSize = (uint32_t)cfg::STREAM_CHUNK;

        int16_t chunk[cfg::STREAM_CHUNK];

        uint32_t cursor = mic.totalSamples();
        const uint32_t overrun0 = mic.overruns();
        const uint32_t t0 = millis();

        uint32_t done = 0;
        uint32_t lost = 0;
        uint32_t lastLog = 0;
        float cpuSum = 0.0f;
        uint32_t cpuN = 0;

        log_.line("[ASR] Recording started");

        while (done < total && ws_.isConnected()) {
            ws_.loop();
            mon.update();

            const uint32_t left = total - done;
            const size_t wanted =
                left < chunkSize ? (size_t)left : (size_t)chunkSize;

            const size_t n = mic.readSince(cursor, chunk, wanted);

            if (n == 0) {
                delay(2);
                continue;
            }

            const AudioStats st = AudioStats::compute(chunk, (int)n);

            if (st.peak < cfg::STREAM_GATE_PEAK) {
                memset(chunk, 0, n * sizeof(int16_t));
            }

            if (!sendChunk(chunk, n)) {
                lost += (uint32_t)n;
            }

            done += (uint32_t)n;

            if (millis() - lastLog >= cfg::STREAM_LOG_MS) {
                lastLog = millis();
                cpuSum += mon.cpu();
                ++cpuN;

                log_.live(
                    "[ASR %4.1f/%2ds] pk %5d avg %5d | sent %5.1fKB | RAM %3luK%s | CPU %3.0f%%%s",
                    (float)done / (float)cfg::SAMPLE_RATE,
                    cfg::STREAM_SECONDS,
                    st.peak,
                    st.avgAbs,
                    (float)(done - lost) * 2.0f / 1024.0f,
                    (unsigned long)mon.ramUsedKB(),
                    mon.ramOver() ? "!" : "",
                    mon.cpu(),
                    mon.cpuOver() ? "!" : ""
                );
            }
        }

        // Guaranteed drain pass for queued TCP frames.
        const uint32_t flushStart = millis();
        while (millis() - flushStart < cfg::STREAM_DRAIN_MS) {
            ws_.loop();
            delay(10);
        }

        const float avgCpu = cpuN ? (cpuSum / (float)cpuN) : mon.cpu();
        const uint32_t totalLost =
            lost + (mic.overruns() - overrun0);

        log_.line(
            "[ASR done %.1fs] sent %.1fKB drop %lu | RAM %3luK peak %3luK | CPU avg %.0f%%%s",
            (millis() - t0) / 1000.0f,
            (float)(done - lost) * 2.0f / 1024.0f,
            (unsigned long)totalLost,
            (unsigned long)mon.ramUsedKB(),
            (unsigned long)mon.ramPeakKB(),
            avgCpu,
            avgCpu > cfg::CPU_BUDGET_PCT ? "!" : ""
        );
    }

private:
    bool connectWifi(uint32_t timeoutMs) {
        WiFi.mode(WIFI_STA);
        WiFi.begin(cfg::WIFI_SSID, cfg::WIFI_PASS);

        const uint32_t start = millis();

        while (WiFi.status() != WL_CONNECTED) {
            if (millis() - start > timeoutMs) return false;
            delay(250);
        }

        return true;
    }

    bool sendChunk(int16_t* data, size_t n) {
        const size_t bytes = n * sizeof(int16_t);

        for (int tries = 0; tries < 10; ++tries) {
            if (ws_.sendBIN((uint8_t*)data, bytes)) return true;
            ws_.loop();
            delayMicroseconds(500);
        }

        return false;
    }

    Logger& log_;
    WebSocketsClient ws_;
    uint32_t lastLoopMs_ = 0;
};

class App {
public:
    App() : asr_(log_) {}

    void begin() {
        log_.begin();
        delay(500);

        mon_.begin();
        mon_.update(true);

        if (!mic_.begin()) {
            halt("I2S mic init failed (check INMP441 wiring/pins)");
        }

        if (cfg::USE_KWS && !kws_.begin()) {
            halt("KWS init failed (model / arena / snapshot - out of internal RAM?)");
        }

        if (cfg::USE_NET) {
            asr_.begin();
        }

        mon_.update(true);

        log_.line(
            "[INIT] %s | ESP32-S3 | INMP441 | %d Hz | DMA %d x %d | arena %luK DRAM | INT RAM %lu/%luK (peak %luK) | PSRAM unused | CPU %.0f%%",
            modeName(),
            cfg::SAMPLE_RATE,
            cfg::DMA_BUFS,
            cfg::MIC_BLOCK,
            (unsigned long)(cfg::KWS_ARENA_BYTES / 1024U),
            (unsigned long)mon_.ramUsedKB(),
            (unsigned long)mon_.ramTotalKB(),
            (unsigned long)mon_.ramPeakKB(),
            mon_.cpu()
        );

        // Every persistent buffer above (mic ring/raw, mel/FFT tables, tensor
        // arena, KWS snapshot) is now heap_caps_malloc'd - nothing is left
        // sitting in .bss as a hidden static allocation, so the RAM numbers
        // in [INIT] already include all of it. This line just names where
        // the bytes went, so "total RAM usage" isn't a black box.
        const size_t micBytes = I2SMic::footprintBytes();
        const size_t kwsBytes = cfg::USE_KWS ? WakeWordDetector::footprintBytes() : 0;
        const size_t explicitBytes = micBytes + kwsBytes;

        log_.line(
            "[MEM] mic(ring+raw) %luB | kws(mel+arena+snap) %luB | named_total %.1fK of %luK used",
            (unsigned long)micBytes,
            (unsigned long)kwsBytes,
            (float)explicitBytes / 1024.0f,
            (unsigned long)mon_.ramUsedKB()
        );
    }

    void tick() {
        mon_.update();

        if (cfg::USE_NET) {
            asr_.loop();
        }

        if (cfg::USE_KWS) {
            WindowReport r;

            if (kws_.poll(mic_, r)) {
                logWindow(r);

                if (r.triggered) {
                    onWake();
                }
            }
        } else if (asr_.ready()) {
            asr_.stream(mic_, mon_);
            delay(2000);
        }

        delay(cfg::LOOP_DELAY_MS);
    }

private:
    const char* modeName() const {
        if (cfg::MODE == RunMode::KWS_ONLY) return "KWS_ONLY";
        if (cfg::MODE == RunMode::WEBSOCKET_ONLY) return "WEBSOCKET_ONLY";
        return "COMBINED";
    }

    void logWindow(const WindowReport& r) {
        mon_.update();

        const char* ramFlag = mon_.ramOver() ? "!" : "";
        const char* cpuFlag = mon_.cpuOver() ? "!" : "";

        if (r.gated) {
            log_.live(
                "[KWS] pk %5d avg %5d | S --- U --- K --- | RAM %3luK%s | CPU %3.0f%%%s | quiet \n",
                r.audio.peak,
                r.audio.avgAbs,
                (unsigned long)mon_.ramUsedKB(),
                ramFlag,
                mon_.cpu(),
                cpuFlag
            );
            return;
        }

        if (r.error) {
            log_.line(
                "[KWS] pk %5d avg %5d | S --- U --- K --- | RAM %3luK%s | CPU %3.0f%%%s | INFERENCE FAILED",
                r.audio.peak,
                r.audio.avgAbs,
                (unsigned long)mon_.ramUsedKB(),
                ramFlag,
                mon_.cpu(),
                cpuFlag
            );
            return;
        }

        log_.line(
            "[KWS] pk %5d avg %5d | S %3.0f%% U %3.0f%% K %3.0f%% | RAM %3luK%s | CPU %3.0f%%%s | %s%s",
            r.audio.peak,
            r.audio.avgAbs,
            r.probs.silence * 100.0f,
            r.probs.unknown * 100.0f,
            r.probs.keyword * 100.0f,
            (unsigned long)mon_.ramUsedKB(),
            ramFlag,
            mon_.cpu(),
            cpuFlag,
            r.probs.name(),
            r.triggered ? " >>> WAKE" : ""
        );
    }

    void onWake() {
        digitalWrite (cfg::ASR_PIN, HIGH);
        digitalWrite (cfg::WEBSOCKET_PIN, LOW);
        log_.line("[KWS] >>> KEYWORD TRIGGERED - STARTING ASR <<<");

        if (!cfg::USE_NET) {
            log_.line("[TEST] wake detected (network disabled in KWS_ONLY)");
        } else if (asr_.ready()) {
            // Keep capture running during ASR: the same ring feeds the
            // WebSocket stream, just as the old analog implementation did.
            asr_.stream(mic_, mon_);
        } else {
            log_.line("[ASR] WebSocket offline - upload aborted");
        }

        kws_.reset();
        digitalWrite (cfg::WEBSOCKET_PIN, HIGH);
        digitalWrite (cfg::ASR_PIN, LOW);
        log_.line("[KWS] Listening");
    }

    void halt(const char* msg) {
        for (;;) {
            log_.line("[FATAL] %s", msg);
            delay(2000);
        }
    }

    Logger log_;
    ResourceMonitor mon_;
    I2SMic mic_;
    WakeWordDetector kws_;
    AsrStreamer asr_;
};

App app;

void setup() {
    LED_setup();
    app.begin();
}

void loop() {
    app.tick();
}
