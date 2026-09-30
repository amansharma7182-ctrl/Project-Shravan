import time
import numpy as np
import tensorflow as tf
import sounddevice as sd

MODEL_PATH = "kws_model_8khz_int8.tflite"
CLASSES = ["hello_shravan", "silence", "unknown"]

SAMPLE_RATE = 8000
CLIP_DURATION_MS = 1000
EXPECTED_SAMPLES = int(SAMPLE_RATE * (CLIP_DURATION_MS / 1000.0))

FRAME_LENGTH = 240
FRAME_STEP = 160
FFT_LENGTH = 256
NUM_MEL_BINS = 40

HOP_SAMPLES = 800
audio_ring_buffer = np.zeros(EXPECTED_SAMPLES, dtype=np.float32)


def get_log_mel_spectrogram(audio_np):
    audio_tensor = tf.convert_to_tensor(audio_np, dtype=tf.float32)

    stft = tf.signal.stft(
        audio_tensor,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH
    )
    spectrogram = tf.abs(stft)

    num_spectrogram_bins = stft.shape[-1]
    lower_edge_hertz = 80.0
    upper_edge_hertz = 3800.0

    linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        NUM_MEL_BINS,
        num_spectrogram_bins,
        SAMPLE_RATE,
        lower_edge_hertz,
        upper_edge_hertz
    )

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_matrix, 1)
    mel_spectrogram.set_shape(
        spectrogram.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:])
    )

    log_mel = tf.math.log(mel_spectrogram + 1e-6)

    min_db = -12.0
    max_db = 3.0
    normalized = (log_mel - min_db) / (max_db - min_db)
    normalized = tf.clip_by_value(normalized, 0.0, 1.0)

    return tf.expand_dims(tf.expand_dims(normalized, -1), 0)


def main():
    global audio_ring_buffer

    print(f"Loading TFLite model: {MODEL_PATH}...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]

    print(f"Model Input Shape: {input_details[0]['shape']}, Dtype: {input_dtype}")
    print(f"Model Output Shape: {output_details[0]['shape']}")

    def audio_callback(indata, frames, time_info, status):
        global audio_ring_buffer

        if status:
            print(status)

        new_samples = indata[:, 0]
        audio_ring_buffer = np.roll(audio_ring_buffer, -len(new_samples))
        audio_ring_buffer[-len(new_samples):] = new_samples

    print("\nStarting live microphone listener... Speak into your mic! (Press Ctrl+C to stop)\n")

    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=HOP_SAMPLES,
        callback=audio_callback
    ):
        while True:
            features = get_log_mel_spectrogram(audio_ring_buffer)

            if input_dtype == np.uint8 or input_dtype == np.int8:
                scale, zero_point = input_details[0]["quantization"]
                quantized_input = np.round(
                    features.numpy() / scale + zero_point
                )
                features_input = quantized_input.astype(input_dtype)
            else:
                features_input = features.numpy().astype(np.float32)

            interpreter.set_tensor(
                input_details[0]["index"],
                features_input
            )
            interpreter.invoke()

            output_data = interpreter.get_tensor(
                output_details[0]["index"]
            )[0]

            if output_details[0]["dtype"] == np.uint8 or output_details[0]["dtype"] == np.int8:
                scale, zero_point = output_details[0]["quantization"]
                output_data = (
                    output_data.astype(np.float32) - zero_point
                ) * scale

            best_idx = np.argmax(output_data)
            confidence = output_data[best_idx] * 100.0
            predicted_label = CLASSES[best_idx]

            prob_str = " | ".join(
                f"{CLASSES[i]}: {output_data[i] * 100:.1f}%"
                for i in range(len(CLASSES))
            )

            if predicted_label == "hello_shravan" and confidence > 75.0:
                print(
                    f"*** DETECTED: {predicted_label.upper()} "
                    f"({confidence:.1f}%) ***  [{prob_str}]"
                )
            else:
                print(
                    f"Listening... Top: {predicted_label} "
                    f"({confidence:.1f}%)  [{prob_str}]"
                )

            time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
