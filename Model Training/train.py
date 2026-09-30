import os
import glob
import random
import numpy as np
import tensorflow as tf


DATASET_DIR = "./dataset"

CLASSES = ["hello_shravan", "silence", "unknown"]
NUM_CLASSES = len(CLASSES)

SAMPLE_RATE = 8000
DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

FRAME_LENGTH = 240
FRAME_STEP = 160
FFT_LENGTH = 256
NUM_MEL_BINS = 40

BATCH_SIZE = 40


SAMPLES_PER_EPOCH_TARGET = 5000
EPOCHS = 60
LEARNING_RATE = 0.003
VAL_SPLIT = 0.2
LABEL_SMOOTHING = 0.1


CLASS_SAMPLE_WEIGHTS = {
    "hello_shravan": 0.20,
    "unknown": 0.45,
    "silence": 0.35,
}
assert abs(sum(CLASS_SAMPLE_WEIGHTS.values()) - 1.0) < 1e-6


RECOMMENDED_MIN_RATIO = {"hello_shravan": 1.0, "unknown": 1.5, "silence": 1.0}


# Dataset split and quality checks
def load_and_split_dataset():
    train_files = {}
    val_files = {}
    counts = {}

    for c in CLASSES:
        folder = os.path.join(DATASET_DIR, c)
        files = glob.glob(os.path.join(folder, "*.wav"))
        if len(files) == 0:
            raise ValueError(f"No .wav files found in directory: '{folder}'")

        random.seed(42)
        random.shuffle(files)

        split_idx = int(len(files) * (1.0 - VAL_SPLIT))
        train_files[c] = files[:split_idx]
        val_files[c] = files[split_idx:]
        counts[c] = len(files)

        print(f"Class '{c}': {len(train_files[c])} Train, {len(val_files[c])} Val")

    _check_dataset_ratio(counts)
    _check_silence_diversity(train_files["silence"])

    return train_files, val_files


def _check_dataset_ratio(counts):
    base = counts["hello_shravan"] / RECOMMENDED_MIN_RATIO["hello_shravan"]
    print("\n--- Dataset ratio check ---")
    for c in CLASSES:
        need = base * RECOMMENDED_MIN_RATIO[c]
        status = "OK" if counts[c] >= need else "LOW"
        print(f"  {c:15s}: {counts[c]:5d} files ({status}, recommended >= {need:.0f})")
    print("Raw counts are secondary here -- CLASS_SAMPLE_WEIGHTS above controls how "
          "often each class is actually seen in training, independent of these counts. "
          "If 'silence' is LOW, prioritize *variety* of background conditions over "
          "raw quantity: a handful more distinct noise environments helps more than "
          "duplicating the same room tone.\n")


def _check_silence_diversity(silence_files, sample_n=200):


    sample = random.sample(silence_files, min(sample_n, len(silence_files)))
    rms = []
    for f in sample:
        wav = decode_and_pad_wav(tf.constant(f)).numpy()
        rms.append(np.sqrt(np.mean(wav ** 2) + 1e-12))
    rms = np.array(rms)
    cv = rms.std() / (rms.mean() + 1e-9)
    print(f"--- Silence class energy diversity: mean RMS={rms.mean():.4f}, CV={cv:.2f} ---")
    if cv < 0.3:
        print("  WARNING: 'silence' clips look very uniform in loudness. Real-world "
              "disturbances will look nothing like this to the model. Add recordings "
              "of varied ambient noise, not just quiet room tone.\n")


# 1-second mono waveform at 8 kHz
def decode_and_pad_wav(file_path):
    file_contents = tf.io.read_file(file_path)
    wav, _ = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)

    length = tf.shape(wav)[0]
    if length < NUM_SAMPLES:
        padding = NUM_SAMPLES - length
        wav = tf.pad(wav, [[0, padding]])
    else:
        wav = wav[:NUM_SAMPLES]
    return wav


def augment_audio(wav, silence_files, class_name):

    shift = tf.random.uniform([], -800, 800, dtype=tf.int32)
    if shift > 0:
        wav = tf.concat([tf.zeros([shift]), wav[:-shift]], axis=0)
    elif shift < 0:
        wav = tf.concat([wav[-shift:], tf.zeros([-shift])], axis=0)


    gain = tf.random.uniform([], 0.7, 1.3)
    wav = wav * gain


    if class_name == "hello_shravan":
        noise_range = (0.02, 0.15)
        burst_prob = 0.15
    else:
        noise_range = (0.05, 0.45)
        burst_prob = 0.35

    if len(silence_files) > 0:
        random_idx = tf.random.uniform([], 0, len(silence_files), dtype=tf.int32)
        noise_path = tf.gather(silence_files, random_idx)
        noise_wav = decode_and_pad_wav(noise_path)
        noise_factor = tf.random.uniform([], noise_range[0], noise_range[1])
        wav = wav + (noise_wav * noise_factor)

    if tf.random.uniform([]) < burst_prob:
        burst = tf.random.normal([NUM_SAMPLES], mean=0.0, stddev=1.0)
        burst_gain = tf.random.uniform([], 0.03, 0.3)
        wav = wav + burst * burst_gain

    return tf.clip_by_value(wav, -1.0, 1.0)


def extract_log_mel_spectrogram(wav):
    stft = tf.signal.stft(
        wav,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH
    )
    spectrogram = tf.abs(stft)

    num_spectrogram_bins = FFT_LENGTH // 2 + 1
    linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        NUM_MEL_BINS, num_spectrogram_bins, SAMPLE_RATE, 80.0, 3800.0
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))

    log_mel = tf.math.log(mel_spectrogram + 1e-6)

    mean = tf.math.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel)
    normalized_log_mel = (log_mel - mean) / (std + 1e-6)

    return tf.expand_dims(normalized_log_mel, -1)


def apply_spec_augment(spectrogram):


    num_frames = tf.shape(spectrogram)[0]


    freq_mask_len = tf.random.uniform([], 0, 8, dtype=tf.int32)
    freq_start = tf.random.uniform([], 0, NUM_MEL_BINS - freq_mask_len, dtype=tf.int32)
    freq_mask = tf.concat([
        tf.ones([num_frames, freq_start, 1]),
        tf.zeros([num_frames, freq_mask_len, 1]),
        tf.ones([num_frames, NUM_MEL_BINS - freq_start - freq_mask_len, 1])
    ], axis=1)
    spectrogram = spectrogram * freq_mask


    max_time_mask = tf.maximum(tf.cast(num_frames, tf.float32) * 0.2, 1.0)
    time_mask_len = tf.random.uniform([], 0, tf.cast(max_time_mask, tf.int32) + 1, dtype=tf.int32)
    time_start = tf.random.uniform([], 0, num_frames - time_mask_len + 1, dtype=tf.int32)
    time_mask = tf.concat([
        tf.ones([time_start, NUM_MEL_BINS, 1]),
        tf.zeros([time_mask_len, NUM_MEL_BINS, 1]),
        tf.ones([num_frames - time_start - time_mask_len, NUM_MEL_BINS, 1])
    ], axis=0)
    spectrogram = spectrogram * time_mask

    return spectrogram


# tf.data pipelines and training-time augmentation
def build_class_dataset(file_list, class_id, class_name, silence_files_tensor, is_training=True):
    ds = tf.data.Dataset.from_tensor_slices(file_list)

    if is_training:
        ds = ds.shuffle(buffer_size=len(file_list)).repeat()

    def process_file(file_path):
        wav = decode_and_pad_wav(file_path)
        if is_training:
            wav = augment_audio(wav, silence_files_tensor, class_name)
        spectrogram = extract_log_mel_spectrogram(wav)
        if is_training:
            spectrogram = apply_spec_augment(spectrogram)
        return spectrogram, class_id

    return ds.map(process_file, num_parallel_calls=tf.data.AUTOTUNE)


def create_pipelines(train_files, val_files):
    silence_files_tensor = tf.constant(train_files["silence"])


    train_datasets = [
        build_class_dataset(train_files[c], CLASSES.index(c), c, silence_files_tensor, is_training=True)
        for c in CLASSES
    ]
    weights = [CLASS_SAMPLE_WEIGHTS[c] for c in CLASSES]
    train_ds = tf.data.Dataset.sample_from_datasets(
        train_datasets, weights=weights
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_datasets = [
        build_class_dataset(val_files[c], CLASSES.index(c), c, silence_files_tensor, is_training=False)
        for c in CLASSES
    ]
    val_ds = val_datasets[0]
    for v_ds in val_datasets[1:]:
        val_ds = val_ds.concatenate(v_ds)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    calib_datasets = [
        build_class_dataset(train_files[c], CLASSES.index(c), c, silence_files_tensor, is_training=False)
        for c in CLASSES
    ]
    calib_ds = tf.data.Dataset.sample_from_datasets(
        calib_datasets, weights=[1.0 / NUM_CLASSES] * NUM_CLASSES
    )

    return train_ds, val_ds, calib_ds


# Lightweight DS-CNN for TFLite Micro
def build_esp32_dscnn(input_shape):


    reg = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=reg)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ESP32_KWS_8KHZ")


# False-accept-rate and threshold evaluation
def _report_false_accept_rate(model, val_ds_eval):


    keyword_idx = CLASSES.index("hello_shravan")

    y_true, y_prob = [], []
    for specs, labels in val_ds_eval:
        probs = model.predict(specs, verbose=0)
        y_true.append(labels.numpy())
        y_prob.append(probs)
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred_argmax = np.argmax(y_prob, axis=1)

    print("Confusion matrix (rows=true, cols=pred), classes:", CLASSES)
    cm = tf.math.confusion_matrix(y_true, y_pred_argmax, num_classes=NUM_CLASSES).numpy()
    print(cm)

    negatives_mask = y_true != keyword_idx
    kw_mask = y_true == keyword_idx
    print("\nThreshold sweep on P(hello_shravan) -- accept only if prob >= threshold:")
    print(f"{'threshold':>10} {'FAR (neg->kw)':>15} {'recall (kw)':>12}")
    for thresh in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
        accepted = y_prob[:, keyword_idx] >= thresh
        far = np.mean(accepted[negatives_mask]) if negatives_mask.any() else float("nan")
        recall = np.mean(accepted[kw_mask]) if kw_mask.any() else float("nan")
        print(f"{thresh:>10.2f} {far:>15.4f} {recall:>12.4f}")

    print("\nPick the lowest threshold that gets FAR near 0 without destroying recall, "
          "then use that same threshold (not argmax) on-device, and additionally "
          "require N consecutive windows above it before firing the wake event.\n")


# Training and export
if __name__ == "__main__":
    print("--- 1. Splitting Datasets & Building Pipelines ---")
    train_files, val_files = load_and_split_dataset()
    train_ds, val_ds, calib_ds = create_pipelines(train_files, val_files)


    val_ds_eval = val_ds  # Keep sparse labels for FAR evaluation.
    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)))
    val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)))

    sample_wav = tf.zeros([NUM_SAMPLES])
    sample_spec = extract_log_mel_spectrogram(sample_wav)
    input_shape = sample_spec.shape
    print(f"Input Spectrogram Shape (8 kHz): {input_shape}")

    print("\n--- 2. Building ESP32 DS-CNN Model ---")
    model = build_esp32_dscnn(input_shape)

    steps_per_epoch = SAMPLES_PER_EPOCH_TARGET // BATCH_SIZE

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * steps_per_epoch,
        alpha=0.01
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "best_kws_model_8khz.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )


    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    print("\n--- 3. Starting Model Training ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    print("\nLoading best model weights for quantization...")
    best_model = tf.keras.models.load_model("best_kws_model_8khz.h5")

    print("\n--- 4. Validation report: confusion matrix + False-Accept-Rate ---")
    _report_false_accept_rate(best_model, val_ds_eval)

    print("\n--- 5. Quantizing to Full INT8 for TFLite Micro ---")
    def representative_dataset_gen():


        for specs, _ in calib_ds.batch(1).take(200):
            yield [tf.cast(specs, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    tflite_filename = "kws_model_8khz_int8.tflite"
    with open(tflite_filename, "wb") as f:
        f.write(tflite_quant_model)

    print(f"Quantized Model Size: {len(tflite_quant_model) / 1024:.2f} KB")

    print("\n--- 6. Exporting C Header File (model_data.h) ---")
    c_header_filename = "model_data.h"
    hex_array = ", ".join([f"0x{b:02x}" for b in tflite_quant_model])
    c_code = f"""// Auto-generated KWS Model Header for ESP32 / TFLite Micro (8 kHz)
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

const unsigned char g_model[] = {{ {hex_array} }};
const unsigned int g_model_len = {len(tflite_quant_model)};

#endif // MODEL_DATA_H
"""
    with open(c_header_filename, "w") as f:
        f.write(c_code)

    print(f"Export successful -> {c_header_filename}")
