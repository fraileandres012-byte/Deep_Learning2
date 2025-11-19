import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import soundfile as sf

SR = 22050
N_MELS = 128
FIXED_TIME_FRAMES = 431

def load_audio_wave(path, sr=SR, mono=True):
    try:
        data, srf = sf.read(path, always_2d=False)
        if data.ndim == 2 and mono:
            data = data.mean(axis=1)
        if srf != sr:
            data = librosa.resample(data.astype(np.float32), orig_sr=srf, target_sr=sr)
        return data.astype(np.float32), sr
    except Exception:
        y, _ = librosa.load(path, sr=sr, mono=mono)
        return y.astype(np.float32), sr

def wav_to_logmel_from_wave(y, sr=SR, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel

def pad_or_crop_mel(mel, target_frames):
    n_mels, n_frames = mel.shape
    if n_frames == target_frames:
        return mel
    elif n_frames > target_frames:
        start = (n_frames - target_frames) // 2
        return mel[:, start:start+target_frames]
    else:
        pad_width = target_frames - n_frames
        return np.pad(mel, ((0,0), (0, pad_width)), mode='constant')

def mel_figure(mel_data, sr=SR, n_mels=N_MELS, title="Mel-spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(mel_data, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    return fig

def gradcam_heatmap(model, x, layer_name, class_index, upsample_to):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(x)
        loss = preds[:, class_index]
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    # Normalización segura
    max_val = np.max(heatmap)
    if max_val == 0: max_val = 1e-8
    heatmap /= max_val
    
    # Redimensión con OpenCV o Scipy sería ideal, pero usamos zoom de ndimage o resize simple
    import cv2
    heatmap = cv2.resize(heatmap.numpy(), (upsample_to[1], upsample_to[0]))
    
    return heatmap

def overlay_gradcam_on_mel(mel_data, heatmap, labels, pred_idx):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(mel_data, sr=SR, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    ax.imshow(heatmap, cmap='jet', alpha=0.4, aspect='auto', 
              extent=[0, mel_data.shape[1]*512/SR, 0, SR/2], origin='lower')
    ax.set_title(f"Grad-CAM Attention: {labels[pred_idx]}")
    return fig
