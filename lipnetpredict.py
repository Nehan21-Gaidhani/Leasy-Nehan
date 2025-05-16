import numpy as np
import sys
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from lipnetutils import extract_lip_frames
from lipnetmodel import build_cnn_lstm_model

# Map label index to word
idx_to_word = {0: 'HELLO', 1: 'WORLD'}

def predict_from_video(model_path, video_path, max_frames=30):
    # Extract lip frames
    lip_frames = extract_lip_frames(video_path, max_frames=max_frames, crop_size=(100, 50))

    # Prepare input for model
            # lip_frames = extract_lip_frames(video_path, max_frames=max_frames, crop_size=frame_size)
    for i, frame in enumerate(lip_frames):
        if frame.shape != (50, 100):
           print(f"⚠️ Frame {i} has shape {frame.shape} instead of (50, 100)")

    clean_frames = [cv2.resize(f, (100, 50)) if f.shape != (50, 100) else f for f in lip_frames]
    sequence = np.array(lip_frames).astype('float32') / 255.0
    sequence = np.expand_dims(sequence, axis=-1)  # (frames, H, W, 1)
    sequence = np.expand_dims(sequence, axis=0)   # (1, frames, H, W, 1)

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Predict
    predictions = model.predict(sequence)
    predicted_class = np.argmax(predictions[0])
    predicted_word = idx_to_word.get(predicted_class, "UNKNOWN")

    return predicted_word, lip_frames


def show_lip_frames(frames, title="Predicted Lip Movement",rows=5):
    total = len(frames)
    cols = int(np.ceil(total / rows))

    plt.figure(figsize=(cols * 2, rows * 2))
    for idx, frame in enumerate(frames):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(frame, cmap='gray')
        plt.title(f"Frame {idx}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# ---------- Run as Script ----------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_word.py path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = "lip_cnn_lstm_word_modelv3.keras"

    if not os.path.exists(video_path):
        print("❌ Video not found:", video_path)
        sys.exit(1)

    if not os.path.exists(model_path):
        print("❌ Trained model not found:", model_path)
        sys.exit(1)

    # Predict and visualize
    predicted_word, lip_frames = predict_from_video(model_path, video_path)
    print(f"🗣️ Predicted word: {predicted_word}")
    show_lip_frames(lip_frames, title=f"🗣️ Predicted: {predicted_word}",rows=5)
