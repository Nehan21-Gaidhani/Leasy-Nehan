import cv2
import numpy as np
import tensorflow as tf
import os
from lipnetutils import extract_lip_frames
import matplotlib.pyplot as plt

# Define your trained word mapping
idx_to_word = {0: 'HELLO', 1: 'WORLD'}

def show_lip_frames(lip_frames, title="Lip Movement"):
    """Visualize 6 lip frames."""
    plt.figure(figsize=(12, 2))
    step = max(1, len(lip_frames) // 6)
    for i in range(min(6, len(lip_frames))):
        idx = i * step
        plt.subplot(1, 6, i + 1)
        plt.imshow(lip_frames[idx], cmap='gray')
        plt.title(f"Frame {idx}")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def predict_from_live(model_path, max_frames=30):
    print("🎥 Starting webcam...")
    cap = cv2.VideoCapture(0)

    frames = []
    while len(frames) < 35:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        cv2.imshow('Recording... Press Q to abort', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("📹 Clip captured.")

    # Save frames as a temporary video
    h, w = frames[0].shape[:2]
    temp_video_path = "temp_live_input.mp4"
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    for f in frames:
        out.write(f)
    out.release()

    # Extract lip frames using same pipeline
    lip_frames = extract_lip_frames(temp_video_path, max_frames=max_frames, crop_size=(100, 50))

    sequence = np.array(lip_frames).astype('float32') / 255.0
    sequence = np.expand_dims(sequence, axis=-1)
    sequence = np.expand_dims(sequence, axis=0)  # Shape: (1, 30, 50, 100, 1)

    # Load model and predict
    model = tf.keras.models.load_model(model_path)
    preds = model.predict(sequence)
    pred_class = np.argmax(preds[0])
    pred_word = idx_to_word.get(pred_class, "UNKNOWN")

    # Display result
    print(f"🧠 Predicted Word: {pred_word}")
    show_lip_frames(lip_frames, title=f"🗣️ Predicted: {pred_word}")

    # Cleanup
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

# Run prediction from webcam
if __name__ == "__main__":
    model_path = "lip_cnn_lstm_word_modelv2.keras"
    if not os.path.exists(model_path):
        print("❌ Model not found:", model_path)
    else:
        predict_from_live(model_path)
