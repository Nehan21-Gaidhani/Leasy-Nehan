import numpy as np
import tensorflow as tf
from newutils import extract_mouth_frames
from tensorflow.keras.models import load_model

MAX_FRAMES = 75
FRAME_WIDTH = 100
FRAME_HEIGHT = 50
CHANNELS = 3

# Load the model
model = load_model("model/lip_model_final.h5")

# Path to test video
video_path = "data/newdata/world2n.mp4"

# Extract frames
frames = extract_mouth_frames(video_path, max_frames=MAX_FRAMES)

# Pad or truncate
if len(frames) < MAX_FRAMES:
    pad_size = MAX_FRAMES - len(frames)
    black_frame = np.zeros_like(frames[0])
    frames += [black_frame] * pad_size
elif len(frames) > MAX_FRAMES:
    frames = frames[:MAX_FRAMES]

X_test = np.array([frames]) / 255.0  # Add batch dim & normalize

# Predict
pred = model.predict(X_test)
predicted_class = np.argmax(pred, axis=-1)[0]

# Map class index to label
index_to_word = {0: "hello", 1: "world"}  # Adjust if label-to-index mapping differs
print("🗣️ Predicted word:", index_to_word[predicted_class])
