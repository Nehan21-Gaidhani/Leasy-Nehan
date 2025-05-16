import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from utils import extract_mouth_frames, idx_to_char

model = load_model('model/lip_model_v2.h5', compile=False)

# Remove CTC lambda for inference
# infer_model = Model(inputs=model.get_layer('input').input,
#                     outputs=model.get_layer('dense').output)
# infer_model = Model(inputs=model.input,
#                     outputs=model.get_layer('dense').output)
for i, layer in enumerate(model.inputs):
    print(f"Input {i}: {layer.name}, shape: {layer.shape}")

# Assuming first input is video frames
video_input = model.inputs[0]
video_output = model.get_layer('dense').output  # Confirm layer name
infer_model = Model(inputs=video_input, outputs=video_output)

video_path = 'data/videos/hello1p.mp4'  # Your test input
frames = extract_mouth_frames(video_path)
frames = np.expand_dims(frames, axis=0)  # Add batch dimension

preds = infer_model.predict(frames)
decoded = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1])[0][0]
decoded = tf.keras.backend.get_value(decoded)

print("Predicted text:", ''.join([idx_to_char[c] for c in decoded[0] if c != -1]))
import matplotlib.pyplot as plt

# def extract_mouth_frames(video_path):
#     import cv2
#     import numpy as np

#     cap = cv2.VideoCapture(video_path)
#     frames = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Resize for consistency (optional)
#         frame = cv2.resize(frame, (100, 50))  # Resize to model input if needed

#         # Convert BGR to RGB for Keras
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # DEBUG: Draw a rectangle to simulate mouth ROI
#         h, w, _ = frame_rgb.shape
#         x1, y1, x2, y2 = int(w * 0.3), int(h * 0.6), int(w * 0.7), int(h * 0.9)
#         mouth_roi = frame_rgb[y1:y2, x1:x2]

#         # Resize mouth ROI to model input shape (50, 100)
#         mouth_resized = cv2.resize(mouth_roi, (100, 50))

#         frames.append(mouth_resized)

#     cap.release()

#     return np.array(frames)

import cv2
import numpy as np

def extract_mouth_frames(video_path, frame_count=75):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (100, 50))  # Resize entire frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, _ = frame_rgb.shape
        x1, y1 = int(w * 0.3), int(h * 0.6)
        x2, y2 = int(w * 0.7), int(h * 0.9)
        mouth_roi = frame_rgb[y1:y2, x1:x2]

        if mouth_roi.size == 0:
            continue  # Skip invalid crop

        mouth_resized = cv2.resize(mouth_roi, (100, 50))
        frames.append(mouth_resized)

    cap.release()

    # Pad if frames are fewer than required
    if len(frames) < frame_count:
        pad = np.zeros((frame_count - len(frames), 50, 100, 3), dtype=np.uint8)
        frames.extend(pad)

    return np.array(frames)

# Show a few extracted frames
frames = extract_mouth_frames(video_path)
print("Frames shape:", frames.shape)

# Visualize 5 frames to verify
import matplotlib.pyplot as plt
def show_frames(frames, num_frames=10):
    plt.figure(figsize=(15, 5))
    step = max(1, len(frames) // num_frames)
    for i in range(0, len(frames), step)[:num_frames]:
        plt.subplot(1, num_frames, i // step + 1)
        plt.imshow(frames[i].astype("uint8"))
        plt.axis('off')
        plt.title(f"Frame {i}")
    plt.tight_layout()
    plt.show()
show_frames(frames)