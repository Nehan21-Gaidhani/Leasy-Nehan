import cv2
import numpy as np
import mediapipe as mp
import os
import tensorflow as tf
from keras import backend as K

# Initialize the MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh

# utils.py
import cv2
import numpy as np

def extract_mouth_frames(video_path, frame_count=75):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return None

    while count < frame_count:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of video or read error.")
            break

        # Fixed mouth region (simulated based on proportion of the frame)
        h, w, _ = frame.shape
        x1, y1 = int(w * 0.3), int(h * 0.6)
        x2, y2 = int(w * 0.7), int(h * 0.9)

        if y2 > y1 and x2 > x1:
            mouth_roi = frame[y1:y2, x1:x2]

            if mouth_roi.size == 0:
                print(f"⚠️ Skipping empty ROI at frame {count}")
                continue

            mouth_resized = cv2.resize(mouth_roi, (100, 50))  # Resize to model input
            mouth_rgb = cv2.cvtColor(mouth_resized, cv2.COLOR_BGR2RGB)

            frames.append(mouth_rgb)
            count += 1
        else:
            print(f"⚠️ Invalid ROI at frame {count}")

    cap.release()

    # Pad with black frames if needed
    if len(frames) < frame_count:
        pad = np.zeros((frame_count - len(frames), 50, 100, 3), dtype=np.uint8)
        frames = np.concatenate([frames, pad], axis=0)

    frames = np.array(frames).astype(np.float32) / 255.0

    return frames







def text_to_labels(text, char_to_idx):
    """
    Convert text labels to numeric indices based on a character-to-index dictionary.
    
    Args:
    text (str): Text string to convert.
    char_to_idx (dict): Dictionary mapping characters to indices.
    
    Returns:
    list: List of indices corresponding to each character in the text.
    """
    return [char_to_idx[c] for c in text]

def ctc_loss_lambda_func(y_pred, labels, input_length, label_length):
    """
    Compute the CTC loss for a batch of sequences.
    
    Args:
    y_pred (tensor): Predictions from the model.
    labels (tensor): Ground truth labels.
    input_length (tensor): Lengths of the input sequences.
    label_length (tensor): Lengths of the label sequences.
    
    Returns:
    tensor: The computed CTC loss value.
    """
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)



import string
chars = string.ascii_lowercase + ' '  # 'abcdefghijklmnopqrstuvwxyz '
idx_to_char = {i: c for i, c in enumerate(chars)}

    
