import cv2
import numpy as np
import string

FRAME_WIDTH = 100
FRAME_HEIGHT = 50

def extract_mouth_frames(video_path, max_frames=75):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return frames

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ End of video or error at frame {count}")
            break

        h, w, _ = frame.shape
        x1, y1 = int(w * 0.35), int(h * 0.45)
        x2, y2 = int(w * 0.65), int(h * 0.65)

        mouth_roi = frame[y1:y2, x1:x2]

        if mouth_roi.size != 0:
            mouth_resized = cv2.resize(mouth_roi, (FRAME_WIDTH, FRAME_HEIGHT))
            frames.append(mouth_resized)
        else:
            print(f"⚠️ Skipping empty ROI at frame {count}")

        count += 1

    cap.release()
    return frames

# Character set and mappings
char_list = list("abcdefghijklmnopqrstuvwxyz ")  # 26 letters + space
char_to_idx = {char: idx + 1 for idx, char in enumerate(char_list)}  # start from 1
idx_to_char = {idx: char for char, idx in char_to_idx.items()}       # reverse mapping
# CTC requires blank token at index 0
idx_to_char[0] = ''  
# newutils.py

def text_to_labels(text, char_to_idx):
    """
    Converts a transcription text to a list of indices.
    
    Parameters:
        text (str): The transcription text to be converted.
        char_to_idx (dict): The character-to-index mapping.
    
    Returns:
        List of integers representing the transcription as indices.
    """
    return [char_to_idx[c] for c in text]

def ctc_loss_lambda_func(y_pred, labels, input_length, label_length):
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)