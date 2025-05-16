import os
import cv2
import numpy as np
import pandas as pd
from lipnetutils import extract_lip_frames

def prepare_lip_data(video_dir, label_csv, max_frames=30, frame_size=( 50,100)):
    df = pd.read_csv(label_csv)
    word_to_idx = {'HELLO': 0, 'WORLD': 1}

    X, y = [], []

    for _, row in df.iterrows():
        video_path = os.path.join(video_dir, row['video'])
        label = row['label'].strip().upper()

        if label not in word_to_idx:
            continue

        lip_frames = extract_lip_frames(video_path, max_frames=max_frames, crop_size=frame_size)
        for i, frame in enumerate(lip_frames):
            if frame.shape != (50, 100):
                print(f"⚠️ Frame {i} has shape {frame.shape} instead of (50, 100)")

        clean_frames = [cv2.resize(f, (100, 50)) if f.shape != (50, 100) else f for f in lip_frames]
        sequence = np.array(clean_frames).astype('float32') / 255.0

        sequence = np.expand_dims(sequence, axis=-1)  # shape: (frames, H, W, 1)

        X.append(sequence)
        y.append(word_to_idx[label])

    return np.array(X), np.array(y)
