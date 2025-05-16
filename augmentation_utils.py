import cv2
import numpy as np
import os
import random

def augment_lip_frames_and_save(frames, base_name, output_dir="augmented_videos", crop_size=(100, 50), fps=25):
    os.makedirs(output_dir, exist_ok=True)

    height, width = crop_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 1. Horizontal Flip
    flipped = [cv2.flip(f, 1) for f in frames]
    save_video(flipped, f"{base_name}_flip.mp4", output_dir, width, height, fps)

    # 2. Brightness Change
    factor = random.uniform(0.6, 1.4)
    bright = [np.clip(f * factor, 0, 255).astype(np.uint8) for f in frames]
    save_video(bright, f"{base_name}_bright.mp4", output_dir, width, height, fps)

    # 3. Temporal Jitter
    jitter = frames[::2]
    while len(jitter) < len(frames):
        jitter.append(np.zeros((height, width), dtype=np.uint8))
    jitter = jitter[:len(frames)]
    save_video(jitter, f"{base_name}_jitter.mp4", output_dir, width, height, fps)

    print(f"✅ Augmented versions saved for {base_name}")


def save_video(frames, filename, output_dir, width, height, fps):
    path = os.path.join(output_dir, filename)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    out.release()
