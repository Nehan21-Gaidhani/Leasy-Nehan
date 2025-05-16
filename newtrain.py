import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, TimeDistributed, Conv2D, MaxPooling2D,
                                     Flatten, LSTM, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from newutils import extract_mouth_frames, char_to_idx

# Constants
MAX_FRAMES = 75
FRAME_WIDTH = 100
FRAME_HEIGHT = 50
CHANNELS = 3
DATASET_DIR = "data/newdata"
CSV_PATH = "data/labels1.csv"
MAX_LABEL_LEN = 20

# Load labels
df = pd.read_csv(CSV_PATH)
print("✅ Loaded labels:", df.shape)

# Build dataset
X, y = [], []

# Create a dictionary for word-to-label mapping
word_to_label = {"hello": 0, "world": 1}

for idx, row in df.iterrows():
    filename = row['video']
    label = row['label'].lower()
    video_path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(video_path):
        print(f"⚠️ Skipping missing file: {video_path}")
        continue

    frames = extract_mouth_frames(video_path, max_frames=MAX_FRAMES)

    # Pad or truncate frames
    if len(frames) < MAX_FRAMES:
        pad_size = MAX_FRAMES - len(frames)
        black_frame = np.zeros_like(frames[0])
        frames += [black_frame] * pad_size
    elif len(frames) > MAX_FRAMES:
        frames = frames[:MAX_FRAMES]

    X.append(frames)

    # Convert labels to integer values (0 for "hello", 1 for "world")
    if label in word_to_label:
        y.append(word_to_label[label])
    else:
        print(f"⚠️ Skipping unsupported label: {label}")
        continue

# Convert X and y to NumPy arrays and normalize
X = np.array(X) / 255.0
y = np.array(y)

# Ensure your input data is NumPy array and check the shapes
print("X shape:", X.shape)
print("y shape:", y.shape)

# Build the model
input_layer = Input(shape=(MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))

x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(input_layer)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(128, return_sequences=False)(x)  # We only need the final output of the sequence
x = Dense(2, activation='softmax')(x)  # 2 units for "hello" and "world"

model = Model(inputs=input_layer, outputs=x)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
model.summary()

# Save best model
os.makedirs("model", exist_ok=True)
checkpoint = ModelCheckpoint("model/lip_model_new.h5", save_best_only=True, monitor='val_loss', mode='min')

# Train the model
model.fit(X, y,
          epochs=20,
          batch_size=1,
          validation_split=0.1,  # Using 10% for validation
          callbacks=[checkpoint])

# Save final model
model.save("model/lip_model_final.h5")
print("✅ Final model saved to model/lip_model_final.h5")
