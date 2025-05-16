import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from newutils2 import extract_mouth_frames, text_to_labels, ctc_loss_lambda_func

# Constants
MAX_FRAMES = 75
FRAME_WIDTH = 100
FRAME_HEIGHT = 50
CHANNELS = 3
MAX_LABEL_LEN = 20

# Paths
DATASET_DIR = "data/newdata"
CSV_PATH = "data/labels1.csv"

# Load labels
df = pd.read_csv(CSV_PATH)
print("✅ Loaded labels:", df.shape)

# Dataset containers
X, Y, input_lengths, label_lengths = [], [], [], []

for idx, row in df.iterrows():
    filename = row['video']
    label_text = row['label']
    video_path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(video_path):
        print(f"⚠️ Skipping missing file: {video_path}")
        continue

    frames = extract_mouth_frames(video_path, max_frames=MAX_FRAMES)
    if len(frames) == 0:
        continue

    # Pad frames
    if len(frames) < MAX_FRAMES:
        pad_size = MAX_FRAMES - len(frames)
        black_frame = np.zeros_like(frames[0])
        frames += [black_frame] * pad_size
    elif len(frames) > MAX_FRAMES:
        frames = frames[:MAX_FRAMES]

    X.append(frames)

    label_seq = text_to_labels(label_text)
    label_lengths.append([len(label_seq)])
    input_lengths.append([MAX_FRAMES])  # Full sequence input

    # Pad labels
    label_seq += [0] * (MAX_LABEL_LEN - len(label_seq))
    Y.append(label_seq)

X = np.array(X) / 255.0
Y = np.array(Y)
input_lengths = np.array(input_lengths)
label_lengths = np.array(label_lengths)

# Shapes
print("X shape:", X.shape)  # (samples, 75, 50, 100, 3)
print("Y shape:", Y.shape)

# Model definition
video_input = Input(shape=(MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS), name='video_input')
x = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(video_input)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(128, return_sequences=True)(x)
x = Dense(len("abcdefghijklmnopqrstuvwxyz ") + 1, activation='softmax')(x)

# Inputs for CTC
labels = Input(name='labels', shape=(MAX_LABEL_LEN,), dtype='int32')
input_len = Input(name='input_length', shape=(1,), dtype='int32')
label_len = Input(name='label_length', shape=(1,), dtype='int32')

# CTC loss
loss_out = Lambda(ctc_loss_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_len, label_len])

model = Model(inputs=[video_input, labels, input_len, label_len], outputs=loss_out)
model.compile(optimizer=Adam(1e-4), loss={'ctc': lambda y_true, y_pred: y_pred})
model.summary()

# Save model
# os.makedirs("model", exist_ok=True)
checkpoint = ModelCheckpoint("model/lip_model_final2.h5", monitor='val_loss', save_best_only=True)

# Dummy y_dummy since CTC is calculated inside Lambda
y_dummy = np.zeros((X.shape[0],))

# Train model
model.fit(
    x=[X, Y, input_lengths, label_lengths],
    y=y_dummy,
    batch_size=1,
    epochs=20,
    validation_split=0.1,
    callbacks=[checkpoint]
)

print("✅ Training complete. Final model saved.")
