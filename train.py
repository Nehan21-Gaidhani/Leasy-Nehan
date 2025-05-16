# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split

# # Parameters
# videos_path = 'videos/'
# frame_size = (64, 64)
# max_frames = 30  # all videos will be padded/truncated to this length

# def load_data():
#     sequences = []
#     labels = []
#     for label_name in os.listdir(videos_path):
#         label_folder = os.path.join(videos_path, label_name)
#         if not os.path.isdir(label_folder):
#             continue
#         for filename in os.listdir(label_folder):
#             if not filename.endswith(".mp4"):
#                 continue
#             filepath = os.path.join(label_folder, filename)
#             cap = cv2.VideoCapture(filepath)
#             frames = []
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frame = cv2.resize(frame, frame_size)
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 frames.append(gray)
#             cap.release()

#             # Pad or truncate to fixed length
#             if len(frames) < max_frames:
#                 padding = [np.zeros(frame_size, dtype=np.uint8)] * (max_frames - len(frames))
#                 frames.extend(padding)
#             else:
#                 frames = frames[:max_frames]

#             sequences.append(frames)
#             labels.append(label_name)
#     return np.array(sequences), labels

# def preprocess_data(sequences, labels):
#     # Convert to proper shape
#     X = np.array(sequences).astype("float32") / 255.0
#     X = np.expand_dims(X, -1)  # (num_samples, time, height, width, 1)

#     # Encode labels
#     le = LabelEncoder()
#     y = le.fit_transform(labels)
#     y = to_categorical(y)

#     return X, y, le

# def build_model(num_classes):
#     model = Sequential()
#     model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=(max_frames, frame_size[0], frame_size[1], 1)))
#     model.add(MaxPooling3D((1, 2, 2)))
#     model.add(Flatten())
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def main():
#     print("Loading and preprocessing data...")
#     sequences, labels = load_data()
#     X, y, le = preprocess_data(sequences, labels)

#     print(f"Classes: {le.classes_}")
#     print("Training model...")

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = build_model(num_classes=y.shape[1])
#     model.fit(X_train, y_train, epochs=100, batch_size=2, validation_data=(X_val, y_val))

#     model.save("lip_reader_model.h5")
#     print("Model saved as lip_reader_model.h5")

# if __name__ == "__main__":
#     main()

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Reshape, Dense, Bidirectional, LSTM, Lambda
from utils import extract_mouth_frames, text_to_labels, ctc_loss_lambda_func

# Set paths
DATA_PATH = 'data/videos'
LABELS_CSV = 'data/labels.csv'

# Characters used (uppercase only for simplicity)
CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '

# Create mappings
char_to_idx = {c: i for i, c in enumerate(CHARS)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
max_label_len = 10  # e.g., 'THANKYOU' = 8 chars

# Load labels
df = pd.read_csv(LABELS_CSV)
X_data, Y_data = [], []

print("Extracting frames...")
for i, row in df.iterrows():
    video_path = os.path.join(DATA_PATH, row['video'])
    word = row['label'].upper()
    frames = extract_mouth_frames(video_path)
    X_data.append(frames)
    Y_data.append(text_to_labels(word, char_to_idx))

X_data = np.array(X_data)
Y_data = tf.keras.preprocessing.sequence.pad_sequences(Y_data, maxlen=max_label_len, padding='post')

# Model input shapes
input_data = Input(name='input', shape=(75, 50, 100, 3))
labels = Input(name='labels', shape=[max_label_len], dtype='float32')
input_len = Input(name='input_length', shape=[1], dtype='int64')
label_len = Input(name='label_length', shape=[1], dtype='int64')

# Model architecture
x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_data)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)
x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)

x = Reshape((75, -1))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dense(len(CHARS), activation='softmax')(x)

# CTC
# CTC
# CTC loss layer
ctc_loss = Lambda(lambda args: ctc_loss_lambda_func(*args), output_shape=(1,), name='ctc')([x, labels, input_len, label_len])


model = Model(inputs=[input_data, labels, input_len, label_len], outputs=ctc_loss)

model.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: y_pred  # Dummy loss
)
model.summary()

# Train
input_lengths = np.ones((len(X_data), 1)) * 75
label_lengths = np.array([[len(label)] for label in Y_data])

model.fit(
    x=[X_data, Y_data, input_lengths, label_lengths],
    y=np.zeros(len(X_data)),
    batch_size=1,
    epochs=100
)

model.save('model/lip_model_v3.h5')
print("Model saved.")
