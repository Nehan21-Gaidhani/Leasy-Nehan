from prepare_data import prepare_lip_data
from lipnetmodel import build_cnn_lstm_model
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare data
video_dir = "data/newdata"
label_csv = "data/labels1.csv"

X, y = prepare_lip_data(video_dir, label_csv, max_frames=30)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train model
model = build_cnn_lstm_model(input_shape=(30, 50, 100, 1), num_classes=2)
model.summary()

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=2)

# Save model
model.save("lip_cnn_lstm_word_modelv3.keras")
print("✅ Model saved to lip_cnn_lstm_word_modelv3.keras")
