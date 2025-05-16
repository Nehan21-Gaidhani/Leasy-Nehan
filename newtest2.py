import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# ----- Load and Prepare Video -----
def preprocess_video(path, max_frames=75):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of video or error at frame", len(frames))
            break
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to (50, 100)
        frame = cv2.resize(frame, (100, 50))
        frames.append(frame)
    cap.release()

    # Pad or truncate to 75 frames
    frames = frames[:max_frames]
    while len(frames) < max_frames:
        frames.append(np.zeros_like(frames[0]))  # pad with black frames

    # Show frames for sanity check
    for i, f in enumerate(frames[:5]):
        cv2.imshow(f"Frame {i}", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        cv2.waitKey(300)
    cv2.destroyAllWindows()

    return np.array(frames) / 255.0  # normalize

# ----- Load Model -----
model = load_model("C:/Users/DELL/Desktop/Leasy Nehan/lip_reader/model/lip_model_final2.h5", compile=False)  # change name if needed
# model.summary()

# ----- Extract intermediate softmax layer -----
# Find correct output layer name
softmax_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Activation) and layer.activation == tf.keras.activations.softmax:
        softmax_layer_name = layer.name
        break

if softmax_layer_name is None:
    raise ValueError("❌ Couldn't find a softmax layer. Check model architecture.")

print(f"✅ Using softmax layer: {softmax_layer_name}")
softmax_output_layer = model.get_layer(softmax_layer_name).output
video_input_layer = model.input
inference_model = tf.keras.Model(inputs=video_input_layer, outputs=softmax_output_layer)

# ----- Prepare Input -----
video_path = "data/newdata/hello1n.mp4"  # update this to your input
X_test = preprocess_video(video_path)
X_test = np.expand_dims(X_test, axis=0)  # shape: (1, 75, 50, 100, 3)

# ----- Predict -----
y_pred = inference_model.predict(X_test)
print("✅ Prediction shape:", y_pred.shape)  # (1, 75, 28)

# ----- Visualize heatmap -----
plt.imshow(y_pred[0].T, cmap='hot', interpolation='nearest', aspect='auto')
plt.title("Prediction Probabilities")
plt.xlabel("Time Frame")
plt.ylabel("Character Index")
plt.colorbar()
plt.show()

# ----- Decode CTC -----
input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]

decoded, _ = K.ctc_decode(y_pred, input_length=input_len, greedy=True)
decoded_sequence = K.get_value(decoded[0])[0]  # (timesteps,)

print("Decoded indices:", decoded_sequence)

# ----- Map indices to characters -----
idx_to_char = {
    0: ' ', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g',
    8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o',
    16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v',
    23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: ''  # 27 is blank
}

predicted_text = ''.join([idx_to_char.get(i, '') for i in decoded_sequence if i != -1])
print("🗣️ Predicted text:", predicted_text if predicted_text else "⚠️ (Empty prediction)")
