import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
LIP_LANDMARKS = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415,
    310, 311, 312, 13, 82, 81, 80, 191
]))

def extract_lip_frames(video_path, max_frames=30, crop_size=(100, 50)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                face_landmarks = results.multi_face_landmarks[0]
                lip_points = [(int(landmark.x * w), int(landmark.y * h))
                              for idx, landmark in enumerate(face_landmarks.landmark)
                              if idx in LIP_LANDMARKS]

                xs, ys = zip(*lip_points)
                x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, w)
                y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, h)

                try:
                    lip_crop = frame[y_min:y_max, x_min:x_max]
                    gray_lip = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray_lip, crop_size)  # Always resize to (100, 50)
                    frames.append(resized)
                    count += 1
                except:
                    continue  # Skip broken frame

    cap.release()

    # Padding if fewer than max_frames
    while len(frames) < max_frames:
        frames.append(np.zeros(crop_size, dtype=np.uint8))

    # Force uniform shape
    frames = [cv2.resize(f, crop_size) for f in frames]
    return frames

# import cv2
# import numpy as np
# import mediapipe as mp

# mp_face_mesh = mp.solutions.face_mesh
# LIP_LANDMARKS = list(set([
#     61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
#     317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415,
#     310, 311, 312, 13, 82, 81, 80, 191
# ]))

# def extract_lip_frames(video_path, max_frames=30, crop_size=(100, 50)):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     if total_frames == 0:
#         cap.release()
#         return [np.zeros(crop_size, dtype=np.uint8)] * max_frames

#     frame_indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
#     all_frames = []
#     current_frame = 0
#     idx_pointer = 0

#     with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
#         while idx_pointer < len(frame_indices):
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if current_frame == frame_indices[idx_pointer]:
#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = face_mesh.process(rgb)

#                 if results.multi_face_landmarks:
#                     h, w, _ = frame.shape
#                     face_landmarks = results.multi_face_landmarks[0]
#                     lip_points = [(int(landmark.x * w), int(landmark.y * h))
#                                   for idx, landmark in enumerate(face_landmarks.landmark)
#                                   if idx in LIP_LANDMARKS]

#                     if lip_points:
#                         xs, ys = zip(*lip_points)
#                         x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, w)
#                         y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, h)

#                         lip_crop = frame[y_min:y_max, x_min:x_max]
#                         if lip_crop.size != 0:
#                             gray_lip = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
#                             resized = cv2.resize(gray_lip, crop_size)
#                             if resized.shape == crop_size[::-1]:  # (height, width)
#                                 all_frames.append(resized)
#                             else:
#                                 all_frames.append(np.zeros(crop_size, dtype=np.uint8))
#                         else:
#                             all_frames.append(np.zeros(crop_size, dtype=np.uint8))
#                     else:
#                         all_frames.append(np.zeros(crop_size, dtype=np.uint8))
#                 else:
#                     all_frames.append(np.zeros(crop_size, dtype=np.uint8))

#                 idx_pointer += 1

#             current_frame += 1

#     cap.release()

#     while len(all_frames) < max_frames:
#         all_frames.append(np.zeros(crop_size, dtype=np.uint8))

#     return all_frames



 