# import cv2
# video_path = "data/newdata/hello1n.mp4"


# # def extract_frames(video_path, resize=(100, 50), max_frames=75):
# #     """
# #     Extracts grayscale frames from a video, resizes, and returns a list.
# #     """
# #     cap = cv2.VideoCapture(video_path)
# #     frames = []

# #     while len(frames) < max_frames:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         resized = cv2.resize(gray, resize)
# #         frames.append(resized)

# #     cap.release()

# #     # Pad if fewer than max_frames
# #     if len(frames) < max_frames:
# #         pad = [np.zeros(resize, dtype=np.uint8)] * (max_frames - len(frames))
# #         frames.extend(pad)

# #     return frames  # list of (H, W)




# # import matplotlib.pyplot as plt
# # import numpy as np

# # def visualize_frames(frames, step=5, title="Sampled Frames"):
# #     """
# #     Visualize 6 evenly spaced frames from the list.
# #     """
# #     plt.figure(figsize=(12, 2))
# #     for j in range(6):
# #         idx = j * step
# #         if idx >= len(frames):
# #             break
# #         plt.subplot(1, 6, j + 1)
# #         plt.imshow(frames[idx], cmap='gray')
# #         plt.title(f'Frame {idx}')
# #         plt.axis('off')
# #     plt.suptitle(title)
# #     plt.tight_layout()
# #     plt.show()

# # frames = extract_frames(video_path)
# # visualize_frames(frames, step=5, title="Hello1n Frames")







# import cv2
# import numpy as np
# import mediapipe as mp

# mp_face_mesh = mp.solutions.face_mesh

# # Mouth landmark indices (lower + upper lips)
# LIP_LANDMARKS = list(set([
#     61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
#     317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 191
# ]))

# def extract_lip_frames(video_path, max_frames=30, crop_size=(100, 50)):
#     """
#     Extract cropped lip regions from the video using MediaPipe FaceMesh.
#     Returns a list of cropped grayscale lip images.
#     """
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     count = 0

#     with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
#         while count < max_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(rgb)

#             if results.multi_face_landmarks:
#                 h, w, _ = frame.shape
#                 face_landmarks = results.multi_face_landmarks[0]
#                 lip_points = [(int(landmark.x * w), int(landmark.y * h)) 
#                               for idx, landmark in enumerate(face_landmarks.landmark) if idx in LIP_LANDMARKS]

#                 xs, ys = zip(*lip_points)
#                 x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, w)
#                 y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, h)

#                 # Crop and resize
#                 lip_crop = frame[y_min:y_max, x_min:x_max]
#                 gray_lip = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
#                 resized = cv2.resize(gray_lip, crop_size)

#                 frames.append(resized)
#                 count += 1

#     cap.release()

#     return frames

# def visualize_frames(frames, step=1, title="Lip Frames"):
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(12, 2))
#     for j in range(min(6, len(frames))):
#         plt.subplot(1, 6, j + 1)
#         plt.imshow(frames[j * step], cmap='gray')
#         plt.title(f'Frame {j * step}')
#         plt.axis('off')
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()


# lip_frames = extract_lip_frames("data/testvideo/sampworld.mp4", max_frames=30)
# # visualize_frames(lip_frames, step=5)
# print("Extracted lip frames:", len(lip_frames))

# import matplotlib.pyplot as plt

# def show_all_lip_frames(frames, rows=5):
#     """
#     Display all extracted lip frames in a grid layout.

#     Args:
#         frames (List): List of grayscale lip frames.
#         rows (int): Number of rows in the plot grid.
#     """
#     total = len(frames)
#     cols = int(np.ceil(total / rows))

#     plt.figure(figsize=(cols * 2, rows * 2))
#     for idx, frame in enumerate(frames):
#         plt.subplot(rows, cols, idx + 1)
#         plt.imshow(frame, cmap='gray')
#         plt.title(f"Frame {idx}")
#         plt.axis('off')

#     plt.tight_layout()
#     plt.show()
# show_all_lip_frames(lip_frames, rows=5)



from augmentation_utils import augment_lip_frames_and_save
from lipnetutils import extract_lip_frames

video_path = "data/newdata/hello1n.mp4"
lip_frames = extract_lip_frames(video_path, max_frames=30)

augment_lip_frames_and_save(lip_frames, base_name="hello1n")
